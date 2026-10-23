from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

BASE = Path(__file__).parent
DATA = BASE / "data"
RESULTS = BASE / "results"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

with np.load(DATA / "bace_features.npz") as features:
    X = features["X"].astype(np.float32)
    y = features["y"]
    source_rows = features["source_row"]

with np.load(DATA / "bace_splits.npz") as splits:
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

metadata = pd.read_csv(
    DATA / "bace_metadata.csv",
    keep_default_na=False,
)

assert np.array_equal(
    metadata["source_row"].to_numpy(), source_rows
), "Metadata and features are not aligned."

model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    max_features=0.5,
    random_state=42,
    n_jobs=2,
)

model.fit(X[train_idx], y[train_idx])
predicted = model.predict(X[test_idx])

# Compare against the training-mean baseline.
baseline = np.full(len(test_idx), y[train_idx].mean())

def metrics(actual, estimates):
    return {
        "MAE": float(mean_absolute_error(actual, estimates)),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, estimates))
        ),
        "R2": float(r2_score(actual, estimates)),
    }

scores = {
    "Mean baseline": metrics(y[test_idx], baseline),
    "Random Forest": metrics(y[test_idx], predicted),
}

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)

train_metadata = metadata.iloc[train_idx].copy()
train_fps = [
    generator.GetFingerprint(Chem.MolFromSmiles(s))
    for s in train_metadata["smiles"]
]

similarities = []

for smiles in metadata.iloc[test_idx]["smiles"]:
    fp = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
    similarities.append(
        max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    )

test_results = pd.DataFrame({
    "source_row": source_rows[test_idx],
    "smiles": metadata.iloc[test_idx]["smiles"].to_numpy(),
    "actual_pIC50": y[test_idx],
    "predicted_pIC50": predicted,
    "nearest_train_similarity": similarities,
})

test_results["absolute_error"] = (
    test_results["actual_pIC50"]
    - test_results["predicted_pIC50"]
).abs()

test_results["similarity_band"] = pd.cut(
    test_results["nearest_train_similarity"],
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=["0–0.3", "0.3–0.5", "0.5–0.7", "0.7–1.0"],
    include_lowest=True,
)

summary = (
    test_results.groupby("similarity_band", observed=True)
    .agg(
        molecules=("absolute_error", "size"),
        MAE=("absolute_error", "mean"),
    )
    .reset_index()
)

joblib.dump(model, MODELS / "random_forest.joblib")

# Only training molecules will provide reference neighbours in the app.
train_metadata.to_csv(
    DATA / "training_reference.csv", index=False
)

test_results.to_csv(
    RESULTS / "test_predictions.csv", index=False
)
summary.to_csv(
    RESULTS / "test_similarity_summary.csv", index=False
)

with open(RESULTS / "test_metrics.json", "w") as file:
    json.dump(scores, file, indent=2)

print("Final test results:")
print(pd.DataFrame(scores).T.round(4).to_string())

print("\nTest error by similarity band:")
print(summary.round(4).to_string(index=False))

print("\nSaved model and training reference.")
print("Keep model settings fixed after reviewing these results.")