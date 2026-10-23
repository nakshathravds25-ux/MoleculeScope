from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

BASE = Path(__file__).parent
DATA = BASE / "data"
RESULTS = BASE / "results"

metadata = pd.read_csv(
    DATA / "bace_metadata.csv",
    keep_default_na=False,
)

with np.load(DATA / "bace_splits.npz") as splits:
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]

predictions = pd.read_csv(
    RESULTS / "validation_predictions.csv"
)

assert np.array_equal(
    predictions["row_index"].to_numpy(), val_idx
), "Validation rows are not aligned."

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)

fingerprints = [
    generator.GetFingerprint(Chem.MolFromSmiles(smiles))
    for smiles in metadata["smiles"]
]

train_fingerprints = [
    fingerprints[i] for i in train_idx
]

nearest_similarity = []
nearest_training_row = []

for i in val_idx:
    similarities = np.asarray(
        DataStructs.BulkTanimotoSimilarity(
            fingerprints[i], train_fingerprints
        )
    )

    nearest_position = int(similarities.argmax())

    nearest_similarity.append(similarities[nearest_position])
    nearest_training_row.append(train_idx[nearest_position])

predictions["nearest_train_similarity"] = nearest_similarity
predictions["nearest_training_row"] = nearest_training_row

predictions["absolute_error"] = (
    predictions["actual_pIC50"] - predictions["Random Forest"]
).abs()

# Exploratory similarity bands, not validated reliability thresholds.
predictions["similarity_band"] = pd.cut(
    predictions["nearest_train_similarity"],
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=["0–0.3", "0.3–0.5", "0.5–0.7", "0.7–1.0"],
    include_lowest=True,
)

summary = (
    predictions.groupby("similarity_band", observed=True)
    .agg(
        molecules=("absolute_error", "size"),
        MAE=("absolute_error", "mean"),
        median_error=("absolute_error", "median"),
    )
    .reset_index()
)

predictions.to_csv(
    RESULTS / "validation_applicability.csv", index=False
)
summary.to_csv(
    RESULTS / "validation_similarity_summary.csv", index=False
)

print("Nearest-training similarity distribution:")
print(
    predictions["nearest_train_similarity"]
    .describe()
    .round(3)
    .to_string()
)

print("\nRandom Forest error by similarity band:")
print(summary.round(4).to_string(index=False))

print("\nExploratory results only.")
print("Similarity is not a probability of correctness.")
print("Test set remains untouched.")