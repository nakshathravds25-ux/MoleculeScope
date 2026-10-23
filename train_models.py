from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

with np.load(DATA / "bace_features.npz") as features:
    X = features["X"].astype(np.float32)
    y = features["y"]

with np.load(DATA / "bace_splits.npz") as splits:
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Ridge": Ridge(alpha=10.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=42,
        n_jobs=2,
    ),
}

scores = []
predictions = pd.DataFrame({
    "row_index": val_idx,
    "actual_pIC50": y_val,
})

for name, model in models.items():
    start = time.perf_counter()

    model.fit(X_train, y_train)
    predicted = model.predict(X_val)

    scores.append({
        "model": name,
        "MAE": mean_absolute_error(y_val, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_val, predicted)),
        "R2": r2_score(y_val, predicted),
        "training_seconds": time.perf_counter() - start,
    })

    predictions[name] = predicted
    print(f"Finished: {name}")

scores = pd.DataFrame(scores).sort_values("MAE")

scores.to_csv(RESULTS / "validation_metrics.csv", index=False)
predictions.to_csv(
    RESULTS / "validation_predictions.csv", index=False
)

print("\nValidation results:")
print(scores.round(4).to_string(index=False))
print("\nLower MAE and RMSE are better; higher R2 is better.")
print("Test set has not been evaluated.")