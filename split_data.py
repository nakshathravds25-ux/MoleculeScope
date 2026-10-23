from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path(__file__).parent
DATA = BASE / "data"

metadata = pd.read_csv(
    DATA / "bace_metadata.csv",
    keep_default_na=False,
)

with np.load(DATA / "bace_features.npz") as features:
    source_rows = features["source_row"]

# Confirm metadata and fingerprint rows are aligned.
assert np.array_equal(
    metadata["source_row"].to_numpy(),
    source_rows,
), "Metadata and features are not aligned."

groups = metadata["scaffold"].to_numpy()

# Hold out test scaffolds first.
outer_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42,
)

development_idx, test_idx = next(
    outer_split.split(metadata, groups=groups)
)

# Reserve part of the remaining scaffolds for validation.
inner_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15 / 0.85,
    random_state=43,
)

train_local, val_local = next(
    inner_split.split(
        metadata.iloc[development_idx],
        groups=groups[development_idx],
    )
)

train_idx = development_idx[train_local]
val_idx = development_idx[val_local]

splits = {
    "train": train_idx,
    "validation": val_idx,
    "test": test_idx,
}

# Check that no scaffold appears in more than one split.
scaffold_sets = {
    name: set(groups[idx])
    for name, idx in splits.items()
}

assert scaffold_sets["train"].isdisjoint(
    scaffold_sets["validation"]
)
assert scaffold_sets["train"].isdisjoint(
    scaffold_sets["test"]
)
assert scaffold_sets["validation"].isdisjoint(
    scaffold_sets["test"]
)

# Check that every molecule appears exactly once.
all_indices = np.concatenate(list(splits.values()))
assert len(all_indices) == len(metadata)
assert len(np.unique(all_indices)) == len(metadata)

np.savez(
    DATA / "bace_splits.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
)

metadata["split"] = ""

for name, idx in splits.items():
    metadata.loc[idx, "split"] = name
    print(
        f"{name}: {len(idx)} molecules "
        f"({len(idx) / len(metadata):.1%}), "
        f"{len(scaffold_sets[name])} scaffolds"
    )

metadata.to_csv(DATA / "bace_split_metadata.csv", index=False)

print("\nScaffold overlap: 0")
print("Every molecule assigned exactly once.")
print("Split files saved.")