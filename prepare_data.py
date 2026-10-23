from pathlib import Path
import pandas as pd
from rdkit import Chem

BASE = Path(__file__).parent
DATA = BASE / "data"

df = pd.read_csv(DATA / "bace.csv")
work = df[["mol", "pIC50"]].copy()
work.insert(0, "source_row", df.index + 1)

# Convert activity values to numbers.
work["pIC50"] = pd.to_numeric(work["pIC50"], errors="coerce")


def canonicalize(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None or molecule.GetNumAtoms() == 0:
        return None

    # Preserve stereochemistry because it can affect activity.
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True
    )


work["smiles"] = work["mol"].apply(canonicalize)

valid_activity = (
    work["pIC50"].notna()
    & ~work["pIC50"].isin([float("inf"), float("-inf")])
)

valid_rows = work["smiles"].notna() & valid_activity
rejected = work.loc[~valid_rows].copy()
valid = work.loc[valid_rows].copy()

# Save all repeated canonical structures for inspection.
duplicates = valid[
    valid.duplicated("smiles", keep=False)
].sort_values("smiles")

# Summarise repeated measurements without silently discarding them.
summary = valid.groupby("smiles", sort=False).agg(
    measurement_count=("pIC50", "size"),
    minimum_pIC50=("pIC50", "min"),
    maximum_pIC50=("pIC50", "max"),
).reset_index()

summary["activity_range"] = (
    summary["maximum_pIC50"] - summary["minimum_pIC50"]
)

valid.to_csv(DATA / "bace_validated.csv", index=False)
rejected.to_csv(DATA / "bace_rejected.csv", index=False)
duplicates.to_csv(DATA / "bace_duplicate_review.csv", index=False)
summary.to_csv(DATA / "bace_structure_summary.csv", index=False)

print("Original rows:", len(work))
print("Valid rows:", len(valid))
print("Rejected rows:", len(rejected))
print("Unique canonical structures:", valid["smiles"].nunique())
print("Extra rows sharing a canonical structure:",
      valid.duplicated("smiles").sum())
print("Structures with multiple measurements:",
      (summary["measurement_count"] > 1).sum())
print("Structures with multiple fragments:",
      valid["smiles"].str.contains(".", regex=False).sum())

if not duplicates.empty:
    print("\nRepeated-structure activity summary:")
    print(summary[summary["measurement_count"] > 1].to_string(index=False))

print("\nValidation files saved in the data folder.")