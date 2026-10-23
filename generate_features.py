from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

BASE = Path(__file__).parent
DATA = BASE / "data"

df = pd.read_csv(DATA / "bace_validated.csv")

# Radius 2 corresponds to ECFP4-style local environments.
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)

fingerprints = []
properties = []

for smiles in df["smiles"]:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Could not parse validated SMILES: {smiles}")

    fingerprint = generator.GetFingerprint(mol)
    array = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    fingerprints.append(array)

    properties.append({
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        # Group shared ring frameworks together for scaffold splitting.
        "scaffold": MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        ),
    })

X = np.asarray(fingerprints, dtype=np.uint8)
y = df["pIC50"].to_numpy(dtype=float)

metadata = pd.concat(
    [df.reset_index(drop=True), pd.DataFrame(properties)],
    axis=1,
)

# Save features and their corresponding row identifiers together.
np.savez_compressed(
    DATA / "bace_features.npz",
    X=X,
    y=y,
    source_row=df["source_row"].to_numpy(),
)

metadata.to_csv(DATA / "bace_metadata.csv", index=False)

print("Fingerprint matrix:", X.shape)
print("Activity values:", y.shape)
print("Unique scaffolds:", metadata["scaffold"].nunique())
print("Molecules with an empty scaffold:",
      metadata["scaffold"].eq("").sum())

print("\nLargest scaffold groups:")
print(metadata["scaffold"].value_counts().head().to_string())

print("\nFeatures and metadata saved.")