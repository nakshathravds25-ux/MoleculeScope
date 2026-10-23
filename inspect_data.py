from pathlib import Path
import pandas as pd

data_path = Path(__file__).parent / "data" / "bace.csv"
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nFirst five records:")
print(df[["mol", "pIC50"]].head())

print("\nMissing values:")
print(df[["mol", "pIC50"]].isna().sum())

print("\nRepeated SMILES:", df["mol"].duplicated().sum())

print("\nActivity summary:")
print(df["pIC50"].describe())