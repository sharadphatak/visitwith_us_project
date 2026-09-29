
from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism.csv"

EXPECTED_COLUMNS = {
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome"
}

def main():
    df = pd.read_csv(DATA_PATH)

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    if df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be unique.")

    if not set(df["ProdTaken"].dropna().unique()).issubset({0, 1}):
        raise ValueError("ProdTaken must contain only 0 and 1.")

    fingerprint = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    print("DATA REGISTRATION PASSED")
    print(f"Rows               : {len(df):,}")
    print(f"Columns            : {df.shape[1]}")
    print(f"Duplicate rows     : {df.duplicated().sum()}")
    print(f"Duplicate customers: {df['CustomerID'].duplicated().sum()}")
    print("Target distribution:")
    print(df["ProdTaken"].value_counts().sort_index())
    print(f"SHA256             : {fingerprint}")

if __name__ == "__main__":
    main()
