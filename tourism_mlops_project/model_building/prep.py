
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism.csv"
OUTPUT_DIR = ROOT / "prepared_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rubric: load directly from repository data folder.
df = pd.read_csv(DATA_PATH)

# Data cleaning.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"].astype(int)

# Stratified split preserves the minority-class proportion.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_train.to_frame("ProdTaken").to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_frame("ProdTaken").to_csv(OUTPUT_DIR / "y_test.csv", index=False)

print("DATA PREPARATION PASSED")
print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))
print("Train purchase rate:", round(float(y_train.mean()), 4))
print("Test purchase rate :", round(float(y_test.mean()), 4))
print("Saved files:")
for p in sorted(OUTPUT_DIR.glob("*.csv")):
    print(" -", p.name)
