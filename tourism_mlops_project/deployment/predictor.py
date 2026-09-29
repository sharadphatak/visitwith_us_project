
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tourism_model.joblib"

FEATURE_COLUMNS = [
    "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation",
    "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

class TourismPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}. "
                "Run the GitHub Actions training workflow first."
            )
        self.model = joblib.load(MODEL_PATH)

    def predict_probability(self, payload: dict) -> float:
        missing = [c for c in FEATURE_COLUMNS if c not in payload]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Rubric: collect inputs and save them into a DataFrame.
        input_df = pd.DataFrame(
            [{column: payload[column] for column in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS
        )
        return float(self.model.predict_proba(input_df)[0, 1])
