"""
predict.py
----------
Loads the persisted pipeline and metadata, and exposes a single
`predict_one` / `predict_batch` interface used by app.py and the tests.
"""
import json

import joblib
import pandas as pd

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, engineer_features

MODEL_PATH = "models/pump_failure_pipeline.joblib"
METADATA_PATH = "models/model_metadata.json"


def load_model():
    pipe = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return pipe, metadata


def risk_category(prob: float) -> str:
    if prob < 0.20:
        return "Low"
    if prob < 0.50:
        return "Moderate"
    if prob < 0.75:
        return "High"
    return "Critical"


def recommendation(category: str) -> str:
    return {
        "Low": "No action required. Continue routine monitoring on the standard schedule.",
        "Moderate": "Schedule a condition check at the next planned maintenance window; "
                    "trend vibration and bearing temperature over the next few operating days.",
        "High": "Plan a maintenance intervention within the next few days: inspect seals, "
                "bearings and suction conditions (possible cavitation) before the next failure.",
        "Critical": "Recommend immediate inspection / controlled shutdown. Multiple sensor "
                    "readings are consistent with imminent mechanical failure.",
    }[category]


def predict_one(raw_input: dict, pipe=None, metadata=None):
    """raw_input: dict of the model's expected raw columns (see NUMERIC_FEATURES
    minus engineered ones, plus CATEGORICAL_FEATURES)."""
    if pipe is None or metadata is None:
        pipe, metadata = load_model()
    df = pd.DataFrame([raw_input])
    df = engineer_features(df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    prob = float(pipe.predict_proba(X)[0, 1])
    threshold = metadata["selected_threshold"]
    pred = int(prob >= threshold)
    cat = risk_category(prob)
    return {
        "failure_probability": round(prob, 4),
        "prediction": pred,
        "threshold_used": threshold,
        "risk_category": cat,
        "recommendation": recommendation(cat),
    }


def predict_batch(df_raw: pd.DataFrame, pipe=None, metadata=None) -> pd.DataFrame:
    if pipe is None or metadata is None:
        pipe, metadata = load_model()
    df = engineer_features(df_raw.copy())
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    probs = pipe.predict_proba(X)[:, 1]
    threshold = metadata["selected_threshold"]
    out = df_raw.copy()
    out["failure_probability"] = probs.round(4)
    out["prediction"] = (probs >= threshold).astype(int)
    out["risk_category"] = [risk_category(p) for p in probs]
    return out


if __name__ == "__main__":
    example = {
        "Age_Years": 7.2, "Operating_Hours": 32000, "Duty_Cycle_Pct": 82,
        "Rotational_Speed_RPM": 2850, "Flow_Rate_m3h": 230,
        "Suction_Pressure_bar": 1.1, "Discharge_Pressure_bar": 7.5,
        "Vibration_X_mm_s": 5.8, "Vibration_Y_mm_s": 4.9,
        "Motor_Current_A": 42, "Motor_Temperature_C": 95,
        "Bearing_Temperature_C": 98, "Seal_Leak_Rate_ml_min": 4.9,
        "Ambient_Temperature_C": 31, "Maintenance_Quality_Score": 0.35,
        "Pump_Type": "B",
    }
    print(json.dumps(predict_one(example), indent=2))
