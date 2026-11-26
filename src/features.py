"""
features.py
------------
Feature engineering shared by train.py and predict.py. Only uses signals
that would be available at prediction time (the four failure-mode flags
Cavitation_Event / Bearing_Failure / Seal_Leakage / Impeller_Wear are
*outcomes*, logged after the fact by maintenance technicians, so — exactly
like AI4I's TWF/HDF/PWF/OSF/RNF — they are excluded from the model's
input features to avoid target leakage. They are kept in the dataset only
for root-cause analysis in the EDA stage.)
"""
import numpy as np
import pandas as pd

LEAKAGE_COLUMNS = ["Cavitation_Event", "Bearing_Failure", "Seal_Leakage", "Impeller_Wear"]
ID_COLUMNS = ["Pump_ID"]
TARGET = "Pump_Failure"

NUMERIC_FEATURES = [
    "Age_Years", "Operating_Hours", "Duty_Cycle_Pct", "Rotational_Speed_RPM",
    "Flow_Rate_m3h", "Suction_Pressure_bar", "Discharge_Pressure_bar",
    "Vibration_X_mm_s", "Vibration_Y_mm_s", "Motor_Current_A",
    "Motor_Temperature_C", "Bearing_Temperature_C", "Seal_Leak_Rate_ml_min",
    "Ambient_Temperature_C", "Maintenance_Quality_Score",
    # engineered:
    "Vibration_Resultant_mm_s", "Temp_Differential_C", "Pressure_Ratio",
    "Specific_Energy_Index",
]
CATEGORICAL_FEATURES = ["Pump_Type"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Vibration_Resultant_mm_s"] = np.sqrt(df["Vibration_X_mm_s"] ** 2 + df["Vibration_Y_mm_s"] ** 2)
    df["Temp_Differential_C"] = df["Bearing_Temperature_C"] - df["Ambient_Temperature_C"]
    df["Pressure_Ratio"] = df["Discharge_Pressure_bar"] / df["Suction_Pressure_bar"].replace(0, np.nan)
    df["Specific_Energy_Index"] = (df["Motor_Current_A"] * df["Discharge_Pressure_bar"]) / (
        df["Flow_Rate_m3h"].replace(0, np.nan)
    )
    return df


def split_X_y(df: pd.DataFrame):
    df = engineer_features(df)
    drop_cols = [c for c in LEAKAGE_COLUMNS + ID_COLUMNS + [TARGET] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET] if TARGET in df.columns else None
    return X[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y
