"""
data_validation.py
-------------------
Schema and data-quality checks run before any modelling. Mirrors the kind
of guardrails you'd want on a real sensor-logger CSV export: column
presence, dtypes, allowed ranges, missingness and duplicate rows.
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "Pump_ID", "Pump_Type", "Age_Years", "Operating_Hours", "Duty_Cycle_Pct",
    "Rotational_Speed_RPM", "Flow_Rate_m3h", "Suction_Pressure_bar",
    "Discharge_Pressure_bar", "Vibration_X_mm_s", "Vibration_Y_mm_s",
    "Motor_Current_A", "Motor_Temperature_C", "Bearing_Temperature_C",
    "Seal_Leak_Rate_ml_min", "Ambient_Temperature_C",
    "Maintenance_Quality_Score", "Pump_Failure",
]

RANGE_CHECKS = {
    "Age_Years": (0, 20),
    "Duty_Cycle_Pct": (0, 100),
    "Rotational_Speed_RPM": (500, 4000),
    "Vibration_X_mm_s": (0, 15),       # values above this are flagged, not dropped
    "Bearing_Temperature_C": (0, 130),
    "Maintenance_Quality_Score": (0, 1),
}


def load_and_validate(path: str = "data/pump_sensor_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    report = {"n_rows_raw": len(df)}

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Duplicate rows (exact duplicates, e.g. double-logged readings)
    n_dupes = df.duplicated(subset=[c for c in df.columns if c != "Pump_ID"]).sum()
    report["n_exact_duplicates"] = int(n_dupes)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "Pump_ID"]).reset_index(drop=True)

    # Missingness
    na_counts = df.isna().sum()
    report["missing_by_column"] = na_counts[na_counts > 0].to_dict()

    # Range / plausibility flags (sensor glitches) — flagged, not silently dropped
    flags = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col in df.columns:
            n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_out:
                flags[col] = n_out
    report["range_violations"] = flags

    report["n_rows_clean"] = len(df)
    return df, report


if __name__ == "__main__":
    data, rpt = load_and_validate()
    print("Validation report:")
    for k, v in rpt.items():
        print(f"  {k}: {v}")
