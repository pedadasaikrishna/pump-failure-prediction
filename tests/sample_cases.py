"""Hand-crafted pump readings spanning healthy, degraded and edge conditions."""

HEALTHY_PUMP = {
    "Age_Years": 1.0, "Operating_Hours": 4000, "Duty_Cycle_Pct": 55,
    "Rotational_Speed_RPM": 1460, "Flow_Rate_m3h": 118,
    "Suction_Pressure_bar": 1.9, "Discharge_Pressure_bar": 4.5,
    "Vibration_X_mm_s": 1.3, "Vibration_Y_mm_s": 1.1,
    "Motor_Current_A": 8.5, "Motor_Temperature_C": 55,
    "Bearing_Temperature_C": 50, "Seal_Leak_Rate_ml_min": 0.5,
    "Ambient_Temperature_C": 26, "Maintenance_Quality_Score": 0.9,
    "Pump_Type": "A",
}

DEGRADED_PUMP = {
    "Age_Years": 8.5, "Operating_Hours": 34000, "Duty_Cycle_Pct": 88,
    "Rotational_Speed_RPM": 2800, "Flow_Rate_m3h": 210,
    "Suction_Pressure_bar": 1.0, "Discharge_Pressure_bar": 7.0,
    "Vibration_X_mm_s": 6.2, "Vibration_Y_mm_s": 5.3,
    "Motor_Current_A": 45, "Motor_Temperature_C": 98,
    "Bearing_Temperature_C": 102, "Seal_Leak_Rate_ml_min": 5.4,
    "Ambient_Temperature_C": 33, "Maintenance_Quality_Score": 0.25,
    "Pump_Type": "B",
}

BORDERLINE_PUMP = {
    "Age_Years": 4.5, "Operating_Hours": 18000, "Duty_Cycle_Pct": 70,
    "Rotational_Speed_RPM": 2900, "Flow_Rate_m3h": 400,
    "Suction_Pressure_bar": 1.5, "Discharge_Pressure_bar": 13.0,
    "Vibration_X_mm_s": 3.1, "Vibration_Y_mm_s": 2.6,
    "Motor_Current_A": 68, "Motor_Temperature_C": 75,
    "Bearing_Temperature_C": 80, "Seal_Leak_Rate_ml_min": 2.4,
    "Ambient_Temperature_C": 29, "Maintenance_Quality_Score": 0.6,
    "Pump_Type": "C",
}

ALL_CASES = {
    "healthy": HEALTHY_PUMP,
    "degraded": DEGRADED_PUMP,
    "borderline": BORDERLINE_PUMP,
}
