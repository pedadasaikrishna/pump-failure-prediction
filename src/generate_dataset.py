"""
generate_dataset.py
--------------------
Builds the synthetic centrifugal-pump sensor dataset used throughout this
project.

Why synthetic data?
  Real plant SCADA/sensor logs for centrifugal pumps are almost never
  released publicly (proprietary OEM + plant data). To get a dataset that
  (a) is not AI4I-2020, NASA C-MAPSS/turbofan, or any "machine failure.csv"
  derivative, and (b) still behaves like genuine rotating-equipment
  condition-monitoring data, we simulate it from first principles:
  pump-affinity-law relationships between flow/speed/pressure, an
  Arrhenius-style bearing-heating model, and a degradation (wear) clock
  that drives four distinct, mutually-informative failure modes
  (cavitation, bearing failure, seal leakage, impeller wear). Gaussian
  sensor noise, random missingness and a handful of injected outliers are
  added afterwards so the data needs the same cleaning/EDA treatment as a
  real plant extract.

Run:
    python src/generate_dataset.py
Produces:
    data/pump_sensor_data.csv   (12,000 rows)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 12000


def generate():
    pump_type = RNG.choice(["A", "B", "C"], size=N, p=[0.5, 0.3, 0.2])
    # Base design points per pump class (centrifugal pumps of increasing duty)
    base = {
        "A": dict(rpm=1450, flow=120, head=45),
        "B": dict(rpm=2900, flow=260, head=90),
        "C": dict(rpm=2900, flow=410, head=140),
    }

    rpm_design = np.array([base[t]["rpm"] for t in pump_type])
    flow_design = np.array([base[t]["flow"] for t in pump_type])
    head_design = np.array([base[t]["head"] for t in pump_type])

    # Operating life / usage
    age_years = RNG.gamma(shape=2.2, scale=1.6, size=N).clip(0, 12)
    operating_hours = age_years * RNG.normal(4200, 400, N).clip(2500, 6000)
    duty_cycle_pct = RNG.uniform(35, 100, N)  # % of time running

    # A latent "wear" clock: increases with age, duty cycle and a random
    # maintenance-quality factor (some pumps are better maintained).
    maintenance_quality = RNG.beta(5, 2, N)  # higher = better maintained
    wear = (
        0.55 * (age_years / 12)
        + 0.30 * (duty_cycle_pct / 100)
        - 0.45 * maintenance_quality
        + RNG.normal(0, 0.08, N)
    )
    wear = (wear - wear.min()) / (wear.max() - wear.min())  # normalise 0-1

    # Operating point noise around design (how far off-BEP the pump runs)
    speed_pct = RNG.normal(100, 6, N).clip(70, 115)  # % of rated speed
    rpm = rpm_design * speed_pct / 100

    # Pump affinity laws: Q ∝ N, H ∝ N^2, with wear reducing achievable head/flow
    flow_rate = flow_design * (speed_pct / 100) * (1 - 0.18 * wear) + RNG.normal(0, 6, N)
    discharge_pressure = (head_design * (speed_pct / 100) ** 2 * (1 - 0.22 * wear)) / 10.2
    discharge_pressure += RNG.normal(0, 0.25, N)
    suction_pressure = RNG.normal(1.8, 0.35, N) - 0.4 * wear + RNG.normal(0, 0.1, N)

    npsh_margin = suction_pressure - (0.35 + 0.5 * (flow_rate / flow_design) ** 2)
    cavitation_risk = (npsh_margin < 0.15).astype(float) * RNG.uniform(0.6, 1.0, N)

    vibration_x_mm_s = 1.2 + 4.5 * wear ** 1.6 + 0.9 * cavitation_risk + RNG.normal(0, 0.3, N)
    vibration_y_mm_s = 1.0 + 4.0 * wear ** 1.7 + 0.5 * cavitation_risk + RNG.normal(0, 0.3, N)
    vibration_x_mm_s = vibration_x_mm_s.clip(0.1, None)
    vibration_y_mm_s = vibration_y_mm_s.clip(0.1, None)

    ambient_temp_c = RNG.normal(28, 4, N)
    motor_current_amp = (flow_rate * discharge_pressure / 60) * RNG.normal(1.0, 0.05, N) + 1.6 * wear
    motor_temperature_c = ambient_temp_c + 22 + 28 * wear + 0.4 * motor_current_amp + RNG.normal(0, 2, N)
    bearing_temperature_c = (
        ambient_temp_c + 18 + 55 * wear ** 1.4 + 3.2 * vibration_x_mm_s + RNG.normal(0, 2.2, N)
    )

    seal_leak_rate_ml_min = (RNG.exponential(0.6, N) + 6 * wear ** 2.2).clip(0, None)

    # ---- Failure-mode indicator flags (analogous to AI4I's TWF/HDF/PWF/OSF) ----
    cavitation_event = ((cavitation_risk > 0.62) & (RNG.uniform(0, 1, N) < 0.6)).astype(int)
    bearing_failure = ((bearing_temperature_c > 86) & (vibration_x_mm_s > 3.6) &
                        (RNG.uniform(0, 1, N) < 0.65)).astype(int)
    seal_leakage = ((seal_leak_rate_ml_min > 4.2) & (RNG.uniform(0, 1, N) < 0.55)).astype(int)
    impeller_wear = ((wear > 0.72) & (discharge_pressure < head_design / 10.2 * 0.82) &
                      (RNG.uniform(0, 1, N) < 0.55)).astype(int)
    random_fault = (RNG.uniform(0, 1, N) < 0.009).astype(int)  # unexplained random failures

    failure = ((cavitation_event + bearing_failure + seal_leakage + impeller_wear + random_fault) > 0).astype(int)

    pump_id = [f"P{10000 + i}" for i in range(N)]

    df = pd.DataFrame({
        "Pump_ID": pump_id,
        "Pump_Type": pump_type,
        "Age_Years": age_years.round(2),
        "Operating_Hours": operating_hours.round(0),
        "Duty_Cycle_Pct": duty_cycle_pct.round(1),
        "Rotational_Speed_RPM": rpm.round(0),
        "Flow_Rate_m3h": flow_rate.round(2),
        "Suction_Pressure_bar": suction_pressure.round(3),
        "Discharge_Pressure_bar": discharge_pressure.round(3),
        "Vibration_X_mm_s": vibration_x_mm_s.round(3),
        "Vibration_Y_mm_s": vibration_y_mm_s.round(3),
        "Motor_Current_A": motor_current_amp.round(2),
        "Motor_Temperature_C": motor_temperature_c.round(2),
        "Bearing_Temperature_C": bearing_temperature_c.round(2),
        "Seal_Leak_Rate_ml_min": seal_leak_rate_ml_min.round(3),
        "Ambient_Temperature_C": ambient_temp_c.round(2),
        "Maintenance_Quality_Score": maintenance_quality.round(3),
        "Cavitation_Event": cavitation_event,
        "Bearing_Failure": bearing_failure,
        "Seal_Leakage": seal_leakage,
        "Impeller_Wear": impeller_wear,
        "Pump_Failure": failure,
    })

    # ---- Inject realistic data-quality issues ----
    # 1) Missing values (MCAR) on a few sensor columns
    for col, frac in [("Suction_Pressure_bar", 0.012), ("Seal_Leak_Rate_ml_min", 0.02),
                       ("Ambient_Temperature_C", 0.008)]:
        idx = RNG.choice(N, size=int(N * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 2) A few sensor glitch outliers (stuck-at / spike readings)
    glitch_idx = RNG.choice(N, size=25, replace=False)
    df.loc[glitch_idx[:13], "Vibration_X_mm_s"] = RNG.uniform(18, 25, 13)
    df.loc[glitch_idx[13:], "Motor_Temperature_C"] = RNG.uniform(140, 160, 12)

    # 3) A handful of exact duplicate rows (common in logger exports)
    dup_idx = RNG.choice(N, size=15, replace=False)
    dup_rows = df.loc[dup_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv("data/pump_sensor_data.csv", index=False)
    print(f"Wrote data/pump_sensor_data.csv with shape {data.shape}")
    print("Failure rate: {:.2%}".format(data["Pump_Failure"].mean()))
