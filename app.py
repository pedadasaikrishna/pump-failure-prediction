"""
app.py
------
Streamlit demo for the Centrifugal Pump Predictive Maintenance project.

Run locally with:
    streamlit run app.py
"""
import json
import sys
import os

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import load_model, predict_one, risk_category  # noqa: E402
from features import NUMERIC_FEATURES  # noqa: E402

st.set_page_config(page_title="Pump Failure Predictor", page_icon="🛠️", layout="wide")

RISK_COLORS = {"Low": "#16a34a", "Moderate": "#f59e0b", "High": "#f97316", "Critical": "#dc2626"}


@st.cache_resource
def get_model():
    return load_model()


def main():
    pipe, metadata = get_model()

    st.title("🛠️ Centrifugal Pump — Predictive Maintenance")
    st.caption(
        "Predicts near-term failure risk for a centrifugal pump from live sensor readings. "
        f"Model: **{metadata['model_name']}** · Decision threshold: **{metadata['selected_threshold']}** · "
        f"Test ROC-AUC: **{metadata['test_metrics_at_selected_threshold']['roc_auc']}**"
    )

    tab_predict, tab_about = st.tabs(["🔍 Predict", "ℹ️ Model, dataset & limitations"])

    with tab_predict:
        st.subheader("Equipment parameter input")
        col1, col2, col3 = st.columns(3)

        with col1:
            pump_type = st.selectbox("Pump type", ["A", "B", "C"], index=0)
            age_years = st.slider("Age (years)", 0.0, 15.0, 3.0, 0.1)
            operating_hours = st.number_input("Operating hours (lifetime)", 0, 80000, 15000, 500)
            duty_cycle = st.slider("Duty cycle (%)", 0, 100, 65)

        with col2:
            rpm = st.number_input("Rotational speed (RPM)", 500, 4000, 1450, 10)
            flow = st.number_input("Flow rate (m³/h)", 0.0, 600.0, 120.0, 1.0)
            suction_p = st.number_input("Suction pressure (bar)", 0.0, 5.0, 1.8, 0.05)
            discharge_p = st.number_input("Discharge pressure (bar)", 0.0, 25.0, 4.5, 0.1)

        with col3:
            vib_x = st.number_input("Vibration X (mm/s)", 0.0, 25.0, 1.5, 0.1)
            vib_y = st.number_input("Vibration Y (mm/s)", 0.0, 25.0, 1.3, 0.1)
            motor_current = st.number_input("Motor current (A)", 0.0, 150.0, 10.0, 0.5)
            motor_temp = st.number_input("Motor temperature (°C)", 0.0, 170.0, 60.0, 1.0)

        col4, col5, col6 = st.columns(3)
        with col4:
            bearing_temp = st.number_input("Bearing temperature (°C)", 0.0, 140.0, 55.0, 1.0)
        with col5:
            seal_leak = st.number_input("Seal leak rate (ml/min)", 0.0, 10.0, 0.8, 0.1)
        with col6:
            ambient_temp = st.number_input("Ambient temperature (°C)", -10.0, 55.0, 27.0, 0.5)

        maint_quality = st.slider(
            "Maintenance quality score (0 = poor, 1 = excellent)", 0.0, 1.0, 0.7, 0.01
        )

        if st.button("Predict failure risk", type="primary"):
            raw_input = {
                "Age_Years": age_years, "Operating_Hours": operating_hours,
                "Duty_Cycle_Pct": duty_cycle, "Rotational_Speed_RPM": rpm,
                "Flow_Rate_m3h": flow, "Suction_Pressure_bar": suction_p,
                "Discharge_Pressure_bar": discharge_p, "Vibration_X_mm_s": vib_x,
                "Vibration_Y_mm_s": vib_y, "Motor_Current_A": motor_current,
                "Motor_Temperature_C": motor_temp, "Bearing_Temperature_C": bearing_temp,
                "Seal_Leak_Rate_ml_min": seal_leak, "Ambient_Temperature_C": ambient_temp,
                "Maintenance_Quality_Score": maint_quality, "Pump_Type": pump_type,
            }
            result = predict_one(raw_input, pipe, metadata)

            st.markdown("### Prediction result")
            c1, c2, c3 = st.columns(3)
            c1.metric("Failure probability", f"{result['failure_probability']:.1%}")
            c2.metric("Prediction", "FAILURE RISK" if result["prediction"] else "Normal")
            c3.markdown(
                f"<div style='padding:0.6em;border-radius:8px;background:{RISK_COLORS[result['risk_category']]};"
                f"color:white;text-align:center;font-weight:600'>{result['risk_category']} risk</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### Maintenance recommendation")
            st.info(result["recommendation"])

            st.markdown("### Explanation of this prediction")
            st.caption(
                "Standard-deviation distance of each sensor reading from the healthy-fleet average "
                "(no SHAP dependency required — see src/explain_importance.py for the full "
                "permutation-importance analysis used in the project report)."
            )
            import numpy as np
            bg = pd.read_csv("reports/tables/background_feature_stats.csv", index_col=0)
            z_scores = {}
            for feat in NUMERIC_FEATURES:
                if feat in raw_input and feat in bg.index and bg.loc[feat, "std"] > 0:
                    z_scores[feat] = (raw_input[feat] - bg.loc[feat, "mean"]) / bg.loc[feat, "std"]
            z_series = pd.Series(z_scores).sort_values(key=abs, ascending=False).head(6)
            st.bar_chart(z_series)

    with tab_about:
        st.subheader("Model")
        st.json({
            "model": metadata["model_name"],
            "selected_threshold": metadata["selected_threshold"],
            "test_metrics": metadata["test_metrics_at_selected_threshold"],
        })
        st.subheader("Dataset")
        st.markdown(
            "- **Source:** synthetically generated centrifugal-pump sensor dataset "
            "(`src/generate_dataset.py`), simulated from pump-affinity-law physics so it is "
            "reproducible and free of any proprietary plant data.\n"
            "- **Size:** 12,000 simulated readings across three pump classes (A/B/C).\n"
            "- **Target:** `Pump_Failure` (binary) — a failure occurred at/near that reading.\n"
            "- **Leakage control:** the four failure-mode flags (cavitation, bearing, seal, "
            "impeller) are technician-logged *outcomes* and are excluded from the model inputs, "
            "exactly as AI4I's TWF/HDF/PWF/OSF/RNF flags are excluded in that dataset's own "
            "modelling guidance."
        )
        st.subheader("Limitations")
        st.markdown(
            "- Trained on simulated data; sensor-noise and fault-signature assumptions, while "
            "physics-informed, are not calibrated against a real plant.\n"
            "- Minority class (failures) is only ~5.6% of rows — recall/precision trade-off was "
            "tuned on a validation split and may need re-tuning for a different failure base rate.\n"
            "- No real time-series / sequence modelling; each reading is scored independently."
        )


if __name__ == "__main__":
    main()
