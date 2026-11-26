"""
Unit tests for the pump-failure prediction pipeline.
Run with:  PYTHONPATH=src pytest tests/ -v
(run from the project root so the relative model/data paths resolve)
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_validation import load_and_validate, REQUIRED_COLUMNS
from features import engineer_features, split_X_y, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from predict import load_model, predict_one, predict_batch, risk_category
from sample_cases import HEALTHY_PUMP, DEGRADED_PUMP, BORDERLINE_PUMP


@pytest.fixture(scope="module")
def model_and_meta():
    return load_model()


def test_dataset_has_required_columns():
    df, _ = load_and_validate()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_dataset_no_duplicates_after_validation():
    df, report = load_and_validate()
    assert report["n_exact_duplicates"] >= 0
    assert df.duplicated(subset=[c for c in df.columns if c != "Pump_ID"]).sum() == 0


def test_feature_engineering_adds_expected_columns():
    df, _ = load_and_validate()
    feats = engineer_features(df)
    for col in ["Vibration_Resultant_mm_s", "Temp_Differential_C", "Pressure_Ratio", "Specific_Energy_Index"]:
        assert col in feats.columns


def test_split_X_y_excludes_leakage_columns():
    df, _ = load_and_validate()
    X, y = split_X_y(df)
    for leak_col in ["Cavitation_Event", "Bearing_Failure", "Seal_Leakage", "Impeller_Wear", "Pump_Failure"]:
        assert leak_col not in X.columns
    assert set(X.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert y.isin([0, 1]).all()


def test_model_file_loads(model_and_meta):
    pipe, metadata = model_and_meta
    assert pipe is not None
    assert "selected_threshold" in metadata
    assert 0 <= metadata["selected_threshold"] <= 1


def test_predict_one_schema(model_and_meta):
    pipe, metadata = model_and_meta
    result = predict_one(HEALTHY_PUMP, pipe, metadata)
    for key in ["failure_probability", "prediction", "threshold_used", "risk_category", "recommendation"]:
        assert key in result
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["prediction"] in (0, 1)


def test_healthy_pump_scores_lower_than_degraded(model_and_meta):
    pipe, metadata = model_and_meta
    healthy = predict_one(HEALTHY_PUMP, pipe, metadata)
    degraded = predict_one(DEGRADED_PUMP, pipe, metadata)
    assert healthy["failure_probability"] < degraded["failure_probability"]
    assert healthy["risk_category"] in ("Low", "Moderate")
    assert degraded["risk_category"] in ("High", "Critical")


def test_borderline_pump_runs_without_error(model_and_meta):
    pipe, metadata = model_and_meta
    result = predict_one(BORDERLINE_PUMP, pipe, metadata)
    assert result["risk_category"] in ("Low", "Moderate", "High", "Critical")


def test_risk_category_boundaries():
    assert risk_category(0.05) == "Low"
    assert risk_category(0.30) == "Moderate"
    assert risk_category(0.60) == "High"
    assert risk_category(0.90) == "Critical"


def test_predict_batch_matches_predict_one(model_and_meta):
    pipe, metadata = model_and_meta
    df = pd.DataFrame([HEALTHY_PUMP, DEGRADED_PUMP, BORDERLINE_PUMP])
    batch_out = predict_batch(df, pipe, metadata)
    assert len(batch_out) == 3
    single_probs = [predict_one(row, pipe, metadata)["failure_probability"]
                     for row in [HEALTHY_PUMP, DEGRADED_PUMP, BORDERLINE_PUMP]]
    for a, b in zip(batch_out["failure_probability"].tolist(), single_probs):
        assert abs(a - b) < 1e-6


def test_missing_optional_sensor_handled_gracefully(model_and_meta):
    """The pipeline's median imputer should handle a NaN in a numeric input."""
    pipe, metadata = model_and_meta
    case = dict(HEALTHY_PUMP)
    case["Suction_Pressure_bar"] = float("nan")
    result = predict_one(case, pipe, metadata)
    assert 0.0 <= result["failure_probability"] <= 1.0
