import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from unittest.mock import patch, MagicMock


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def reference_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 5))


@pytest.fixture
def drifted_data():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((500, 5))
    # abrupt mean shift in second half simulates real distribution change
    data[250:] += 3.0
    return data


@pytest.fixture
def clean_stream():
    rng = np.random.default_rng(99)
    return rng.standard_normal((500, 5))


@pytest.fixture
def small_dataset():
    """Small balanced dataset for fast model training in tests."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=2, random_state=42
    )
    return X[:160], X[160:], y[:160], y[160:]


@pytest.fixture
def sample_dataframe():
    """Minimal bank marketing style dataframe for preprocessing tests."""
    return pd.DataFrame({
        "age":              [30, 45, 25, 60, 35],
        "job":              ["admin", "blue-collar", "technician", "retired", "admin"],
        "marital":          ["married", "single", "married", "divorced", "single"],
        "education":        ["university.degree", "basic.9y", "high.school", "basic.6y", "university.degree"],
        "default":          ["no", "no", "unknown", "no", "yes"],
        "housing":          ["yes", "no", "yes", "no", "yes"],
        "loan":             ["no", "no", "yes", "no", "no"],
        "contact":          ["cellular", "telephone", "cellular", "cellular", "telephone"],
        "month":            ["may", "jul", "aug", "may", "nov"],
        "day_of_week":      ["mon", "tue", "wed", "thu", "fri"],
        "duration":         [120, 300, 80, 450, 200],
        "campaign":         [1, 2, 1, 3, 1],
        "pdays":            [999, 3, 999, 6, 999],
        "previous":         [0, 1, 0, 2, 0],
        "poutcome":         ["nonexistent", "success", "nonexistent", "failure", "nonexistent"],
        "emp.var.rate":     [-1.8, 1.1, -1.8, 1.4, -1.8],
        "cons.price.idx":   [92.89, 93.99, 92.89, 93.44, 92.89],
        "cons.conf.idx":    [-46.2, -36.4, -46.2, -41.8, -46.2],
        "euribor3m":        [1.31, 4.86, 1.31, 4.96, 1.31],
        "nr.employed":      [5099.1, 5228.1, 5099.1, 5191.0, 5099.1],
        "y":                ["no", "yes", "no", "yes", "no"],
    })


# ── Data loader tests ─────────────────────────────────────────────────────────

def test_target_encoding_is_binary(sample_dataframe):
    df = sample_dataframe.copy()
    df["y"] = (df["y"] == "yes").astype(int)
    assert set(df["y"].unique()).issubset({0, 1})


def test_no_missing_values_after_encoding(sample_dataframe):
    df = sample_dataframe.copy()
    df["y"] = (df["y"] == "yes").astype(int)
    assert not df.isnull().any().any()


def test_split_produces_correct_array_dimensions(sample_dataframe):
    from src.training.data_loader import _split_dataframe
    import config

    df = sample_dataframe.copy()
    df["y"] = (df["y"] == "yes").astype(int)

    X_train, X_test, y_train, y_test = _split_dataframe(df)

    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


# ── Trainer tests ─────────────────────────────────────────────────────────────

def test_build_model_returns_valid_estimator():
    from src.training.trainer import build_model
    import config

    for model_name in config.MODELS_TO_TRAIN:
        model = build_model(model_name)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


def test_evaluate_returns_all_required_metric_keys(small_dataset):
    from src.training.trainer import build_model, evaluate

    X_train, X_test, y_train, y_test = small_dataset
    model = build_model("random_forest")
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    assert {"accuracy", "f1", "precision", "recall", "roc_auc", "confusion_matrix"}.issubset(metrics)


def test_all_scalar_metrics_are_in_valid_range(small_dataset):
    from src.training.trainer import build_model, evaluate

    X_train, X_test, y_train, y_test = small_dataset
    model = build_model("random_forest")
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    for key in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


def test_pick_best_model_selects_highest_f1():
    from src.training.trainer import pick_best_model

    mock_results = {
        "model_a": {"metrics": {"f1": 0.85}},
        "model_b": {"metrics": {"f1": 0.92}},
        "model_c": {"metrics": {"f1": 0.78}},
    }
    best_name, _ = pick_best_model(mock_results)
    assert best_name == "model_b"


# ── Drift simulation tests ────────────────────────────────────────────────────

def test_sudden_drift_only_affects_post_start_region(reference_data):
    from src.simulation.create_drift import sudden_drift
    import config

    drifted = sudden_drift(reference_data)
    start   = int(len(reference_data) * config.DRIFT_START_FRAC)

    pre_original = reference_data[:start].mean()
    assert abs(drifted[:start].mean() - pre_original) < 1e-6
    assert drifted[start:].mean() > pre_original + 0.5


def test_all_drift_types_preserve_input_shape(reference_data):
    from src.simulation.create_drift import apply_drift

    for drift_type in ["sudden", "gradual", "recurring", "incremental"]:
        result = apply_drift(reference_data, drift_type)
        assert result.shape == reference_data.shape


def test_unknown_drift_type_raises_value_error(reference_data):
    from src.simulation.create_drift import apply_drift

    with pytest.raises(ValueError, match="Unknown drift type"):
        apply_drift(reference_data, "fake_drift")


def test_gradual_drift_is_stronger_later(reference_data):
    from src.simulation.create_drift import gradual_drift
    import config

    drifted = gradual_drift(reference_data)
    start   = int(len(reference_data) * config.DRIFT_START_FRAC)
    mid     = start + (len(reference_data) - start) // 2

    assert drifted[mid:].mean() > drifted[start:mid].mean()


# ── KS drift detector tests ───────────────────────────────────────────────────

def test_ks_detector_flags_drift_in_shifted_stream(reference_data, drifted_data):
    from src.detection.ks_detector import detect_ks_drift

    _, flags = detect_ks_drift(reference_data, drifted_data)
    assert flags.any()


def test_ks_detector_low_false_positives_on_clean_stream(reference_data, clean_stream):
    from src.detection.ks_detector import detect_ks_drift

    _, flags = detect_ks_drift(reference_data, clean_stream)
    assert flags.mean() < 0.3


def test_ks_scores_are_valid_pvalues(reference_data, drifted_data):
    from src.detection.ks_detector import detect_ks_drift

    scores, flags = detect_ks_drift(reference_data, drifted_data)
    assert len(scores) == len(flags)
    assert np.all(scores >= 0) and np.all(scores <= 1)


# ── PSI drift detector tests ──────────────────────────────────────────────────

def test_psi_detects_drift_in_shifted_stream(reference_data, drifted_data):
    from src.detection.psi_detector import detect_psi_drift

    _, flags = detect_psi_drift(reference_data, drifted_data)
    assert "alert" in flags or "warning" in flags


def test_psi_scores_are_non_negative(reference_data, drifted_data):
    from src.detection.psi_detector import detect_psi_drift

    scores, flags = detect_psi_drift(reference_data, drifted_data)
    assert len(scores) == len(flags)
    assert np.all(scores >= 0)


def test_psi_on_identical_distributions_is_near_zero():
    from src.detection.psi_detector import compute_psi_single_feature

    rng  = np.random.default_rng(0)
    data = rng.standard_normal(1000)
    psi  = compute_psi_single_feature(data, data)
    assert psi < 0.05


# ── Metrics evaluation tests ──────────────────────────────────────────────────

def test_perfect_detector_achieves_f1_of_one():
    from src.monitoring.metrics import compute_rates

    ground_truth = np.array([False] * 4 + [True] * 6)
    predicted    = np.array([False] * 4 + [True] * 6)
    rates = compute_rates(ground_truth, predicted)

    assert rates["f1"]  == pytest.approx(1.0)
    assert rates["fpr"] == pytest.approx(0.0)
    assert rates["fnr"] == pytest.approx(0.0)


def test_detector_that_never_flags_has_zero_f1():
    from src.monitoring.metrics import compute_rates

    ground_truth = np.array([False] * 4 + [True] * 6)
    predicted    = np.zeros(10, dtype=bool)
    rates = compute_rates(ground_truth, predicted)

    assert rates["f1"]  == pytest.approx(0.0)
    assert rates["fnr"] == pytest.approx(1.0)


def test_detection_latency_is_correct_for_late_detector():
    from src.monitoring.metrics import detection_latency

    ground_truth = np.array([False] * 4 + [True] * 6)
    predicted    = np.array([False] * 6 + [True] * 4)
    assert detection_latency(ground_truth, predicted) == 2


def test_detection_latency_is_none_when_never_detected():
    from src.monitoring.metrics import detection_latency

    ground_truth = np.array([False] * 4 + [True] * 6)
    predicted    = np.zeros(10, dtype=bool)
    assert detection_latency(ground_truth, predicted) is None


def test_cost_score_penalizes_false_negatives_more_than_false_positives():
    from src.monitoring.metrics import cost_score

    # missing real drift (FN) costs more than a false alarm (FP)
    assert cost_score(fpr=1.0, fnr=0.0) < cost_score(fpr=0.0, fnr=1.0)


def test_evaluate_all_detectors_returns_correct_dataframe_shape():
    from src.monitoring.metrics import evaluate_all_detectors

    flags = {
        "shap_drift": np.array([False] * 4 + [True] * 6),
        "psi":        np.array(["ok"] * 4 + ["alert"] * 6),
        "ks":         np.array([False] * 5 + [True] * 5),
    }
    result = evaluate_all_detectors(flags, n_windows=10)

    assert isinstance(result, pd.DataFrame)
    assert "f1" in result.columns
    assert "detector" in result.columns
    assert len(result) == 3