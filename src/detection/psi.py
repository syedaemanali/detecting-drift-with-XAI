import logging
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

N_BINS = 10
EPSILON = 1e-6  # avoids log(0) when a bin is empty


def _apply_temporal_confirmation(boolean_flags):
    window = config.TEMPORAL_CONFIRM_WINDOW
    min_alerts = config.TEMPORAL_CONFIRM_MIN_ALERTS

    smoothed = []
    for i in range(len(boolean_flags)):
        start = max(0, i - window + 1)
        votes = np.sum(boolean_flags[start : i + 1])
        smoothed.append(votes >= min_alerts)
    return np.array(smoothed, dtype=bool)


def compute_psi_single_feature(reference, current):
    """PSI between two 1D arrays. Uses reference distribution to define bin edges."""
    bin_edges = np.percentile(reference, np.linspace(0, 100, N_BINS + 1))
    bin_edges = np.unique(bin_edges)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / len(reference) + EPSILON
    cur_pct = cur_counts / len(current) + EPSILON

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi


def detect_psi_drift(
    reference_data,
    stream_data,
    feature_weights=None,
    warning_threshold=None,
    alert_threshold=None,
    apply_confirmation=True,
    warmup_windows=0,
):
    """
    compute mean PSI across all features.
    """
    window_size = config.STREAM_WINDOW_SIZE
    n_windows = len(stream_data) // window_size
    n_features = reference_data.shape[1]

    active_warning = config.PSI_WARNING_THRESHOLD if warning_threshold is None else warning_threshold
    active_alert = config.PSI_ALERT_THRESHOLD if alert_threshold is None else alert_threshold

    weights = None
    if feature_weights is not None:
        weights = np.asarray(feature_weights, dtype=float)
        if len(weights) != n_features:
            raise ValueError("feature_weights length must match number of features")
        weights = np.clip(weights, EPSILON, None)
        weights = weights / np.sum(weights)

    psi_scores = []
    raw_flags = []

    for i in range(n_windows):
        window = stream_data[i * window_size : (i + 1) * window_size]

        feature_psi = [
            compute_psi_single_feature(reference_data[:, f], window[:, f])
            for f in range(n_features)
        ]
        mean_psi = np.average(feature_psi, weights=weights) if weights is not None else np.mean(feature_psi)
        psi_scores.append(mean_psi)

        raw_flags.append(mean_psi >= active_alert)

    alert_flags = np.array(raw_flags, dtype=bool)
    if apply_confirmation:
        alert_flags = _apply_temporal_confirmation(alert_flags)

    drift_flags = []
    for i, mean_psi in enumerate(psi_scores):
        if i < warmup_windows:
            status = "ok"
        elif alert_flags[i]:
            status = "alert"
        elif mean_psi >= active_warning:
            status = "warning"
        else:
            status = "ok"
        drift_flags.append(status)

        log.info("PSI window %d score %.4f status %s", i, mean_psi, status)

    return np.array(psi_scores), np.array(drift_flags)