import logging
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

N_BINS = 10
EPSILON = 1e-6  # avoids log(0) when a bin is empty


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


def detect_psi_drift(reference_data, stream_data):
    """
    Slide a window over stream_data and compute mean PSI across all features.
    Returns per-window PSI scores and drift flags based on configured thresholds.
    """
    window_size = config.STREAM_WINDOW_SIZE
    n_windows = len(stream_data) // window_size
    n_features = reference_data.shape[1]

    psi_scores = []
    drift_flags = []

    for i in range(n_windows):
        window = stream_data[i * window_size : (i + 1) * window_size]

        feature_psi = [
            compute_psi_single_feature(reference_data[:, f], window[:, f])
            for f in range(n_features)
        ]
        mean_psi = np.mean(feature_psi)
        psi_scores.append(mean_psi)

        if mean_psi >= config.PSI_ALERT_THRESHOLD:
            drift_flags.append("alert")
        elif mean_psi >= config.PSI_WARNING_THRESHOLD:
            drift_flags.append("warning")
        else:
            drift_flags.append("ok")

        log.info("PSI window %d score %.4f status %s", i, mean_psi, drift_flags[-1])

    return np.array(psi_scores), np.array(drift_flags)