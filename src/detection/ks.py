import logging
import numpy as np
from scipy import stats

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _apply_temporal_confirmation(boolean_flags):
    window = config.TEMPORAL_CONFIRM_WINDOW
    min_alerts = config.TEMPORAL_CONFIRM_MIN_ALERTS

    smoothed = []
    for i in range(len(boolean_flags)):
        start = max(0, i - window + 1)
        votes = np.sum(boolean_flags[start : i + 1])
        smoothed.append(votes >= min_alerts)
    return np.array(smoothed, dtype=bool)


def detect_ks_drift(
    reference_data,
    stream_data,
    alpha=None,
    apply_confirmation=True,
    warmup_windows=0,
):
    """
    Bonferroni correction so we don't get alot of false positives
    when testing many features simultaneously.
    """
    window_size = config.STREAM_WINDOW_SIZE
    n_windows = len(stream_data) // window_size
    n_features = reference_data.shape[1]

    # Bonferroni : divide by number of features being tested
    active_alpha = config.KS_ALPHA if alpha is None else alpha
    corrected_alpha = active_alpha / n_features

    ks_scores = []
    raw_flags = []

    for i in range(n_windows):
        window = stream_data[i * window_size : (i + 1) * window_size]

        feature_pvalues = []
        for f in range(n_features):
            _, pvalue = stats.ks_2samp(reference_data[:, f], window[:, f])
            feature_pvalues.append(pvalue)

        # drift flagged if any feature fails the significance test
        any_drift = any(p < corrected_alpha for p in feature_pvalues)
        mean_pvalue = np.mean(feature_pvalues)

        ks_scores.append(mean_pvalue)
        raw_flags.append(any_drift)

    drift_flags = np.array(raw_flags, dtype=bool)
    if apply_confirmation:
        drift_flags = _apply_temporal_confirmation(drift_flags)

    if warmup_windows > 0:
        drift_flags[:warmup_windows] = False

    for i, (mean_pvalue, any_drift) in enumerate(zip(ks_scores, drift_flags)):
        log.info(
            "KS window %d mean pvalue %.4f drift %s",
            i, mean_pvalue, any_drift
        )

    return np.array(ks_scores), np.array(drift_flags)