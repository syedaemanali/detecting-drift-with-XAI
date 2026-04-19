import logging
import numpy as np
from scipy import stats

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def detect_ks_drift(reference_data, stream_data):
    """
    Slide a window over stream_data and run a KS test per feature against reference.
    Bonferroni correction applied so we don't get flooded with false positives
    when testing many features simultaneously.
    """
    window_size = config.STREAM_WINDOW_SIZE
    n_windows = len(stream_data) // window_size
    n_features = reference_data.shape[1]

    # Bonferroni corrected alpha — divide by number of features being tested
    corrected_alpha = config.KS_ALPHA / n_features

    ks_scores = []
    drift_flags = []

    for i in range(n_windows):
        window = stream_data[i * window_size : (i + 1) * window_size]

        feature_pvalues = []
        for f in range(n_features):
            _, pvalue = stats.ks_2samp(reference_data[:, f], window[:, f])
            feature_pvalues.append(pvalue)

        # drift flagged if any feature fails the corrected significance test
        any_drift = any(p < corrected_alpha for p in feature_pvalues)
        mean_pvalue = np.mean(feature_pvalues)

        ks_scores.append(mean_pvalue)
        drift_flags.append(any_drift)

        log.info(
            "KS window %d mean pvalue %.4f drift %s",
            i, mean_pvalue, any_drift
        )

    return np.array(ks_scores), np.array(drift_flags)