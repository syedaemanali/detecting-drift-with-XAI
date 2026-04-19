import logging
import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def build_ground_truth(n_windows):
    """Windows before drift_start are clean, everything after is drifted."""
    drift_window = int(n_windows * config.DRIFT_START_FRAC)
    ground_truth = np.zeros(n_windows, dtype=bool)
    ground_truth[drift_window:] = True
    return ground_truth


def normalize_flags(flags):
    # PSI returns "ok"/"warning"/"alert" strings, everything else is already boolean
    if flags.dtype.kind in ("U", "S", "O"):
        return flags != "ok"
    return flags.astype(bool)


def detection_latency(ground_truth, predicted_flags):
    """Windows between when drift actually started and when detector first flagged it."""
    true_start = np.argmax(ground_truth)
    detected_at = np.argmax(predicted_flags)

    if not predicted_flags.any():
        return None  # never detected

    latency = max(0, detected_at - true_start)
    return int(latency)


def compute_rates(ground_truth, predicted_flags):
    tp = np.sum(ground_truth & predicted_flags)
    fp = np.sum(~ground_truth & predicted_flags)
    tn = np.sum(~ground_truth & ~predicted_flags)
    fn = np.sum(ground_truth & ~predicted_flags)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"fpr": fpr, "fnr": fnr, "precision": precision, "recall": recall, "f1": f1}


def cost_score(fpr, fnr, fp_weight=1.0, fn_weight=3.0):
    # false negatives cost more — missing real drift lets a bad model stay in production
    return fp_weight * fpr + fn_weight * fnr


def evaluate_detector(detector_name, flags, ground_truth):
    predicted = normalize_flags(flags)
    rates = compute_rates(ground_truth, predicted)
    latency = detection_latency(ground_truth, predicted)
    cost = cost_score(rates["fpr"], rates["fnr"])

    result = {
        "detector":  detector_name,
        "latency":   latency if latency is not None else -1,
        "fpr":       round(rates["fpr"], 4),
        "fnr":       round(rates["fnr"], 4),
        "precision": round(rates["precision"], 4),
        "recall":    round(rates["recall"], 4),
        "f1":        round(rates["f1"], 4),
        "cost":      round(cost, 4),
    }

    log.info(
        "%s latency %s windows  F1 %.4f  FPR %.4f  FNR %.4f  cost %.4f",
        detector_name, latency, rates["f1"], rates["fpr"], rates["fnr"], cost
    )

    return result


def evaluate_all_detectors(detector_results, n_windows):
    ground_truth = build_ground_truth(n_windows)
    rows = []

    for detector_name, flags in detector_results.items():
        row = evaluate_detector(detector_name, flags, ground_truth)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(config.METRICS_FILE, index=False)
    log.info("Metrics summary saved to %s", config.METRICS_FILE)

    return summary_df