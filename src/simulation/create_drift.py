import logging
import numpy as np

import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


def _drift_start_index(data):
    return int(len(data) * config.DRIFT_START_FRAC)


def sample_drift_mask(n_samples, drift_type, period=100, drift_fraction=0.5):
    """Boolean mask indicating which samples belong to drifted segments."""
    start = int(n_samples * config.DRIFT_START_FRAC)
    mask = np.zeros(n_samples, dtype=bool)

    if drift_type in ("sudden", "gradual", "incremental"):
        mask[start:] = True
        return mask

    if drift_type == "recurring":
        drift_on_samples = int(period * drift_fraction)
        for i in range(start, n_samples):
            position_in_cycle = (i - start) % period
            if position_in_cycle < drift_on_samples:
                mask[i] = True
        return mask

    raise ValueError(
        f"Unknown drift type {drift_type}, choose from {list(DRIFT_FUNCTIONS)}"
    )


def sudden_drift(data):
    """mean jumps by drift_magnitude * std instantly."""
    out = data.copy().astype(float)
    start = _drift_start_index(out)
    feature_std = np.std(out[:start], axis=0)
    out[start:] += config.DRIFT_MAGNITUDE * feature_std
    log.info("Sudden drift injected from index %d onward", start)
    return out


def gradual_drift(data):
    """from zero shift at drift_start to full magnitude at the end."""
    out = data.copy().astype(float)
    start = _drift_start_index(out)
    feature_std = np.std(out[:start], axis=0)
    n_drifting = len(out) - start

    ramp = np.linspace(0, config.DRIFT_MAGNITUDE, n_drifting)
    out[start:] += ramp[:, np.newaxis] * feature_std
    log.info("Gradual drift injected from index %d with linear ramp", start)
    return out


def recurring_drift(data, period=100, drift_fraction=0.5):
    """Drift turns on and off in cycles after drift_start"""
    out = data.copy().astype(float)
    start = _drift_start_index(out)
    feature_std = np.std(out[:start], axis=0)
    drift_on_samples = int(period * drift_fraction)

    for i in range(start, len(out)):
        position_in_cycle = (i - start) % period
        if position_in_cycle < drift_on_samples:
            out[i] += config.DRIFT_MAGNITUDE * feature_std

    log.info("Recurring drift injected after index %d with period %d", start, period)
    return out


def incremental_drift(data):
    """Drift grows sample by sample from drift_start"""
    out = data.copy().astype(float)
    start = _drift_start_index(out)
    feature_std = np.std(out[:start], axis=0)
    n_drifting = len(out) - start

    # each sample gets a slightly larger shift than the previous one
    step_size = config.DRIFT_MAGNITUDE / n_drifting
    for i, idx in enumerate(range(start, len(out))):
        out[idx] += step_size * (i + 1) * feature_std

    log.info(
        "Incremental drift injected from index %d growing step %.6f", start, step_size
    )
    return out


DRIFT_FUNCTIONS = {
    "sudden": sudden_drift,
    "gradual": gradual_drift,
    "recurring": recurring_drift,
    "incremental": incremental_drift,
}


def apply_drift(data, drift_type):
    if drift_type not in DRIFT_FUNCTIONS:
        raise ValueError(
            f"Unknown drift type {drift_type}, choose from {list(DRIFT_FUNCTIONS)}"
        )
    return DRIFT_FUNCTIONS[drift_type](data)
