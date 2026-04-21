import json
import logging
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime

import config
from src.training.data_loader import load_data_variants, get_feature_names
from src.training.trainer import train_all_models, pick_best_model, save_best_model
from src.training.eda import run_full_eda
from src.simulation.create_drift import apply_drift
from src.simulation.create_drift import sample_drift_mask
from src.detection.shap import detect_shap_drift, compute_mean_shap_vector, build_explainer
from src.detection.psi import detect_psi_drift
from src.detection.ks import detect_ks_drift
from src.monitoring.metrics import evaluate_all_detectors
from src.visualization.plots import run_all_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _safe_quantile(values, q, fallback):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return fallback
    return float(np.quantile(values, q))


def calibrate_detectors(best_model, reference_data, calibration_stream):
    """Calibrate thresholds from clean pre-drift windows in the same stream domain."""
    window_size = config.STREAM_WINDOW_SIZE
    reference_windows = len(reference_data) // window_size
    calibration_windows = len(calibration_stream) // window_size

    if reference_windows < 2 or calibration_windows < 2:
        log.warning("Not enough windows for calibration; using config defaults")
        return {
            "shap_threshold": config.SHAP_DRIFT_THRESHOLD,
            "psi_warning": config.PSI_WARNING_THRESHOLD,
            "psi_alert": config.PSI_ALERT_THRESHOLD,
            "ks_alpha": config.KS_ALPHA,
            "warmup_windows": min(config.WARMUP_WINDOWS, calibration_windows),
        }

    pre_drift_windows = max(1, int(len(calibration_stream) * config.DRIFT_START_FRAC) // window_size)
    warmup_windows = min(config.WARMUP_WINDOWS, pre_drift_windows)
    calibration_window_count = min(config.CALIBRATION_WINDOWS, pre_drift_windows)
    calibration_window_count = max(calibration_window_count, warmup_windows + 1)
    calibration_window_count = min(calibration_window_count, calibration_windows)

    clean_reference = reference_data
    clean_stream = calibration_stream[: calibration_window_count * window_size]

    shap_distances, _, reference_shap = detect_shap_drift(
        best_model,
        clean_reference,
        clean_stream,
        threshold=config.SHAP_DRIFT_THRESHOLD,
        apply_confirmation=False,
        warmup_windows=0,
    )

    feature_weights = np.abs(reference_shap)
    if np.sum(feature_weights) > 0:
        feature_weights = feature_weights / np.sum(feature_weights)
    else:
        feature_weights = None

    psi_scores, _ = detect_psi_drift(
        clean_reference,
        clean_stream,
        feature_weights=feature_weights,
        apply_confirmation=False,
        warmup_windows=0,
    )

    ks_scores, _ = detect_ks_drift(
        clean_reference,
        clean_stream,
        alpha=config.KS_ALPHA,
        apply_confirmation=False,
        warmup_windows=0,
    )

    thresholds = {
        "shap_threshold": _safe_quantile(shap_distances, config.CALIBRATION_QUANTILE, config.SHAP_DRIFT_THRESHOLD),
        "psi_warning": _safe_quantile(psi_scores, config.PSI_WARNING_QUANTILE, config.PSI_WARNING_THRESHOLD),
        "psi_alert": _safe_quantile(psi_scores, config.PSI_ALERT_QUANTILE, config.PSI_ALERT_THRESHOLD),
        # KS detector flags lower p-values as drift. Use lower-tail quantile from clean data.
        "ks_alpha": _safe_quantile(ks_scores, 1 - config.CALIBRATION_QUANTILE, config.KS_ALPHA),
        "warmup_windows": warmup_windows,
        "calibration_windows": calibration_window_count,
    }

    if thresholds["psi_warning"] > thresholds["psi_alert"]:
        thresholds["psi_warning"] = thresholds["psi_alert"]

    log.info("Calibrated thresholds: %s", thresholds)
    return thresholds


def build_window_ground_truth(drift_type, n_samples, n_windows):
    sample_mask = sample_drift_mask(n_samples, drift_type)
    window_size = config.STREAM_WINDOW_SIZE

    window_truth = []
    for i in range(n_windows):
        window = sample_mask[i * window_size : (i + 1) * window_size]
        drift_fraction = float(np.mean(window)) if len(window) else 0.0
        window_truth.append(drift_fraction >= config.WINDOW_DRIFT_LABEL_THRESHOLD)
    return np.array(window_truth, dtype=bool)


def bootstrap_directories():
    dirs = [
        config.DATA_DIR, config.MODELS_DIR, config.RESULTS_DIR,
        config.PLOTS_DIR, config.SHAP_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    init_paths = [
        "src/__init__.py",
        "src/training/__init__.py",
        "src/simulation/__init__.py",
        "src/detection/__init__.py",
        "src/monitoring/__init__.py",
        "src/visualization/__init__.py",
    ]
    for p in init_paths:
        path = Path(p)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()

    log.info("Project directories and init files ready")


def run_training_phase(data_variants, feature_names):
    """Run baseline and SMOTE model suites in one experiment and choose one global best model."""
    log.info("Starting unified model training phase (baseline + SMOTE)")

    X_train_raw, X_test, y_train_raw, y_test = data_variants["raw"]
    X_train_smote, _, y_train_smote, _ = data_variants["smote"]

    baseline_results = train_all_models(
        X_train_raw,
        X_test,
        y_train_raw,
        y_test,
        experiment_name=config.MLFLOW_EXPERIMENT_NAME,
        model_name_suffix="baseline",
        run_name_prefix="baseline",
    )

    smote_results = train_all_models(
        X_train_smote,
        X_test,
        y_train_smote,
        y_test,
        experiment_name=config.MLFLOW_EXPERIMENT_NAME,
        model_name_suffix="smote",
        run_name_prefix="smote",
    )

    combined_results = {}
    for model_name, payload in baseline_results.items():
        combined_results[f"baseline_{model_name}"] = payload
    for model_name, payload in smote_results.items():
        combined_results[f"smote_{model_name}"] = payload

    best_name, best_model = pick_best_model(combined_results)
    save_best_model(best_name, best_model)

    # Use raw train/test for EDA distributions; model metrics come from each suite payload.
    run_full_eda(X_train_raw, X_test, y_train_raw, y_test, combined_results, feature_names)

    log.info("Unified training phase complete, global best model is %s", best_name)
    return best_name, best_model, combined_results


def build_ensemble_flags(shap_flags, psi_flags, ks_flags):
    psi_alert_flags = np.asarray(psi_flags) == "alert"
    votes = (
        np.asarray(shap_flags, dtype=bool).astype(int)
        + psi_alert_flags.astype(int)
        + np.asarray(ks_flags, dtype=bool).astype(int)
    )
    return votes >= 2


def run_drift_experiment(best_model, X_train, X_test, feature_names):
    log.info("Starting drift detection experiments")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    thresholds = calibrate_detectors(best_model, X_train, X_test)
    all_summaries = {}

    for drift_type in config.DRIFT_TYPES:
        log.info("Running %s drift experiment", drift_type)

        with mlflow.start_run(run_name=f"drift_{drift_type}"):
            drifted_stream = apply_drift(X_test, drift_type)

            shap_distances, shap_flags, reference_shap = detect_shap_drift(
                best_model,
                X_train,
                drifted_stream,
                threshold=thresholds["shap_threshold"],
                apply_confirmation=True,
                warmup_windows=thresholds["warmup_windows"],
            )

            feature_weights = np.abs(reference_shap)
            if np.sum(feature_weights) > 0:
                feature_weights = feature_weights / np.sum(feature_weights)
            else:
                feature_weights = None

            psi_scores, psi_flags = detect_psi_drift(
                X_train,
                drifted_stream,
                feature_weights=feature_weights,
                warning_threshold=thresholds["psi_warning"],
                alert_threshold=thresholds["psi_alert"],
                apply_confirmation=True,
                warmup_windows=thresholds["warmup_windows"],
            )
            ks_scores, ks_flags = detect_ks_drift(
                X_train,
                drifted_stream,
                alpha=thresholds["ks_alpha"],
                apply_confirmation=True,
                warmup_windows=thresholds["warmup_windows"],
            )

            ensemble_flags = build_ensemble_flags(shap_flags, psi_flags, ks_flags)

            n_windows = len(shap_distances)
            ground_truth = build_window_ground_truth(drift_type, len(drifted_stream), n_windows)

            summary_df = evaluate_all_detectors(
                {
                    "shap_drift": shap_flags,
                    "psi": psi_flags,
                    "ks": ks_flags,
                    "ensemble_alert": ensemble_flags,
                },
                n_windows,
                ground_truth=ground_truth,
            )
            all_summaries[drift_type] = summary_df

            for _, row in summary_df.iterrows():
                prefix = row["detector"]
                mlflow.log_metrics({
                    f"{prefix}_f1":      row["f1"],
                    f"{prefix}_fpr":     row["fpr"],
                    f"{prefix}_fnr":     row["fnr"],
                    f"{prefix}_latency": row["latency"],
                    f"{prefix}_cost":    row["cost"],
                })
            mlflow.log_param("drift_type", drift_type)
            mlflow.log_param("shap_threshold", thresholds["shap_threshold"])
            mlflow.log_param("psi_warning_threshold", thresholds["psi_warning"])
            mlflow.log_param("psi_alert_threshold", thresholds["psi_alert"])
            mlflow.log_param("ks_alpha", thresholds["ks_alpha"])
            mlflow.log_param("warmup_windows", thresholds["warmup_windows"])
            mlflow.log_param("calibration_windows", thresholds["calibration_windows"])
            mlflow.log_param("ensemble_rule", "majority_vote_2_of_3")

            explainer = build_explainer(best_model, X_train)
            last_window = drifted_stream[-config.STREAM_WINDOW_SIZE:]
            drifted_shap = compute_mean_shap_vector(explainer, last_window)

            shap_snapshots = [
                compute_mean_shap_vector(
                    explainer,
                    drifted_stream[i * config.STREAM_WINDOW_SIZE : (i + 1) * config.STREAM_WINDOW_SIZE]
                )
                for i in range(n_windows)
            ]

            run_all_plots(
                shap_distances, psi_scores, ks_scores,
                shap_snapshots, reference_shap, drifted_shap,
                summary_df, all_summaries, feature_names, drift_type
            )

            log.info("%s drift experiment complete", drift_type)

    return all_summaries


def export_results(all_summaries):
    export_dir = config.RESULTS_DIR / "paper_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # combine all drift type summaries into one table
    rows = []
    for drift_type, df in all_summaries.items():
        df = df.copy()
        df.insert(0, "drift_type", drift_type)
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True)
    combined["detector"] = combined["detector"].str.replace("_", " ").str.title()
    combined["drift_type"] = combined["drift_type"].str.title()

    combined.to_csv(export_dir / "metrics_table.csv", index=False)

    combined.to_latex(
        export_dir / "metrics_table.tex",
        index=False,
        float_format="%.4f",
        caption="Drift Detector Performance Across Drift Types",
        label="tab:results"
    )

    # summary json 
    last_summary = list(all_summaries.values())[-1]
    summary = {
        "generated_at":              datetime.now().isoformat(),
        "best_detector_by_f1":       last_summary.loc[last_summary["f1"].idxmax(),      "detector"],
        "best_detector_by_cost":     last_summary.loc[last_summary["cost"].idxmin(),     "detector"],
        "best_detector_by_latency":  last_summary.loc[last_summary["latency"].idxmin(),  "detector"],
        "per_drift_type": {
            drift_type: {
                "best_f1":      df.loc[df["f1"].idxmax(),     "detector"],
                "lowest_cost":  df.loc[df["cost"].idxmin(),   "detector"],
                "fastest":      df.loc[df["latency"].idxmin(),"detector"],
            }
            for drift_type, df in all_summaries.items()
        }
    }

    with open(export_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Results exported to %s", export_dir)


def main():
    bootstrap_directories()
    feature_names = get_feature_names()
    data_variants = load_data_variants()

    best_name, best_model, training_results = run_training_phase(data_variants, feature_names)

    raw_train_split = data_variants["raw"]
    X_train_reference, X_test_stream, _, _ = raw_train_split

    all_summaries = run_drift_experiment(best_model, X_train_reference, X_test_stream, feature_names)
    export_results(all_summaries)

    log.info("Pipeline finished with global best model: %s", best_name)
    for drift_type, df in all_summaries.items():
        log.info("Results for %s drift", drift_type)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()