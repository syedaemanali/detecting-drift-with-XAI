import json
import logging
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime

import config
from src.training.data_loader import load_and_preprocess, get_feature_names
from src.training.trainer import train_all_models, pick_best_model, save_best_model
from src.training.eda import run_full_eda
from src.simulation.create_drift import apply_drift
from src.detection.shap import detect_shap_drift, compute_mean_shap_vector, build_explainer
from src.detection.psi import detect_psi_drift
from src.detection.ks import detect_ks_drift
from src.monitoring.metrics import evaluate_all_detectors
from src.visualization.plots import run_all_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


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


def run_training_phase(X_train, X_test, y_train, y_test, feature_names):
    log.info("Starting model training phase")
    results = train_all_models(X_train, X_test, y_train, y_test)
    best_name, best_model = pick_best_model(results)
    save_best_model(best_name, best_model)
    run_full_eda(X_train, X_test, y_train, y_test, results, feature_names)
    log.info("Training phase complete, best model is %s", best_name)
    return best_name, best_model, results


def run_drift_experiment(best_model, X_train, X_test, feature_names):
    log.info("Starting drift detection experiments")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    all_summaries = {}

    for drift_type in config.DRIFT_TYPES:
        log.info("Running %s drift experiment", drift_type)

        with mlflow.start_run(run_name=f"drift_{drift_type}"):
            drifted_stream = apply_drift(X_test, drift_type)

            shap_distances, shap_flags, reference_shap = detect_shap_drift(
                best_model, X_train, drifted_stream
            )
            psi_scores, psi_flags = detect_psi_drift(X_train, drifted_stream)
            ks_scores,  ks_flags  = detect_ks_drift(X_train, drifted_stream)

            n_windows = len(shap_distances)

            summary_df = evaluate_all_detectors(
                {"shap_drift": shap_flags, "psi": psi_flags, "ks": ks_flags},
                n_windows
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

    # combine all drift type summaries into one flat table
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

    # summary json — best detector per criterion
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
    X_train, X_test, y_train, y_test = load_and_preprocess()

    best_name, best_model, training_results = run_training_phase(
        X_train, X_test, y_train, y_test, feature_names
    )

    all_summaries = run_drift_experiment(best_model, X_train, X_test, feature_names)

    export_results(all_summaries)

    log.info("Pipeline finished, results in results/paper_export and plots in plots/")
    for drift_type, df in all_summaries.items():
        log.info("Results for %s drift", drift_type)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()