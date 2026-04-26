import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

PASTEL = list(config.PALETTE.values())


def _save(fig, filename):
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PLOTS_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path.name)


def plot_drift_overview(shap_distances, psi_scores, ks_scores, drift_type):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    n_windows = len(shap_distances)
    drift_window = int(n_windows * config.DRIFT_START_FRAC)
    x = np.arange(n_windows)

    datasets = [
        (
            axes[0],
            shap_distances,
            config.PALETTE["shap_drift"],
            "SHAP Cosine Distance",
            config.SHAP_DRIFT_THRESHOLD,
        ),
        (
            axes[1],
            psi_scores,
            config.PALETTE["psi"],
            "PSI Score",
            config.PSI_ALERT_THRESHOLD,
        ),
        (axes[2], ks_scores, config.PALETTE["ks"], "KS Mean P-Value", config.KS_ALPHA),
    ]

    for ax, scores, color, label, threshold in datasets:
        ax.plot(x, scores, color=color, linewidth=2, label=label)
        ax.axhline(
            threshold,
            color=config.PALETTE["alert"],
            linestyle="--",
            linewidth=1,
            label="threshold",
        )
        ax.axvline(
            drift_window,
            color="#999999",
            linestyle=":",
            linewidth=1.5,
            label="drift start",
        )
        ax.fill_between(
            x,
            scores,
            threshold,
            where=(np.array(scores) > threshold),
            alpha=0.2,
            color=config.PALETTE["alert"],
        )
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_facecolor(config.PALETTE["background"])

    axes[-1].set_xlabel("Window Index")
    fig.suptitle(f"Drift Detection Overview — {drift_type.title()} Drift", fontsize=14)
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, f"drift_overview_{drift_type}.png")


def plot_shap_vectors_over_time(shap_snapshots, feature_names, drift_type):
    # shap_snapshots is a list of mean shap vectors, one per window
    shap_matrix = np.array(shap_snapshots)
    n_windows, n_features = shap_matrix.shape

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(shap_matrix.T, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Window Index")
    ax.set_title(
        f"SHAP Feature Importance Over Time — {drift_type.title()} Drift", fontsize=13
    )
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    ax.axvline(
        int(n_windows * config.DRIFT_START_FRAC),
        color="white",
        linestyle="--",
        linewidth=2,
        label="drift start",
    )
    ax.legend(fontsize=9)
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, f"shap_heatmap_{drift_type}.png")


def plot_detector_comparison_bar(summary_df):
    metrics = ["f1", "precision", "recall", "fpr", "fnr"]
    detectors = summary_df["detector"].tolist()
    x = np.arange(len(metrics))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, detector in enumerate(detectors):
        row = summary_df[summary_df["detector"] == detector].iloc[0]
        values = [row[m] for m in metrics]
        offset = (i - len(detectors) / 2) * bar_width + bar_width / 2
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=detector.replace("_", " ").title(),
            color=PASTEL[i % len(PASTEL)],
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_title("Detector Performance Comparison", fontsize=14)
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_facecolor(config.PALETTE["background"])
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, "detector_comparison_bar.png")


def plot_latency_comparison(summary_df):
    detectors = summary_df["detector"].tolist()
    latencies = summary_df["latency"].tolist()
    colors = [PASTEL[i % len(PASTEL)] for i in range(len(detectors))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [d.replace("_", " ").title() for d in detectors],
        latencies,
        color=colors,
        width=0.5,
    )
    ax.bar_label(bars, fmt="%d windows", padding=3, fontsize=9)
    ax.set_title("Detection Latency by Detector", fontsize=13)
    ax.set_ylabel("Windows Until Detection")
    ax.set_facecolor(config.PALETTE["background"])
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, "latency_comparison.png")


def plot_cost_comparison(summary_df):
    detectors = summary_df["detector"].tolist()
    costs = summary_df["cost"].tolist()
    colors = [PASTEL[i % len(PASTEL)] for i in range(len(detectors))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [d.replace("_", " ").title() for d in detectors], costs, color=colors, width=0.5
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title("Cost Score by Detector (lower is better)", fontsize=13)
    ax.set_ylabel("Weighted Cost")
    ax.set_facecolor(config.PALETTE["background"])
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, "cost_comparison.png")


def plot_radar_chart(summary_df):
    metrics = ["f1", "precision", "recall", "fpr", "fnr"]
    detectors = summary_df["detector"].tolist()
    n = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, detector in enumerate(detectors):
        row = summary_df[summary_df["detector"] == detector].iloc[0]
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(
            angles,
            values,
            linewidth=2,
            label=detector.replace("_", " ").title(),
            color=PASTEL[i % len(PASTEL)],
        )
        ax.fill(angles, values, alpha=0.15, color=PASTEL[i % len(PASTEL)])

    ax.set_thetagrids(np.degrees(angles[:-1]), [m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Detector Radar Chart", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.patch.set_facecolor(config.PALETTE["background"])
    _save(fig, "detector_radar.png")


def plot_drift_type_heatmap(all_summaries):
    # all_summaries is a dict of {drift_type: summary_df}
    detectors = list(all_summaries.values())[0]["detector"].tolist()
    drift_types = list(all_summaries.keys())

    f1_matrix = np.zeros((len(detectors), len(drift_types)))

    for j, drift_type in enumerate(drift_types):
        df = all_summaries[drift_type]
        for i, detector in enumerate(detectors):
            f1_matrix[i, j] = df[df["detector"] == detector]["f1"].values[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[d.title() for d in drift_types],
        yticklabels=[d.replace("_", " ").title() for d in detectors],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("F1 Score per Detector per Drift Type", fontsize=13)
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, "drift_type_heatmap.png")


def plot_shap_reference_vs_drifted(
    reference_shap, drifted_shap, feature_names, drift_type
):
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        x - width / 2,
        reference_shap,
        width,
        label="Reference",
        color=config.PALETTE["ks"],
    )
    ax.bar(
        x + width / 2,
        drifted_shap,
        width,
        label="Drifted",
        color=config.PALETTE["alert"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"SHAP Importance Shift — {drift_type.title()} Drift", fontsize=13)
    ax.set_ylabel("Mean |SHAP|")
    ax.legend()
    ax.set_facecolor(config.PALETTE["background"])
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, f"shap_shift_{drift_type}.png")


def run_all_plots(
    shap_distances,
    psi_scores,
    ks_scores,
    shap_snapshots,
    reference_shap,
    drifted_shap,
    summary_df,
    all_summaries,
    feature_names,
    drift_type,
):

    log.info("Generating all drift visualizations")

    plot_drift_overview(shap_distances, psi_scores, ks_scores, drift_type)
    plot_shap_vectors_over_time(shap_snapshots, feature_names, drift_type)
    plot_detector_comparison_bar(summary_df)
    plot_latency_comparison(summary_df)
    plot_cost_comparison(summary_df)
    plot_radar_chart(summary_df)
    plot_drift_type_heatmap(all_summaries)
    plot_shap_reference_vs_drifted(
        reference_shap, drifted_shap, feature_names, drift_type
    )

    log.info("All plots saved to %s", config.PLOTS_DIR)
