import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PASTEL = list(config.PALETTE.values())


def _save(fig, filename):
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PLOTS_DIR / filename
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot %s", path.name)


def plot_class_distribution(y):
    counts = pd.Series(y).value_counts().sort_index()
    labels = ["No Subscription", "Subscribed"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=["#F4A896", "#7B9CDA"], width=0.5)
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set_title("Class Distribution", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_facecolor(config.PALETTE["background"])
    fig.patch.set_facecolor(config.PALETTE["background"])
    _save(fig, "class_distribution.png")


def plot_feature_distributions(X, feature_names):
    n_cols = 4
    n_rows = int(np.ceil(len(feature_names) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i, feature in enumerate(feature_names):
        axes[i].hist(X[:, i], bins=30, color=PASTEL[i % len(PASTEL)], edgecolor="white")
        axes[i].set_title(feature, fontsize=9)
        axes[i].set_facecolor(config.PALETTE["background"])

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=15, y=1.01)
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, "feature_distributions.png")


def plot_correlation_heatmap(X, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13)
    fig.patch.set_facecolor(config.PALETTE["background"])
    _save(fig, "correlation_heatmap.png")


def plot_feature_vs_target(X, y, feature_names):
    n_cols = 4
    n_rows = int(np.ceil(len(feature_names) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i, feature in enumerate(feature_names):
        df = pd.DataFrame({"value": X[:, i], "target": y})
        df.groupby("target")["value"].plot(
            kind="kde", ax=axes[i],
            color=["#F4A896", "#7B9CDA"], legend=True
        )
        axes[i].set_title(feature, fontsize=9)
        axes[i].set_facecolor(config.PALETTE["background"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distribution by Target Class", fontsize=15, y=1.01)
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, "feature_vs_target.png")


def plot_confusion_matrices(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    for ax, (model_name, data) in zip(axes, results.items()):
        cm = data["metrics"]["confusion_matrix"]
        disp = ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(model_name.replace("_", " ").title(), fontsize=11)
        ax.set_facecolor(config.PALETTE["background"])

    fig.suptitle("Confusion Matrices", fontsize=14)
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def plot_model_comparison(results):
    metrics_to_plot = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    model_names = list(results.keys())

    x = np.arange(len(metrics_to_plot))
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, model_name in enumerate(model_names):
        values = [results[model_name]["metrics"][m] for m in metrics_to_plot]
        offset = (i - len(model_names) / 2) * bar_width + bar_width / 2
        bars = ax.bar(x + offset, values, width=bar_width, label=model_name.replace("_", " ").title(),
                      color=PASTEL[i % len(PASTEL)])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot])
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison", fontsize=14)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    ax.set_facecolor(config.PALETTE["background"])
    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, "model_comparison_bar.png")


def plot_radar_chart(results):
    metrics_to_plot = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    model_names = list(results.keys())
    n_metrics = len(metrics_to_plot)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, model_name in enumerate(model_names):
        values = [results[model_name]["metrics"][m] for m in metrics_to_plot]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model_name.replace("_", " ").title(),
                color=PASTEL[i % len(PASTEL)])
        ax.fill(angles, values, alpha=0.15, color=PASTEL[i % len(PASTEL)])

    ax.set_thetagrids(np.degrees(angles[:-1]), [m.replace("_", " ").title() for m in metrics_to_plot])
    ax.set_ylim(0, 1)
    ax.set_title("Model Radar Chart", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.patch.set_facecolor(config.PALETTE["background"])
    _save(fig, "model_radar.png")


def plot_feature_importance(results, feature_names):
    # only tree-based models expose feature_importances_
    tree_models = {
        name: data["model"]
        for name, data in results.items()
        if hasattr(data["model"], "feature_importances_")
    }

    if not tree_models:
        log.info("No tree-based models found, skipping feature importance plot")
        return

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, tree_models.items()):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        ax.barh(
            [feature_names[i] for i in sorted_idx],
            importances[sorted_idx],
            color="#7B9CDA"
        )
        ax.invert_yaxis()
        ax.set_title(f"Feature Importance — {model_name.replace('_', ' ').title()}", fontsize=11)
        ax.set_facecolor(config.PALETTE["background"])

    fig.patch.set_facecolor(config.PALETTE["background"])
    fig.tight_layout()
    _save(fig, "feature_importance.png")


def run_full_eda(X_train, X_test, y_train, y_test, results, feature_names):
    log.info("Running full EDA")

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    plot_class_distribution(y_all)
    plot_feature_distributions(X_all, feature_names)
    plot_correlation_heatmap(X_all, feature_names)
    plot_feature_vs_target(X_all, y_all, feature_names)
    plot_confusion_matrices(results)
    plot_model_comparison(results)
    plot_radar_chart(results)
    plot_feature_importance(results, feature_names)

    log.info("EDA complete, all plots saved to %s", config.PLOTS_DIR)