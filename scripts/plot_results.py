import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yandex-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plots(root):
    root = Path(root)
    folder = root / "plots"
    folder.mkdir(exist_ok=True)
    experiments = pl.read_csv(root / "experiments.csv")
    chosen = pl.read_csv(root / "selected_features.csv")
    importance = pl.read_csv(root / "feature_importance.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 120, "savefig.dpi": 180})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name, bbox_inches="tight")
        plt.close(fig)
    top = importance.head(30).reverse()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh([s.replace("__", " / ") for s in top["feature"].to_list()], top["importance_gain"], color="#2864a6")
    ax.set_xlabel("LightGBM training split gain (proposal signal)")
    ax.set_title("Top 30 features of the locked automatic model")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "top30_importance.png")
    dev = experiments.filter((pl.col("split") == "validation_search") & (pl.col("seed") == 42))
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(np.arange(dev.height), dev["logloss"], "o-", color="#2864a6")
    ax.set_xticks(np.arange(dev.height), dev["experiment_name"].to_list(), rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Validation search logloss ↓")
    ax.set_title("Experiment comparison; search validation is reused")
    ax.grid(axis="y", alpha=.2)
    save(fig, "metric_by_experiment.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    for family, color in [("auto_adaptive", "#2864a6"), ("auto_fixed_windows", "#d9842a")]:
        subset = dev.filter(pl.col("experiment_name").str.starts_with(family)).sort("n_features")
        ax.plot(subset["n_features"], subset["logloss"], "o-", label=family, color=color)
    ax.set_xlabel("Number of model input features (including base)")
    ax.set_ylabel("Validation search logloss ↓")
    ax.set_title("Feature budget and diminishing returns")
    ax.legend()
    save(fig, "metric_vs_features.png")
    by_window = chosen.group_by("window").len().sort("len", descending=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(by_window["window"], by_window["len"], color="#2864a6")
    ax.set_ylabel("Selected FeatureSpecs")
    ax.set_title("Primary temporal window distribution (trend long windows in CSV)")
    save(fig, "window_distribution.png")
    entities = sorted(chosen["entity"].unique().to_list())
    windows = sorted(chosen["window"].unique().to_list(), key=lambda w: 100000 if w == "all_history" else int(w[:-1]))
    matrix = np.zeros((len(entities), len(windows)), dtype=int)
    for row in chosen.group_by("entity", "window").len().iter_rows(named=True):
        matrix[entities.index(row["entity"]), windows.index(row["window"])] = row["len"]
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(windows)), windows)
    ax.set_yticks(range(len(entities)), entities)
    for i in range(len(entities)):
        for j in range(len(windows)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_title("Entity × primary window: selected feature count")
    fig.colorbar(im, ax=ax, label="FeatureSpecs")
    save(fig, "entity_window_heatmap.png")
    groups = importance.group_by("entity").agg(pl.col("importance_gain").sum()).sort("importance_gain", descending=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(groups["entity"], groups["importance_gain"], color="#2864a6")
    ax.set_title("Training gain by feature family (not causal contribution)")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "feature_family_gain.png")
    profile = json.loads((root / "generation_profile.json").read_text())
    counts = [profile["n_candidates"], profile["n_after_pruning"], chosen.height]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Initial candidates", "After train pruning", "Locked selected"], counts, color=["#93aec9", "#5086b4", "#2864a6"])
    for i, value in enumerate(counts):
        ax.text(i, value, str(value), ha="center", va="bottom")
    ax.set_ylabel("Historical FeatureSpecs (base columns excluded)")
    save(fig, "feature_counts.png")
    if "test" in experiments["split"].to_list():
        final = experiments.filter(pl.col("split") == "test").group_by("experiment_name").agg(
            pl.col("logloss").mean().alias("mean"), pl.col("logloss").std().fill_null(0).alias("std"))
        final = final.sort("experiment_name")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.errorbar(np.arange(final.height), final["mean"], yerr=final["std"], fmt="o", capsize=5, color="#2864a6")
        ax.set_xticks(np.arange(final.height), final["experiment_name"].to_list(), rotation=25, ha="right")
        ax.set_ylabel("Test logloss ↓")
        ax.set_title("Locked models: mean ± seed SD; categorical/warmup arms have one seed")
        save(fig, "test_metrics.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    plots(parser.parse_args().output)
