"""Generate benchmark visualizations from a metrics CSV."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Consistent colour palette per model (enough for 12+ models)
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F",
]


def generate_plots(metrics_df: pd.DataFrame, plots_dir: str) -> None:
    """Generate all visualisation plots and save to plots_dir."""
    if metrics_df.empty:
        logger.warning("No metrics to plot.")
        return

    out = Path(plots_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = metrics_df["model"].unique().tolist()
    color_map = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}

    _plot_quality_bars(metrics_df, color_map, out)
    _plot_quality_heatmap(metrics_df, out)
    _plot_validation_rate(metrics_df, out)

    has_runtime = (
        "runtime_s" in metrics_df.columns
        and metrics_df["runtime_s"].astype(str).ne("").any()
    )
    if has_runtime:
        _plot_runtime(metrics_df, color_map, out)
        _plot_tokens(metrics_df, color_map, out)
        _plot_throughput(metrics_df, color_map, out)

    _plot_radar(metrics_df, color_map, out)

    logger.info("Plots saved to %s", out)


# -----------------------------------------------------------------------
# 1. Grouped bar chart — quality metrics per model (one chart per field)
# -----------------------------------------------------------------------

def _plot_quality_bars(df: pd.DataFrame, colors: dict, out: Path) -> None:
    quality_cols = ["rouge1", "rouge2", "rougeL", "bleu", "chrf", "exact_match"]

    for field, fdf in df.groupby("field"):
        models = fdf["model"].tolist()
        n_models = len(models)
        n_metrics = len(quality_cols)
        x = np.arange(n_metrics)
        width = 0.8 / max(n_models, 1)

        fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), 6))
        for i, (_, row) in enumerate(fdf.iterrows()):
            vals = [row[c] for c in quality_cols]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=row["model"],
                   color=colors.get(row["model"], "#999999"))

        ax.set_xticks(x)
        ax.set_xticklabels(quality_cols, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"Quality Metrics — {field}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"quality_{field}.png", dpi=150)
        plt.close(fig)


# -----------------------------------------------------------------------
# 2. Heatmap — all quality metrics, models x metrics
# -----------------------------------------------------------------------

def _plot_quality_heatmap(df: pd.DataFrame, out: Path) -> None:
    quality_cols = ["rouge1", "rouge2", "rougeL", "bleu", "chrf",
                    "bertscore_f1", "exact_match"]

    for field, fdf in df.groupby("field"):
        models = fdf["model"].tolist()
        data = fdf[quality_cols].values.astype(float)

        fig, ax = plt.subplots(figsize=(max(8, len(quality_cols) * 1.2),
                                        max(4, len(models) * 0.6)))
        im = ax.imshow(data, aspect="auto", cmap="YlGnBu")

        ax.set_xticks(range(len(quality_cols)))
        ax.set_xticklabels(quality_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(quality_cols)):
                val = data[i, j]
                fmt = f"{val:.1f}" if val >= 1 else f"{val:.3f}"
                ax.text(j, i, fmt, ha="center", va="center", fontsize=8,
                        color="white" if val > data.max() * 0.65 else "black")

        ax.set_title(f"Quality Heatmap — {field}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out / f"heatmap_{field}.png", dpi=150)
        plt.close(fig)


# -----------------------------------------------------------------------
# 3. Validation rate bar chart
# -----------------------------------------------------------------------

def _plot_validation_rate(df: pd.DataFrame, out: Path) -> None:
    for field, fdf in df.groupby("field"):
        models = fdf["model"].tolist()
        rates = (fdf["n_valid"] / fdf["n_total"] * 100).tolist()

        fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.8), 5))
        x = np.arange(len(models))
        bars = ax.bar(x, rates, color="#55A868")
        ax.set_ylabel("Validation Rate (%)")
        ax.set_title(f"JSON + Pydantic Validation Rate — {field}")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")

        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"validation_rate_{field}.png", dpi=150)
        plt.close(fig)


# -----------------------------------------------------------------------
# 4. Runtime bar chart (model-level, deduplicated across fields)
# -----------------------------------------------------------------------

def _plot_runtime(df: pd.DataFrame, colors: dict, out: Path) -> None:
    rt = (
        df[df["runtime_s"].astype(str).ne("")]
        .drop_duplicates(subset=["model"])[["model", "runtime_s"]]
        .copy()
    )
    if rt.empty:
        return
    rt["runtime_s"] = rt["runtime_s"].astype(float)

    models = rt["model"].tolist()
    times = rt["runtime_s"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.8), 5))
    bar_colors = [colors.get(m, "#999") for m in models]
    bars = ax.barh(models, times, color=bar_colors)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title("Inference Runtime per Model")

    for bar, val in zip(bars, times):
        label = f"{val:.0f}s" if val >= 60 else f"{val:.1f}s"
        ax.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height() / 2,
                label, ha="left", va="center", fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "runtime.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# 5. Token count stacked bar chart (in / out)
# -----------------------------------------------------------------------

def _plot_tokens(df: pd.DataFrame, colors: dict, out: Path) -> None:
    tk = (
        df[df["tokens_in"].astype(str).ne("")]
        .drop_duplicates(subset=["model"])[["model", "tokens_in", "tokens_out"]]
        .copy()
    )
    if tk.empty:
        return
    tk["tokens_in"] = tk["tokens_in"].astype(int)
    tk["tokens_out"] = tk["tokens_out"].astype(int)

    models = tk["model"].tolist()
    tok_in = tk["tokens_in"].tolist()
    tok_out = tk["tokens_out"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.8), 5))
    x = np.arange(len(models))
    ax.bar(x, tok_in, label="Input tokens", color="#4C72B0")
    ax.bar(x, tok_out, bottom=tok_in, label="Output tokens", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Token Count")
    ax.set_title("Token Usage per Model (Input + Output)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tokens.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# 6. Throughput bar chart (tokens/sec)
# -----------------------------------------------------------------------

def _plot_throughput(df: pd.DataFrame, colors: dict, out: Path) -> None:
    tp = (
        df[df["tokens_per_sec"].astype(str).ne("")]
        .drop_duplicates(subset=["model"])[["model", "tokens_per_sec"]]
        .copy()
    )
    if tp.empty:
        return
    tp["tokens_per_sec"] = tp["tokens_per_sec"].astype(float)

    models = tp["model"].tolist()
    tps = tp["tokens_per_sec"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.8), 5))
    x = np.arange(len(models))
    bar_colors = [colors.get(m, "#999") for m in models]
    bars = ax.bar(x, tps, color=bar_colors)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Output Throughput per Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")

    for bar, val in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tps) * 0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# 7. Radar / spider chart — normalised quality metrics per model
# -----------------------------------------------------------------------

def _plot_radar(df: pd.DataFrame, colors: dict, out: Path) -> None:
    quality_cols = ["rouge1", "rougeL", "bleu", "chrf", "bertscore_f1", "exact_match"]

    for field, fdf in df.groupby("field"):
        models = fdf["model"].tolist()
        data = fdf[quality_cols].values.astype(float)

        # Normalise each metric to [0, 1] for radar
        col_max = data.max(axis=0)
        col_max[col_max == 0] = 1  # avoid division by zero
        normed = data / col_max

        n_metrics = len(quality_cols)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for i, model in enumerate(models):
            values = normed[i].tolist() + [normed[i][0]]
            ax.plot(angles, values, linewidth=1.5, label=model,
                    color=colors.get(model, "#999"))
            ax.fill(angles, values, alpha=0.08, color=colors.get(model, "#999"))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(quality_cols, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Model Comparison (normalised) — {field}", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"radar_{field}.png", dpi=150)
        plt.close(fig)
