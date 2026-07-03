"""Regenerate polished V4 paper figures from local data and result files."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


# Resolve manuscript and work paths from the script location.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[3]
WORK_DIR = SCRIPT_DIR.parent
V4_DIR = SCRIPT_DIR.parents[1] / "v4"
DATASET_DIR = PROJECT_DIR / "code" / "data_preparation" / "data"
RESULTS_DIR = WORK_DIR / "results"

PALETTE = {
    "blue": "#0F4D92",
    "blue_2": "#3775BA",
    "red": "#B64342",
    "red_2": "#E9A6A1",
    "green": "#3C8D61",
    "green_2": "#8BCF8B",
    "teal": "#42949E",
    "violet": "#7B4FA3",
    "orange": "#C95F2A",
    "gray": "#767676",
    "light_gray": "#F3F5F7",
    "border": "#D5DCE5",
}

MODEL_ORDER = [
    "random_classifier",
    "logistic_regression",
    "random_forest",
    "xgboost",
    "textcnn",
    "bert",
]
MODEL_LABELS = {
    "random_classifier": "Stratified random",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "textcnn": "TextCNN",
    "bert": "BERT",
}
MODEL_STYLES = {
    "random_classifier": {"color": "#8A8A8A", "marker": "o", "linestyle": "--"},
    "logistic_regression": {"color": PALETTE["blue_2"], "marker": "s", "linestyle": "-"},
    "random_forest": {"color": PALETTE["green"], "marker": "^", "linestyle": "-"},
    "xgboost": {"color": PALETTE["orange"], "marker": "D", "linestyle": "-"},
    "textcnn": {"color": PALETTE["violet"], "marker": "P", "linestyle": "-."},
    "bert": {"color": PALETTE["red"], "marker": "X", "linestyle": "-."},
}


def apply_style(font_size: int = 13) -> None:
    """Apply a restrained publication style for all regenerated figures."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.6,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.frameon": False,
            "legend.fontsize": font_size - 2,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, filename: str, dpi: int = 300, png: bool = False) -> None:
    """Save a figure to V4 with tight paper-friendly padding."""

    out = V4_DIR / filename
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04, dpi=dpi)
    if png:
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out}")


def draw_listing_card(
    filename: str,
    title: str,
    subtitle: str,
    lines: list[str],
    width: float = 10.8,
    height: float = 4.1,
    wrap_at: int = 96,
) -> None:
    """Render prompt or code text as a crisp listing-style image."""

    apply_style(13)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off()

    ax.text(0.02, 0.93, title, ha="left", va="top", fontsize=16, weight="bold", color=PALETTE["blue"])
    ax.text(0.02, 0.84, subtitle, ha="left", va="top", fontsize=10.5, color="#4D4D4D")

    box = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.68,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=PALETTE["border"],
        facecolor=PALETTE["light_gray"],
        transform=ax.transAxes,
    )
    ax.add_patch(box)

    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(line, width=wrap_at, subsequent_indent="    "))

    y = 0.71
    line_height = 0.045 if len(wrapped) <= 13 else 0.038
    for raw in wrapped[:18]:
        color = PALETTE["blue"] if raw.startswith(("#", "Task", "Output", "Scale")) else "#1F2933"
        ax.text(
            0.055,
            y,
            raw,
            ha="left",
            va="top",
            fontsize=9.4,
            family="DejaVu Sans Mono",
            color=color,
            transform=ax.transAxes,
        )
        y -= line_height

    save(fig, filename, dpi=320)


def draw_commit_card(filename: str, label: str, commit_hash: str, message: str) -> None:
    """Render one representative commit example as a compact code card."""

    apply_style(12)
    fig, ax = plt.subplots(figsize=(8.0, 2.25))
    ax.set_axis_off()

    box = FancyBboxPatch(
        (0.035, 0.12),
        0.93,
        0.76,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=PALETTE["border"],
        facecolor="#FAFBFC",
        transform=ax.transAxes,
    )
    ax.add_patch(box)

    ax.text(0.07, 0.76, label, weight="bold", color=PALETTE["blue"], fontsize=13, transform=ax.transAxes)
    ax.text(0.07, 0.58, f"commit {commit_hash}", family="DejaVu Sans Mono", fontsize=9.6, transform=ax.transAxes)
    ax.text(0.07, 0.43, "Author: dzx0217 <dzx0217@qq.com>", family="DejaVu Sans Mono", fontsize=9.6, transform=ax.transAxes)
    ax.text(0.07, 0.28, message, family="DejaVu Sans Mono", fontsize=10.4, color="#111827", transform=ax.transAxes)
    save(fig, filename, dpi=320)


def regenerate_prompt_and_example_figures() -> None:
    """Replace prompt screenshots and commit screenshots with clean listings."""

    developer_prompt = [
        'Task: classify one git commit message into its primary purpose.',
        'Choose exactly one tag from this label set:',
        '1. <feature>  Adds new functionality or user-visible behavior.',
        '2. <bugfix>   Fixes a bug, error, or incorrect behavior.',
        '3. <refactor> Restructures code without changing behavior.',
        '4. <cleanup>  Removes dead code, comments, or non-functional clutter.',
        'Output only the tag, such as <bugfix>.',
    ]
    draw_listing_card(
        "developer_past_favor_prompt.png",
        "Developer Commit-Intent Prompt",
        "Prompt used for the developer_past_*_ratio features.",
        developer_prompt,
        height=3.9,
    )

    readme_prompt = [
        "Score the latest README available at or before the SATD introduction commit.",
        "Scale: 0 to 5, where 0 means no usable README and 5 means complete, current documentation.",
        "Rubric:",
        "1. Overview: project purpose and main capabilities.",
        "2. Installation: environment, dependencies, build steps.",
        "3. Usage: runnable examples, commands, configuration.",
        "4. Maintenance: update recency, links, contribution guidance.",
        "Output only one integer from 0 to 5.",
    ]
    draw_listing_card(
        "readme_score_prompt.png",
        "README Quality Scoring Prompt",
        "Prompt template for the project_readme_score feature.",
        readme_prompt,
        height=4.35,
    )

    draw_commit_card(
        "feature.png",
        "Feature implementation",
        "b862f4320d61a0dc3aacac2d25446d395c4e3c2f",
        "Add OAuth2 login flow with Google provider",
    )
    draw_commit_card(
        "bugfix.png",
        "Bug fix",
        "b4d101d278a4cbf0cd6970cde0bc09c9ec72fe87",
        "Fix null-pointer exception in UserService when email is missing",
    )
    draw_commit_card(
        "refactor.png",
        "Code refactoring",
        "7ddf957babd0304957074cbc7a377fbc819e030a",
        "Refactor OrderController to use service layer abstraction",
    )
    draw_commit_card(
        "cleanup.png",
        "Code cleanup",
        "a540c5859de7e2452f4e402b362bf9476ac23607",
        "Remove unused helper methods and obsolete classes",
    )


def load_satd_records(updated: bool = True) -> pd.DataFrame:
    """Load the SATD dataset used by the V4 manuscript figures."""

    name = "merged_data_40501_updated.json" if updated else "merged_data_40501.json"
    with (DATASET_DIR / name).open("r", encoding="utf-8") as handle:
        return pd.DataFrame(json.load(handle))


def plot_type_distribution() -> None:
    """Redraw the per-type self-repaid distribution as a cleaner 100% bar chart."""

    apply_style(12)
    df = load_satd_records(updated=False)
    order = ["defect debt", "design debt", "documentation debt", "requirement debt", "test debt", "other type"]
    labels = ["Defect", "Design", "Documentation", "Requirement", "Test", "Other"]
    counts = (
        df.groupby(["satd_type", "is_self_fixed"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(order)
    )
    counts = counts.rename(columns={0: "Non-self-repaid", 1: "Self-repaid"})
    props = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    x = np.arange(len(order))
    ax.bar(x, props["Non-self-repaid"], width=0.62, color=PALETTE["blue"], edgecolor="white", linewidth=1.0)
    ax.bar(
        x,
        props["Self-repaid"],
        width=0.62,
        bottom=props["Non-self-repaid"],
        color=PALETTE["red"],
        edgecolor="white",
        linewidth=1.0,
    )

    for idx, satd_type in enumerate(order):
        non_self = int(counts.loc[satd_type, "Non-self-repaid"])
        self = int(counts.loc[satd_type, "Self-repaid"])
        ax.text(idx, props.loc[satd_type, "Non-self-repaid"] / 2, f"{non_self:,}", ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.text(idx, props.loc[satd_type, "Non-self-repaid"] + props.loc[satd_type, "Self-repaid"] / 2, f"{self:,}", ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.text(idx, 1.025, f"n={non_self + self:,}", ha="center", va="bottom", fontsize=8.5, color="#4D4D4D")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Within-type proportion")
    ax.set_xlabel("SATD type")
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(["Non-self-repaid", "Self-repaid"], loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2)
    save(fig, "is_self_fixed_satd_distribution_per_type.pdf", dpi=600)


def plot_project_boxplot() -> None:
    """Redraw the project-level self-repaying rate boxplot without the cramped table."""

    apply_style(12)
    df = load_satd_records(updated=True)
    project_rates = (
        df.groupby(["project_name", "satd_type"], observed=True)
        .agg(self_fix_rate=("is_self_fixed", "mean"), satd_count=("is_self_fixed", "size"))
        .reset_index()
    )
    order = ["defect debt", "design debt", "documentation debt", "requirement debt", "test debt"]
    labels = ["Defect", "Design", "Documentation", "Requirement", "Test"]
    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["green_2"], PALETTE["teal"], PALETTE["violet"]]
    data = [project_rates.loc[project_rates["satd_type"] == satd_type, "self_fix_rate"].dropna() for satd_type in order]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        medianprops={"color": "#111827", "linewidth": 1.8},
        whiskerprops={"color": "#374151", "linewidth": 1.1},
        capprops={"color": "#374151", "linewidth": 1.1},
        flierprops={"marker": "o", "markersize": 2.8, "markerfacecolor": "white", "markeredgecolor": "#6B7280", "alpha": 0.45},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=color, edgecolor="#111827", alpha=0.82, linewidth=1.1)

    medians = [float(np.median(values)) for values in data]
    counts = [int(df.loc[df["satd_type"] == satd_type].shape[0]) for satd_type in order]
    for idx, (median, count) in enumerate(zip(medians, counts), start=1):
        ax.text(idx, 1.035, f"median={median:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(idx, -0.105, f"n={count:,}", ha="center", va="top", fontsize=8.5, color="#4D4D4D")

    ax.axhline(0.5, color=PALETTE["gray"], linestyle=":", linewidth=1.1, zorder=0)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.14, 1.12)
    ax.set_ylabel("Project-level self-repaying rate")
    ax.set_xlabel("SATD type")
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}" if 0 <= value <= 1 else "")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, "satd_self_fixing_rate_boxplot.pdf", dpi=600)


def load_temporal_rows() -> pd.DataFrame:
    """Load and combine structured-feature and text-only temporal metrics."""

    structured = pd.read_csv(RESULTS_DIR / "temporal_55feat_results.csv").rename(
        columns={
            "split_pct": "train_percentage",
            "acc": "accuracy",
            "prec": "precision",
            "rec": "recall",
        }
    )
    structured["train_percentage"] = (
        pd.to_numeric(structured["train_percentage"]) * 100
    ).round().astype(int)
    text = pd.read_csv(RESULTS_DIR / "temporal_text_model_metrics_10_90_full.csv")
    text["train_percentage"] = pd.to_numeric(text["train_percentage"]).astype(int)
    metric_columns = [
        "model",
        "train_percentage",
        "auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
    ]
    rows = pd.concat([structured[metric_columns], text[metric_columns]], ignore_index=True)
    return rows[rows["model"].isin(MODEL_ORDER)].copy()


def plot_six_model_temporal() -> None:
    """Redraw Fig. 11 with larger panels and a dedicated legend row."""

    apply_style(12)
    rows = load_temporal_rows()
    metrics = [("auc", "AUC", (0.48, 0.76)), ("f1", "F1", (0.50, 0.73)), ("mcc", "MCC", (-0.04, 0.35))]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.15), sharex=True)

    for ax, (metric, title, limits) in zip(axes, metrics):
        for model in MODEL_ORDER:
            sub = rows[rows["model"] == model].sort_values("train_percentage")
            style = MODEL_STYLES[model]
            ax.plot(
                sub["train_percentage"],
                sub[metric],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.1,
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=MODEL_LABELS[model],
            )
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Training window (%)")
        ax.set_xticks(range(10, 100, 10))
        ax.set_ylim(*limits)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.grid(axis="x", color="#F1F3F5", linewidth=0.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Metric value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.18, top=0.78, wspace=0.23)
    save(fig, "six_model_temporal_comparison.pdf", dpi=600)


def plot_shap_bar() -> None:
    """Redraw the global SHAP importance bar chart with model feature names."""

    apply_style(11)
    df = pd.read_csv(RESULTS_DIR / "xgboost_shap_55feat_mean_abs.csv", encoding="utf-8-sig").head(15)
    df["label"] = df["feature"]
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.9, 5.2))
    colors = [PALETTE["blue"] if i >= 10 else PALETTE["blue_2"] for i in range(len(df))]
    ax.barh(df["label"], df["mean_abs_shap"], color=colors, edgecolor="white", linewidth=0.8, height=0.68)
    for y, value in enumerate(df["mean_abs_shap"]):
        ax.text(value + 0.004, y, f"{value:.3f}", va="center", fontsize=8.8, color="#374151")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.set_xlim(0, max(df["mean_abs_shap"]) * 1.18)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    save(fig, "xgboost_shap_temporal_global_importance.pdf", dpi=600)


def main() -> None:
    """Regenerate selected V4 figures that benefit from controlled redesign."""

    plot_project_boxplot()
    plot_six_model_temporal()


if __name__ == "__main__":
    main()
