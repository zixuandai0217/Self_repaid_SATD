"""Compute 55-feature significance tests and emit a LaTeX table for V4."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = next(parent for parent in PLAN_DIR.parents if parent.name == "Self Repaid SATD")
DATA_PATH = PLAN_DIR / "data" / "merged_data_40501_extended.json"
RESULTS_DIR = PLAN_DIR / "results"
V4_DIR = PROJECT_DIR / "paper" / "current" / "V4"
JSS_MANUSCRIPT_DIR = PROJECT_DIR / "paper" / "current" / "v4_jss" / "manuscript"

NUMERIC_FEATURES = [
    "code_cyclomatic_complexity", "code_file_lines", "code_imported_modules",
    "satd_position_in_file", "code_method_declaration_params",
    "developer_active_days", "developer_added_satd_count", "developer_removed_satd_count",
    "developer_total_commits", "developer_active_commits", "developer_last_commit_days",
    "developer_last_remove_satd_days", "developer_ownership",
    "developer_past_bugfix_ratio", "developer_past_cleanup_ratio",
    "developer_past_feature_ratio", "developer_past_refactor_ratio",
    "project_active_days", "project_total_commits", "project_active_commits",
    "project_total_developers", "project_active_developers", "project_files",
    "project_file_frequency", "project_file_authors", "project_last_commit_days",
    "project_readme_score", "satd_length", "satd_path_depth",
    "developer_fix_ratio", "developer_contribution_ratio",
    "comment_span", "method_body_lines",
    "dev_proj_past_satd_count", "proj_hist_satd_count",
    "comment_has_fixme", "comment_has_issue_ref", "comment_has_url", "comment_has_question",
    "is_test_file", "is_test_method", "is_void_method", "is_static_method",
    "developer_satd_density", "developer_fix_speed", "developer_is_top_committer",
    "ownership_x_active_commits", "ownership_x_contribution_ratio",
    "log1p_developer_total_commits",
]

CATEGORICAL_FEATURES = [
    "satd_add_is_weekend_or_night", "satd_quality_score", "satd_type",
    "f_comment_type", "comment_keyword_type", "method_visibility",
]

DISPLAY_NAMES = {
    "code_cyclomatic_complexity": "cyclomatic_complexity",
    "code_file_lines": "file_lines",
    "code_imported_modules": "imported_modules",
    "satd_position_in_file": "position_in_file",
    "code_method_declaration_params": "method_params",
    "developer_active_days": "d_active_days",
    "developer_added_satd_count": "added_satd_count",
    "developer_removed_satd_count": "repaid_satd_count",
    "developer_total_commits": "d_total_commits",
    "developer_active_commits": "d_active_commits",
    "developer_last_commit_days": "d_last_commit_days",
    "developer_last_remove_satd_days": "last_repay_satd_days",
    "developer_ownership": "ownership",
    "developer_past_bugfix_ratio": "past_bugfix_ratio",
    "developer_past_cleanup_ratio": "past_cleanup_ratio",
    "developer_past_feature_ratio": "past_feature_ratio",
    "developer_past_refactor_ratio": "past_refactor_ratio",
    "project_active_days": "p_active_days",
    "project_total_commits": "p_total_commits",
    "project_active_commits": "p_active_commits",
    "project_total_developers": "total_developers",
    "project_active_developers": "active_developers",
    "project_files": "files",
    "project_file_frequency": "file_frequency",
    "project_file_authors": "file_authors",
    "project_last_commit_days": "p_last_commit_days",
    "project_readme_score": "readme_score",
    "satd_length": "length",
    "satd_path_depth": "path_depth",
    "developer_fix_ratio": "repay_ratio",
    "developer_contribution_ratio": "contribution_ratio",
    "developer_satd_density": "satd_density",
    "developer_fix_speed": "repay_speed",
    "developer_is_top_committer": "is_top_committer",
    "log1p_developer_total_commits": "log1p_total_commits",
    "satd_add_is_weekend_or_night": "add_is_weekend_or_night",
    "satd_quality_score": "quality_score",
    "satd_type": "satd_type",
    "f_comment_type": "comment_type",
}

DIMENSIONS = {
    "Code": [
        "code_cyclomatic_complexity", "code_file_lines", "code_imported_modules",
        "code_method_declaration_params", "comment_span", "method_body_lines",
        "is_test_file", "is_test_method",
        "method_visibility", "is_void_method", "is_static_method",
    ],
    "Comment": [
        "satd_quality_score", "satd_length", "satd_path_depth", "satd_position_in_file",
        "satd_add_is_weekend_or_night", "f_comment_type", "comment_has_fixme",
        "comment_has_issue_ref", "comment_has_url", "comment_has_question",
        "comment_keyword_type", "satd_type",
    ],
    "Developer": [
        "developer_active_days", "developer_added_satd_count", "developer_removed_satd_count",
        "developer_total_commits", "developer_active_commits", "developer_last_commit_days",
        "developer_last_remove_satd_days", "developer_ownership",
        "developer_past_bugfix_ratio", "developer_past_cleanup_ratio",
        "developer_past_feature_ratio", "developer_past_refactor_ratio",
        "developer_fix_ratio", "developer_contribution_ratio", "developer_satd_density",
        "developer_fix_speed", "log1p_developer_total_commits",
    ],
    "Project": [
        "project_active_days", "project_total_commits", "project_active_commits",
        "project_total_developers", "project_active_developers", "project_files",
        "project_file_frequency", "project_file_authors", "project_last_commit_days",
        "project_readme_score", "proj_hist_satd_count",
    ],
    "Cross": [
        "dev_proj_past_satd_count", "developer_is_top_committer", "ownership_x_active_commits",
        "ownership_x_contribution_ratio",
    ],
}

BINARY_FEATURES = {
    "satd_add_is_weekend_or_night", "comment_has_fixme", "comment_has_issue_ref",
    "comment_has_url", "comment_has_question", "is_test_file", "is_test_method",
    "is_void_method", "is_static_method", "developer_is_top_committer",
}


def latex_escape(value: str) -> str:
    """Escape table text for LaTeX."""
    return value.replace("_", r"\_").replace("%", r"\%")


def format_p(value: float) -> str:
    """Format p-values compactly for the manuscript table."""
    if pd.isna(value):
        return "--"
    if value == 0 or value < 1e-300:
        return r"$<10^{-300}$"
    exponent = math.floor(math.log10(value))
    mantissa = value / (10 ** exponent)
    if exponent < -3:
        return rf"${mantissa:.2f}\times 10^{{{exponent}}}$"
    return f"{value:.3f}"


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Compute Benjamini-Hochberg adjusted q-values for multiple tests."""
    values = p_values.astype(float)
    order = values.sort_values().index
    adjusted = pd.Series(index=values.index, dtype=float)
    previous = 1.0
    total = len(values)
    for rank, index in reversed(list(enumerate(order, start=1))):
        q_value = min(previous, values.loc[index] * total / rank)
        adjusted.loc[index] = q_value
        previous = q_value
    return adjusted.clip(upper=1.0)


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    """Return chi-square p-value and Cramer's V for a contingency table."""
    chi2, p_value, _, _ = chi2_contingency(table)
    n_total = table.to_numpy().sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    effect = math.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else np.nan
    return p_value, effect


def cliff_delta(group0: pd.Series, group1: pd.Series) -> tuple[float, float]:
    """Return Mann-Whitney p-value and Cliff's delta for two numeric groups."""
    u_stat, p_value = mannwhitneyu(group0, group1, alternative="two-sided")
    n0, n1 = len(group0), len(group1)
    effect = (2 * u_stat - n0 * n1) / (n0 * n1)
    return p_value, effect


def summarize_numeric(group0: pd.Series, group1: pd.Series) -> str:
    """Summarize median values for non-self and self-repaid groups."""
    return f"median {group1.median():.2f} vs {group0.median():.2f}"


def summarize_binary(df: pd.DataFrame, feature: str) -> str:
    """Summarize positive rates for binary features."""
    rates = df.groupby("is_self_fixed")[feature].mean()
    return f"rate {rates.get(1, np.nan):.2f} vs {rates.get(0, np.nan):.2f}"


def feature_dimension(feature: str) -> str:
    """Find the paper dimension for a feature."""
    for dimension, features in DIMENSIONS.items():
        if feature in features:
            return dimension
    raise KeyError(f"No dimension mapping for {feature}")


def compute_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistical tests for the 55-feature set."""
    rows = []
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        display = DISPLAY_NAMES.get(feature, feature)
        dimension = feature_dimension(feature)
        if feature in BINARY_FEATURES or feature in CATEGORICAL_FEATURES:
            series = df[feature].fillna("missing")
            table = pd.crosstab(series, df["is_self_fixed"])
            p_value, effect = cramers_v(table)
            test = r"$\chi^2$"
            effect_name = "V"
            summary = summarize_binary(df, feature) if feature in BINARY_FEATURES else "association"
        else:
            values = pd.to_numeric(df[feature], errors="coerce")
            group0 = values[df["is_self_fixed"] == 0].dropna()
            group1 = values[df["is_self_fixed"] == 1].dropna()
            p_value, effect = cliff_delta(group0, group1)
            test = "MWU"
            effect_name = r"$\delta$"
            summary = summarize_numeric(group0, group1)
        rows.append({
            "dimension": dimension,
            "feature": feature,
            "display": display,
            "test": test,
            "p_value": p_value,
            "effect": effect,
            "effect_abs": abs(effect) if pd.notna(effect) else np.nan,
            "effect_name": effect_name,
            "summary": summary,
            "significant": p_value < 0.05 if pd.notna(p_value) else False,
        })
    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result["p_value"])
    result["fdr_significant"] = result["q_value"] < 0.05
    result["dimension_order"] = result["dimension"].map({name: i for i, name in enumerate(DIMENSIONS)})
    return result.sort_values(["dimension_order", "effect_abs"], ascending=[True, False])


def write_latex_table(results: pd.DataFrame, path: Path) -> None:
    """Write the compact 55-feature table used by V4."""
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Significance analysis for the 55-feature set. MWU denotes the Mann-Whitney U test with Cliff's $\delta$; $\chi^2$ denotes the Chi-square test with Cramer's V. P-values are raw values; Benjamini-Hochberg FDR correction is reported in the text. The summary column reports self-repaid values before non-self-repaid values.}",
        r"\label{tab:feature_significance}",
        r"\scriptsize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llcccl@{}}",
        r"\toprule",
        r"\textbf{Dim.} & \textbf{Feature} & \textbf{Test} & \textbf{P-value} & \textbf{Effect} & \textbf{Summary} \\",
        r"\midrule",
    ]
    last_dimension = None
    for _, row in results.iterrows():
        if last_dimension is not None and row.dimension != last_dimension:
            lines.append(r"\midrule")
        effect = f"{row.effect_name}={row.effect:.3f}" if pd.notna(row.effect) else "--"
        feature = rf"\texttt{{{latex_escape(row.display)}}}"
        summary = latex_escape(row.summary)
        lines.append(
            f"{row.dimension} & {feature} & {row.test} & {format_p(row.p_value)} & {effect} & {summary} \\\\"
        )
        last_dimension = row.dimension
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate CSV and LaTeX outputs for the 55-feature significance analysis."""
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    for feature in NUMERIC_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    results = compute_results(df)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "feature_significance_55feat.csv"
    tex_paths = [
        V4_DIR / "feature_significance_55feat_table.tex",
        JSS_MANUSCRIPT_DIR / "feature_significance_55feat_table.tex",
    ]
    results.drop(columns=["dimension_order"]).to_csv(csv_path, index=False)
    for tex_path in tex_paths:
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        write_latex_table(results, tex_path)
    print(f"Wrote {csv_path}")
    for tex_path in tex_paths:
        print(f"Wrote {tex_path}")
    print(f"Features tested: {len(results)}")
    print(f"Significant at p<0.05: {int(results.significant.sum())}")
    print(f"Significant after BH FDR q<0.05: {int(results.fdr_significant.sum())}")


if __name__ == "__main__":
    main()
