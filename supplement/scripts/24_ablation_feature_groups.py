"""Ablation study: compare XGBoost performance across three feature groups.

Groups:
  A) Developer + Cross-dimensional (21 features) — "behavioral inertia"
  B) Code + Comment + Project (34 features) — "contextual"
  C) All 55 features — full model reference

Uses the same XGBoost hyperparameters tuned on the 55-feature set (from
optuna_55feat_params.json) to ensure a fair comparison. Evaluated under the
primary 80%/20% chronological split and mean across 9 expanding-window splits.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import warnings
from pathlib import Path

# Preload an optional libomp override before xgboost on macOS.
_libomp_path = os.environ.get("LIBOMP_DYLIB", "")
if os.path.exists(_libomp_path):
    ctypes.cdll.LoadLibrary(_libomp_path)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from temporal_utils import make_temporal_splits

DEFAULT_DATASET = PLAN_DIR / "data" / "merged_data_40501_extended.json"
RESULTS_DIR = PLAN_DIR / "results"
SEED = 42

# ── Feature group definitions ─────────────────────────────────────────────────

# Group A: Developer + Cross-dimensional (behavioral inertia)
DEV_NUMERIC = [
    "developer_active_days", "developer_added_satd_count", "developer_removed_satd_count",
    "developer_total_commits", "developer_active_commits", "developer_last_commit_days",
    "developer_last_remove_satd_days", "developer_ownership",
    "developer_past_bugfix_ratio", "developer_past_cleanup_ratio",
    "developer_past_feature_ratio", "developer_past_refactor_ratio",
    "developer_fix_ratio", "developer_contribution_ratio",
    "developer_satd_density", "developer_fix_speed",
    "log1p_developer_total_commits",
    "dev_proj_past_satd_count", "developer_is_top_committer",
    "ownership_x_active_commits", "ownership_x_contribution_ratio",
]
DEV_CATEGORICAL: list[str] = []

# Group B: Code + Comment + Project (contextual signals)
CTX_NUMERIC = [
    "code_cyclomatic_complexity", "code_file_lines", "code_imported_modules",
    "code_method_declaration_params",
    "comment_span", "method_body_lines",
    "is_test_file", "is_test_method", "is_void_method", "is_static_method",
    "satd_position_in_file", "satd_length", "satd_path_depth",
    "comment_has_fixme", "comment_has_issue_ref", "comment_has_url", "comment_has_question",
    "project_active_days", "project_total_commits", "project_active_commits",
    "project_total_developers", "project_active_developers", "project_files",
    "project_file_frequency", "project_file_authors", "project_last_commit_days",
    "project_readme_score",
    "proj_hist_satd_count",
]
CTX_CATEGORICAL = [
    "method_visibility",
    "satd_add_is_weekend_or_night", "satd_quality_score", "satd_type",
    "f_comment_type", "comment_keyword_type",
]

# Group C: All 55 features, ordered exactly as the final V4 pipeline.
# XGBoost uses column sampling, so preserving this order keeps the full-model
# reference numerically aligned with `20_pipeline_55feat.py`.
ALL_NUMERIC = [
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
ALL_CATEGORICAL = [
    "satd_add_is_weekend_or_night", "satd_quality_score", "satd_type",
    "f_comment_type", "comment_keyword_type", "method_visibility",
]

FEATURE_GROUPS = {
    "Developer+Cross (21)": (DEV_NUMERIC, DEV_CATEGORICAL),
    "Code+Comment+Project (34)": (CTX_NUMERIC, CTX_CATEGORICAL),
    "All 55 features": (ALL_NUMERIC, ALL_CATEGORICAL),
}


def make_preprocessor(num_feats: list[str], cat_feats: list[str]) -> ColumnTransformer:
    """Build column transformer for a given feature subset."""
    transformers = []
    if num_feats:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())])
        transformers.append(("num", num_pipe, num_feats))
    if cat_feats:
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("cat", cat_pipe, cat_feats))
    return ColumnTransformer(transformers)


def load_data(path: Path):
    """Load dataset and build feature DataFrame + label array."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    all_num = list(set(DEV_NUMERIC + CTX_NUMERIC))
    all_cat = list(set(CTX_CATEGORICAL))
    for col in all_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in all_cat:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"": "missing", "None": "missing", "nan": "missing"})
    y = df["is_self_fixed"].values.astype(int)
    return records, df, y


def build_xgboost(params: dict, pos: int, neg: int) -> XGBClassifier:
    """Instantiate XGBoost with fixed tuned parameters."""
    spw = neg / pos if pos > 0 else 1.0
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 700),
        max_depth=params.get("max_depth", 3),
        learning_rate=params.get("lr", 0.025),
        subsample=params.get("subsample", 0.89),
        colsample_bytree=params.get("colsample_bytree", 0.48),
        reg_alpha=params.get("reg_alpha", 1.12),
        reg_lambda=params.get("reg_lambda", 8.56),
        scale_pos_weight=spw,
        objective="binary:logistic", random_state=SEED, n_jobs=-1, verbosity=0,
    )


def evaluate_group(records, df, y, xgb_params, group_name, num_feats, cat_feats):
    """Run 9-split temporal validation for one feature group."""
    splits = make_temporal_splits(records)
    results = []

    for sp in splits:
        tr_idx = list(range(0, sp.train_size))
        te_idx = list(range(sp.train_size, sp.train_size + sp.test_size))
        df_train, y_train = df.iloc[tr_idx], y[tr_idx]
        df_test, y_test = df.iloc[te_idx], y[te_idx]
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)

        pre = make_preprocessor(num_feats, cat_feats)
        model = build_xgboost(xgb_params, pos, neg)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(df_train, y_train)

        proba = pipe.predict_proba(df_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        results.append({
            "group": group_name,
            "split_pct": sp.train_percentage,
            "auc": roc_auc_score(y_test, proba),
            "acc": accuracy_score(y_test, pred),
            "prec": precision_score(y_test, pred, zero_division=0),
            "rec": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "mcc": matthews_corrcoef(y_test, pred),
        })

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Feature Group Ablation Study (XGBoost, Temporal Validation)")
    print("=" * 70)

    # Load XGBoost params from prior Optuna tuning
    params_path = RESULTS_DIR / "optuna_55feat_params.json"
    with open(params_path, "r") as f:
        all_params = json.load(f)
    xgb_params = all_params["xgboost"]
    print(f"\nXGBoost params (from 55-feature Optuna): {xgb_params}")

    print("\nLoading data...")
    records, df, y = load_data(DEFAULT_DATASET)
    print(f"  {len(records)} records loaded.")

    all_results = []

    for group_name, (num_feats, cat_feats) in FEATURE_GROUPS.items():
        n_total = len(num_feats) + len(cat_feats)
        print(f"\n{'─' * 60}")
        print(f"Evaluating: {group_name} ({len(num_feats)} numeric + {len(cat_feats)} categorical = {n_total})")
        print(f"{'─' * 60}")

        group_results = evaluate_group(records, df, y, xgb_params, group_name, num_feats, cat_feats)
        all_results.extend(group_results)

        # Print 80/20 result
        r80 = [r for r in group_results if r["split_pct"] == 80]
        if r80:
            r = r80[0]
            print(f"  80/20 split: AUC={r['auc']:.4f}  MCC={r['mcc']:.4f}  F1={r['f1']:.4f}")

        # Print mean across all splits
        aucs = [r["auc"] for r in group_results]
        mccs = [r["mcc"] for r in group_results]
        f1s = [r["f1"] for r in group_results]
        print(f"  Mean (9 splits): AUC={np.mean(aucs):.4f}  MCC={np.mean(mccs):.4f}  F1={np.mean(f1s):.4f}")

    # Save full results
    results_df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "ablation_feature_groups.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n\nFull results saved: {csv_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Feature Group Ablation (XGBoost)")
    print("=" * 70)
    print(f"\n{'Feature Group':<30} {'80/20 AUC':>10} {'80/20 MCC':>10} {'Mean AUC':>10} {'Mean MCC':>10}")
    print("-" * 70)

    for group_name in FEATURE_GROUPS:
        grp = results_df[results_df["group"] == group_name]
        r80 = grp[grp["split_pct"] == 80].iloc[0]
        mean_auc = grp["auc"].mean()
        mean_mcc = grp["mcc"].mean()
        print(f"{group_name:<30} {r80['auc']:>10.4f} {r80['mcc']:>10.4f} {mean_auc:>10.4f} {mean_mcc:>10.4f}")

    print("-" * 70)
    print("\nInterpretation:")
    print("  - If Developer+Cross ≈ All-55: contextual features add little beyond behavioral inertia")
    print("  - If Code+Comment+Project << All-55: developer features are essential")
    print("  - Gap between groups quantifies each dimension's contribution")

    # Generate markdown report
    md_lines = [
        "# Feature Group Ablation Study",
        "",
        "## Design",
        "",
        "Three XGBoost variants trained with **identical hyperparameters** (from 55-feature Optuna tuning)",
        "under the same chronological validation protocol:",
        "",
        "| Group | Features | Description |",
        "|-------|----------|-------------|",
        "| A | Developer + Cross-dimensional (21) | Behavioral inertia: ownership, activity, SATD history |",
        "| B | Code + Comment + Project (34) | Contextual signals: code structure, comment text properties, project context |",
        "| C | All 55 features | Full model (reference) |",
        "",
        "## Results: Primary 80%/20% Chronological Split",
        "",
        "| Feature Group | AUC | Accuracy | Precision | Recall | F1 | MCC |",
        "|---------------|-----|----------|-----------|--------|----|----|",
    ]

    for group_name in FEATURE_GROUPS:
        r80 = results_df[(results_df["group"] == group_name) & (results_df["split_pct"] == 80)].iloc[0]
        md_lines.append(
            f"| {group_name} | {r80['auc']:.3f} | {r80['acc']:.3f} | "
            f"{r80['prec']:.3f} | {r80['rec']:.3f} | {r80['f1']:.3f} | {r80['mcc']:.3f} |"
        )

    md_lines += [
        "",
        "## Results: Mean Across 9 Expanding-Window Splits",
        "",
        "| Feature Group | Mean AUC | Mean MCC | Mean F1 |",
        "|---------------|----------|----------|---------|",
    ]

    for group_name in FEATURE_GROUPS:
        grp = results_df[results_df["group"] == group_name]
        md_lines.append(f"| {group_name} | {grp['auc'].mean():.3f} | {grp['mcc'].mean():.3f} | {grp['f1'].mean():.3f} |")

    md_lines += [
        "",
        "## Interpretation",
        "",
        "The ablation quantifies the relative contribution of behavioral inertia",
        "(developer history features) vs. contextual signals (code/comment/project features).",
        "If Group A (Developer+Cross) performs close to the full model, it indicates that",
        "the model primarily relies on past behavioral patterns rather than structural context.",
        "",
    ]

    md_path = RESULTS_DIR / "ablation_feature_groups.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nMarkdown report: {md_path}")


if __name__ == "__main__":
    main()
