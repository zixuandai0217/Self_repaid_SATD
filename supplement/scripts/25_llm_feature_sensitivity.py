"""Evaluate whether the final XGBoost result depends on LLM-derived features.

The final 55-feature model contains six auxiliary features produced with
DeepSeek-V3: quality_score, four prior-commit intent ratios, and readme_score.
This script removes those six features, reruns the same chronological XGBoost
protocol with the already tuned hyperparameters, and writes a sensitivity report.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

# Preload an optional libomp override before xgboost on macOS.
_libomp_path = os.environ.get("LIBOMP_DYLIB", "")
if os.path.exists(_libomp_path):
    ctypes.cdll.LoadLibrary(_libomp_path)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from temporal_utils import make_temporal_splits

DEFAULT_DATASET = PLAN_DIR / "data" / "merged_data_40501_extended.json"
RESULTS_DIR = PLAN_DIR / "results"
SEED = 42

FULL_NUMERIC_FEATURES = [
    "code_cyclomatic_complexity",
    "code_file_lines",
    "code_imported_modules",
    "satd_position_in_file",
    "code_method_declaration_params",
    "developer_active_days",
    "developer_added_satd_count",
    "developer_removed_satd_count",
    "developer_total_commits",
    "developer_active_commits",
    "developer_last_commit_days",
    "developer_last_remove_satd_days",
    "developer_ownership",
    "developer_past_bugfix_ratio",
    "developer_past_cleanup_ratio",
    "developer_past_feature_ratio",
    "developer_past_refactor_ratio",
    "project_active_days",
    "project_total_commits",
    "project_active_commits",
    "project_total_developers",
    "project_active_developers",
    "project_files",
    "project_file_frequency",
    "project_file_authors",
    "project_last_commit_days",
    "project_readme_score",
    "satd_length",
    "satd_path_depth",
    "developer_fix_ratio",
    "developer_contribution_ratio",
    "comment_span",
    "method_body_lines",
    "dev_proj_past_satd_count",
    "proj_hist_satd_count",
    "comment_has_fixme",
    "comment_has_issue_ref",
    "comment_has_url",
    "comment_has_question",
    "is_test_file",
    "is_test_method",
    "is_void_method",
    "is_static_method",
    "developer_satd_density",
    "developer_fix_speed",
    "developer_is_top_committer",
    "ownership_x_active_commits",
    "ownership_x_contribution_ratio",
    "log1p_developer_total_commits",
]

FULL_CATEGORICAL_FEATURES = [
    "satd_add_is_weekend_or_night",
    "satd_quality_score",
    "satd_type",
    "f_comment_type",
    "comment_keyword_type",
    "method_visibility",
]

LLM_NUMERIC_FEATURES = {
    "developer_past_bugfix_ratio",
    "developer_past_cleanup_ratio",
    "developer_past_feature_ratio",
    "developer_past_refactor_ratio",
    "project_readme_score",
}
LLM_CATEGORICAL_FEATURES = {"satd_quality_score"}

NUMERIC_FEATURES = [f for f in FULL_NUMERIC_FEATURES if f not in LLM_NUMERIC_FEATURES]
CATEGORICAL_FEATURES = [f for f in FULL_CATEGORICAL_FEATURES if f not in LLM_CATEGORICAL_FEATURES]


def make_preprocessor() -> ColumnTransformer:
    """Build preprocessing for the no-LLM-feature sensitivity model."""
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())])
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, NUMERIC_FEATURES), ("cat", cat_pipe, CATEGORICAL_FEATURES)]
    )


def load_data(path: Path) -> tuple[list[dict], pd.DataFrame, np.ndarray]:
    """Load records and coerce feature columns for sklearn pipelines."""
    records = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": "missing", "None": "missing", "nan": "missing"})
        )
    return records, df, df["is_self_fixed"].values.astype(int)


def build_xgboost(params: dict, pos: int, neg: int) -> XGBClassifier:
    """Create the XGBoost model using the final 55-feature tuned hyperparameters."""
    scale_pos_weight = neg / pos if pos else 1.0
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 700),
        max_depth=params.get("max_depth", 3),
        learning_rate=params.get("lr", 0.025),
        subsample=params.get("subsample", 0.89),
        colsample_bytree=params.get("colsample_bytree", 0.48),
        reg_alpha=params.get("reg_alpha", 1.12),
        reg_lambda=params.get("reg_lambda", 8.56),
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def evaluate_no_llm(records: list[dict], df: pd.DataFrame, y: np.ndarray, params: dict) -> pd.DataFrame:
    """Run the chronological expanding-window protocol without LLM-derived features."""
    rows = []
    for split in make_temporal_splits(records):
        train_idx = list(range(split.train_size))
        test_idx = list(range(split.train_size, split.train_size + split.test_size))
        x_train, y_train = df.iloc[train_idx], y[train_idx]
        x_test, y_test = df.iloc[test_idx], y[test_idx]
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)

        pipe = Pipeline(
            [
                ("pre", make_preprocessor()),
                ("model", build_xgboost(params, pos, neg)),
            ]
        )
        pipe.fit(x_train, y_train)
        proba = pipe.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append(
            {
                "model": "xgboost_no_llm_features",
                "split_pct": split.train_percentage,
                "auc": roc_auc_score(y_test, proba),
                "acc": accuracy_score(y_test, pred),
                "prec": precision_score(y_test, pred, zero_division=0),
                "rec": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "mcc": matthews_corrcoef(y_test, pred),
            }
        )
    return pd.DataFrame(rows)


def write_report(results: pd.DataFrame) -> None:
    """Write CSV and Markdown summaries for the no-LLM sensitivity analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "llm_feature_sensitivity.csv"
    md_path = RESULTS_DIR / "llm_feature_sensitivity.md"
    results.to_csv(csv_path, index=False)

    full = pd.read_csv(RESULTS_DIR / "temporal_55feat_results.csv")
    full_xgb = full[full["model"] == "xgboost"].copy()
    full_xgb["split_pct"] = (full_xgb["split_pct"] * 100).round().astype(int)
    merged = full_xgb.merge(results, on="split_pct", suffixes=("_full", "_no_llm"))

    row80 = merged[merged["split_pct"] == 80].iloc[0]
    mean_full = full_xgb[["auc", "mcc", "f1"]].mean()
    mean_no = results[["auc", "mcc", "f1"]].mean()

    lines = [
        "# LLM-Derived Feature Sensitivity",
        "",
        "## Design",
        "",
        "The sensitivity model removes six DeepSeek-V3-derived features from the final 55-feature XGBoost setting:",
        "",
        "- `satd_quality_score` / paper name `quality_score`",
        "- `developer_past_feature_ratio` / paper name `past_feature_ratio`",
        "- `developer_past_bugfix_ratio` / paper name `past_bugfix_ratio`",
        "- `developer_past_refactor_ratio` / paper name `past_refactor_ratio`",
        "- `developer_past_cleanup_ratio` / paper name `past_cleanup_ratio`",
        "- `project_readme_score` / paper name `readme_score`",
        "",
        "The same tuned XGBoost hyperparameters and chronological split definitions are reused.",
        "",
        "## Summary",
        "",
        "| Setting | 80/20 AUC | 80/20 MCC | Mean AUC | Mean MCC |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Full 55-feature XGBoost | {row80['auc_full']:.3f} | {row80['mcc_full']:.3f} | "
            f"{mean_full['auc']:.3f} | {mean_full['mcc']:.3f} |"
        ),
        (
            f"| No LLM-derived features (49) | {row80['auc_no_llm']:.3f} | {row80['mcc_no_llm']:.3f} | "
            f"{mean_no['auc']:.3f} | {mean_no['mcc']:.3f} |"
        ),
        "",
        "## Per-Split Results",
        "",
        "| Training window | Full AUC | No-LLM AUC | Full MCC | No-LLM MCC |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {row.split_pct}% | {row.auc_full:.3f} | {row.auc_no_llm:.3f} | "
            f"{row.mcc_full:.3f} | {row.mcc_no_llm:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(
        "80/20 no-LLM: "
        f"AUC={row80['auc_no_llm']:.4f}, MCC={row80['mcc_no_llm']:.4f}; "
        f"mean AUC={mean_no['auc']:.4f}, mean MCC={mean_no['mcc']:.4f}"
    )


def main() -> None:
    """Run the no-LLM-derived feature sensitivity experiment."""
    params = json.loads((RESULTS_DIR / "optuna_55feat_params.json").read_text(encoding="utf-8"))
    records, df, y = load_data(DEFAULT_DATASET)
    results = evaluate_no_llm(records, df, y, params["xgboost"])
    write_report(results)


if __name__ == "__main__":
    main()
