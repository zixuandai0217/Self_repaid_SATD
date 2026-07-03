"""Run automated sensitivity checks for self-repaid identity labels."""

from __future__ import annotations

import ctypes
import json
import os
import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
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
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_DIR = SCRIPT_DIR.parent
SELF_REPAID_ROOT = next(parent for parent in SCRIPT_DIR.parents if parent.name == "Self Repaid SATD")
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_utils import make_temporal_splits

DEFAULT_DATASET = PLAN_DIR / "data" / "merged_data_40501_extended.json"
DEFAULT_DB = SELF_REPAID_ROOT / "code" / "data_preparation" / "data" / "satd.db"
RESULTS_DIR = PLAN_DIR / "results"
SEED = 42

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "satd_add_is_weekend_or_night",
    "satd_quality_score",
    "satd_type",
    "f_comment_type",
    "comment_keyword_type",
    "method_visibility",
]

LABEL_VARIANTS = [
    "original",
    "email_only",
    "strict_normalized",
    "expanded_normalized",
]


@dataclass(frozen=True)
class IdentityPair:
    """Store the add/remove identities used to derive alternative labels."""

    adder_name: str | None
    adder_email: str | None
    remover_name: str | None
    remover_email: str | None
    original_label: int


def _strip_accents(value: str) -> str:
    """Remove diacritics so names compare consistently across encodings."""

    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_email(value: str | None) -> str:
    """Normalize an email address for deterministic identity comparison."""

    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.strip("<>")
    if not text or "@" not in text:
        return text
    local, domain = text.split("@", 1)
    if domain == "users.noreply.github.com":
        local = re.sub(r"^\d+\+", "", local)
    return f"{local}@{domain}"


def normalize_name(value: str | None) -> str:
    """Normalize a commit author name for deterministic identity comparison."""

    if value is None:
        return ""
    text = _strip_accents(str(value)).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def email_local_part(value: str | None) -> str:
    """Return the normalized local part of an email address."""

    email = normalize_email(value)
    if "@" not in email:
        return ""
    local = email.split("@", 1)[0]
    return re.sub(r"[^a-z0-9]+", "", local)


def is_bot_or_noreply(name: str | None, email: str | None) -> bool:
    """Flag masked or automated identities that increase matching uncertainty."""

    name_text = normalize_name(name)
    email_text = normalize_email(email)
    bot_patterns = ("bot", "github actions", "dependabot", "renovate", "jenkins")
    return (
        any(pattern in name_text for pattern in bot_patterns)
        or "noreply" in email_text
        or email_text.endswith("@users.noreply.github.com")
    )


def derive_identity_labels(pair: IdentityPair) -> dict[str, int]:
    """Derive alternative self-repaid labels from one add/remove identity pair."""

    add_email = normalize_email(pair.adder_email)
    rem_email = normalize_email(pair.remover_email)
    add_name = normalize_name(pair.adder_name)
    rem_name = normalize_name(pair.remover_name)
    email_match = bool(add_email and rem_email and add_email == rem_email)
    name_match = bool(add_name and rem_name and add_name == rem_name)

    add_local = email_local_part(pair.adder_email)
    rem_local = email_local_part(pair.remover_email)
    local_match = bool(
        add_local
        and rem_local
        and add_local == rem_local
        and len(add_local) >= 4
        and add_local not in {"git", "github", "user", "admin", "root", "dev"}
    )

    return {
        "original": int(pair.original_label),
        "email_only": int(email_match),
        "strict_normalized": int(email_match or name_match),
        "expanded_normalized": int(email_match or name_match or local_match),
    }


def load_base_records(path: Path) -> tuple[list[dict], pd.DataFrame]:
    """Load the cleaned V4 dataset and coerce model feature columns."""

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
    return records, df


def load_identity_pairs(db_path: Path) -> pd.DataFrame:
    """Extract raw add/remove author identities from the SATDBailiff database."""

    query = """
    SELECT DISTINCT
        s_added.satd_id AS satd_id,
        c_removed.author_date AS remove_date,
        s_removed.second_commit AS remove_commit_hash,
        c_added.author_name AS adder_name,
        c_added.author_email AS adder_email,
        c_removed.author_name AS remover_name,
        c_removed.author_email AS remover_email,
        CASE
            WHEN c_added.author_name = c_removed.author_name
              OR c_added.author_email = c_removed.author_email
            THEN 1 ELSE 0
        END AS original_label_raw
    FROM satd s_removed
    JOIN satd s_added
      ON s_removed.satd_instance_id = s_added.satd_instance_id
     AND s_added.resolution = 'SATD_ADDED'
    JOIN commits c_added
      ON s_added.p_id = c_added.p_id
     AND s_added.second_commit = c_added.commit_hash
    JOIN commits c_removed
      ON s_removed.p_id = c_removed.p_id
     AND s_removed.second_commit = c_removed.commit_hash
    WHERE s_removed.resolution = 'SATD_REMOVED'
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def join_identities(base_df: pd.DataFrame, identity_df: pd.DataFrame) -> pd.DataFrame:
    """Join cleaned records with raw remover identities and validate coverage."""

    keys = ["satd_id", "remove_date"]
    base_with_id = base_df.reset_index(names="_base_row_id")
    joined = base_with_id.merge(identity_df, on=keys, how="left")
    missing = joined["remover_email"].isna() & joined["remover_name"].isna()
    if missing.any():
        raise ValueError(f"Missing raw remover identity for {int(missing.sum())} cleaned records")

    joined["_matches_clean_label"] = (
        joined["is_self_fixed"].astype(int) == joined["original_label_raw"].astype(int)
    ).astype(int)
    candidate_counts = joined.groupby("_base_row_id").size()
    duplicate_candidates = int((candidate_counts > 1).sum())
    if duplicate_candidates:
        print(
            "Resolved "
            f"{duplicate_candidates} cleaned records with multiple raw removal candidates "
            "by preferring the candidate whose exact-OR label matches the cleaned label."
        )
    joined = (
        joined.sort_values(
            ["_base_row_id", "_matches_clean_label", "remove_commit_hash"],
            ascending=[True, False, True],
        )
        .drop_duplicates("_base_row_id", keep="first")
        .sort_values("_base_row_id")
        .drop(columns=["_base_row_id", "_matches_clean_label"])
        .reset_index(drop=True)
    )
    rename_map = {}
    for field in ["adder_name", "adder_email"]:
        if f"{field}_x" in joined.columns:
            rename_map[f"{field}_x"] = field
        if f"{field}_y" in joined.columns:
            joined = joined.drop(columns=[f"{field}_y"])
    if rename_map:
        joined = joined.rename(columns=rename_map)

    mismatched = joined["is_self_fixed"].astype(int) != joined["original_label_raw"].astype(int)
    if mismatched.any():
        print(
            "Warning: reconstructed database exact-OR labels differ from the cleaned "
            f"dataset for {int(mismatched.sum())} records; continuing and reporting "
            "these as db_exact_label_disagreement."
        )
    joined["db_exact_label_disagreement"] = mismatched.astype(int)
    return joined


def add_label_variants(joined: pd.DataFrame) -> pd.DataFrame:
    """Append alternative label variants and identity-risk indicators."""

    label_rows: list[dict[str, int]] = []
    bot_or_noreply: list[int] = []
    for row in joined.itertuples(index=False):
        pair = IdentityPair(
            adder_name=getattr(row, "adder_name"),
            adder_email=getattr(row, "adder_email"),
            remover_name=getattr(row, "remover_name"),
            remover_email=getattr(row, "remover_email"),
            original_label=int(getattr(row, "is_self_fixed")),
        )
        label_rows.append(derive_identity_labels(pair))
        bot_or_noreply.append(
            int(
                is_bot_or_noreply(pair.adder_name, pair.adder_email)
                or is_bot_or_noreply(pair.remover_name, pair.remover_email)
            )
        )

    labels = pd.DataFrame(label_rows)
    out = joined.copy()
    for label in LABEL_VARIANTS:
        out[f"label_{label}"] = labels[label].astype(int).values
    out["identity_bot_or_noreply"] = bot_or_noreply
    out["identity_email_missing"] = (
        out["adder_email"].fillna("").astype(str).str.strip().eq("")
        | out["remover_email"].fillna("").astype(str).str.strip().eq("")
    ).astype(int)
    return out


def make_preprocessor() -> ColumnTransformer:
    """Build preprocessing for the 55-feature XGBoost sensitivity model."""

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


def build_xgboost(params: dict, pos: int, neg: int) -> XGBClassifier:
    """Instantiate XGBoost with the final 55-feature tuned parameters."""

    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=params.get("n_estimators", 700),
        max_depth=params.get("max_depth", 3),
        learning_rate=params.get("lr", 0.025),
        subsample=params.get("subsample", 0.89),
        colsample_bytree=params.get("colsample_bytree", 0.48),
        reg_alpha=params.get("reg_alpha", 1.12),
        reg_lambda=params.get("reg_lambda", 8.56),
        scale_pos_weight=(neg / pos if pos else 1.0),
        objective="binary:logistic",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def evaluate_label_variant(
    records: list[dict], df: pd.DataFrame, label_column: str, params: dict
) -> list[dict]:
    """Run chronological XGBoost validation for one alternative label."""

    results: list[dict] = []
    y = df[label_column].values.astype(int)
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
        results.append(
            {
                "label_variant": label_column.replace("label_", ""),
                "split_pct": split.train_percentage,
                "auc": roc_auc_score(y_test, proba),
                "acc": accuracy_score(y_test, pred),
                "prec": precision_score(y_test, pred, zero_division=0),
                "rec": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "mcc": matthews_corrcoef(y_test, pred),
            }
        )
    return results


def summarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize prevalence and agreement for every identity-label variant."""

    rows = []
    original = df["label_original"].astype(int)
    for variant in LABEL_VARIANTS:
        values = df[f"label_{variant}"].astype(int)
        rows.append(
            {
                "label_variant": variant,
                "positives": int(values.sum()),
                "negatives": int((1 - values).sum()),
                "positive_rate": float(values.mean()),
                "flips_vs_original": int((values != original).sum()),
                "positive_to_negative": int(((original == 1) & (values == 0)).sum()),
                "negative_to_positive": int(((original == 0) & (values == 1)).sum()),
                "agreement_with_original": float((values == original).mean()),
                "cohen_kappa_with_original": float(cohen_kappa_score(original, values)),
            }
        )
    return pd.DataFrame(rows)


def summarize_survival(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize survival days under each identity-label variant."""

    rows = []
    for variant in LABEL_VARIANTS:
        label_col = f"label_{variant}"
        for label_value, label_name in [(1, "self"), (0, "non_self")]:
            subset = df[df[label_col] == label_value]["satd_survival_days"].astype(float)
            rows.append(
                {
                    "label_variant": variant,
                    "group": label_name,
                    "count": int(len(subset)),
                    "median_survival_days": float(subset.median()),
                    "mean_survival_days": float(subset.mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_identity_risks(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize automated identity-risk indicators."""

    rows = []
    n = len(df)
    for column in ["identity_bot_or_noreply", "identity_email_missing", "db_exact_label_disagreement"]:
        flagged = int(df[column].sum())
        rows.append({"indicator": column, "count": flagged, "rate": flagged / n})
    return pd.DataFrame(rows)


def write_markdown_report(
    label_summary: pd.DataFrame,
    survival_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    model_results: pd.DataFrame,
) -> Path:
    """Write a reviewer-facing Markdown report for the sensitivity experiment."""

    md_path = RESULTS_DIR / "identity_label_sensitivity.md"
    lines = [
        "# Identity Label Sensitivity",
        "",
        "## Design",
        "",
        "This experiment reconstructs add/remove commit identities from `code/data_preparation/data/satd.db` "
        "and rejoins them to the cleaned 40,501-row V4 dataset by `satd_id + remove_date`. "
        "It does not use manual annotation.",
        "",
        "Label variants:",
        "",
        "- `original`: exact author-name OR exact author-email match, matching the V4 dataset.",
        "- `email_only`: normalized email addresses must match.",
        "- `strict_normalized`: normalized email OR normalized full name must match.",
        "- `expanded_normalized`: strict normalized match OR matching non-generic email local part.",
        "",
        "## Label Distribution and Agreement",
        "",
        "| Variant | Self count | Self rate | Flips vs original | Positive->negative | Negative->positive | Agreement | Kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in label_summary.itertuples(index=False):
        lines.append(
            f"| {row.label_variant} | {row.positives} | {row.positive_rate:.3f} | "
            f"{row.flips_vs_original} | {row.positive_to_negative} | "
            f"{row.negative_to_positive} | {row.agreement_with_original:.3f} | "
            f"{row.cohen_kappa_with_original:.3f} |"
        )

    lines += [
        "",
        "## Identity-Risk Indicators",
        "",
        "| Indicator | Count | Rate |",
        "|---|---:|---:|",
    ]
    for row in risk_summary.itertuples(index=False):
        lines.append(f"| {row.indicator} | {row.count} | {row.rate:.3f} |")

    lines += [
        "",
        "## Survival Summary",
        "",
        "| Variant | Group | Count | Median days | Mean days |",
        "|---|---|---:|---:|---:|",
    ]
    for row in survival_summary.itertuples(index=False):
        lines.append(
            f"| {row.label_variant} | {row.group} | {row.count} | "
            f"{row.median_survival_days:.1f} | {row.mean_survival_days:.1f} |"
        )

    lines += [
        "",
        "## XGBoost Temporal Sensitivity",
        "",
        "| Variant | 80/20 AUC | 80/20 MCC | Mean AUC | Mean MCC | Mean F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in LABEL_VARIANTS:
        subset = model_results[model_results["label_variant"] == variant]
        row80 = subset[subset["split_pct"] == 80].iloc[0]
        lines.append(
            f"| {variant} | {row80['auc']:.3f} | {row80['mcc']:.3f} | "
            f"{subset['auc'].mean():.3f} | {subset['mcc'].mean():.3f} | "
            f"{subset['f1'].mean():.3f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The email-only variant is the strict lower-bound label definition because it discards "
        "same-name matches across different email addresses. The normalized variants estimate "
        "how much the exact-match rule changes after case, punctuation, accent, and GitHub "
        "noreply normalization. Large performance or survival shifts would indicate that the "
        "paper's conclusions are sensitive to identity matching; small shifts support robustness.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    """Run all automated identity-label sensitivity experiments."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading cleaned V4 dataset...")
    records, base_df = load_base_records(DEFAULT_DATASET)
    print(f"Loaded {len(records)} records.")

    print("Loading raw identity pairs from SQLite...")
    identity_df = load_identity_pairs(DEFAULT_DB)
    joined = join_identities(base_df, identity_df)
    joined = add_label_variants(joined)

    label_summary = summarize_labels(joined)
    survival_summary = summarize_survival(joined)
    risk_summary = summarize_identity_risks(joined)

    params = json.loads((RESULTS_DIR / "optuna_55feat_params.json").read_text(encoding="utf-8"))[
        "xgboost"
    ]
    model_rows: list[dict] = []
    for variant in LABEL_VARIANTS:
        print(f"Evaluating XGBoost with label variant: {variant}")
        model_rows.extend(evaluate_label_variant(records, joined, f"label_{variant}", params))
    model_results = pd.DataFrame(model_rows)

    csv_path = RESULTS_DIR / "identity_label_sensitivity.csv"
    model_results.to_csv(csv_path, index=False)
    label_summary.to_csv(RESULTS_DIR / "identity_label_distribution.csv", index=False)
    survival_summary.to_csv(RESULTS_DIR / "identity_label_survival.csv", index=False)
    risk_summary.to_csv(RESULTS_DIR / "identity_risk_indicators.csv", index=False)
    md_path = write_markdown_report(label_summary, survival_summary, risk_summary, model_results)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
