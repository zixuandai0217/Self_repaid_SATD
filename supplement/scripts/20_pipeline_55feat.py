"""Full pipeline for 55-feature model: Optuna tuning + 9-split temporal + SHAP.

Removes 6 target-derived features (self_fixed_count, self_fix_rate,
dev_proj_past_sf_rate, dev_proj_past_sf_count, proj_hist_sf_rate,
self_fix_rate_x_ownership) that are computed from is_self_fixed labels.
Remaining: 49 numeric + 6 categorical = 55 features.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = next(parent for parent in PLAN_DIR.parents if parent.name == "Self Repaid SATD")
sys.path.insert(0, str(SCRIPT_DIR))
from temporal_utils import is_monotonic_by_add_date, make_temporal_splits

DEFAULT_DATASET = PLAN_DIR / "data" / "merged_data_40501_extended.json"
RESULTS_DIR = PLAN_DIR / "results"
FIGURES_DIR = PLAN_DIR / "figures"
V4_DIR = PROJECT_DIR / "paper" / "current" / "V4"
JSS_MANUSCRIPT_DIR = PROJECT_DIR / "paper" / "current" / "v4_jss" / "manuscript"
TEXT_MODEL_CSV = RESULTS_DIR / "temporal_text_model_metrics_10_90_full.csv"

SEED = 42

SHAP_BAR_EDGE = "#F3F6FA"
SHAP_AXIS_COLOR = "#3E464C"
SHAP_GRID_COLOR = "#DDE6EA"
SHAP_BAR_CMAP = LinearSegmentedColormap.from_list("shap_bar_rank", ["#DCE9F6", "#4E79A7"])
SHAP_CMAP = LinearSegmentedColormap.from_list("shap_feature_value", ["#4E79A7", "#EEF3F5", "#E58B4A"])

# 49 numeric features (6 target-derived features removed)
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


def make_preprocessor():
    """Build column transformer for the 55-feature set."""
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, NUMERIC_FEATURES), ("cat", cat_pipe, CATEGORICAL_FEATURES)])


def optimal_mcc_threshold(y_true, proba):
    """Find threshold that maximizes MCC."""
    best_t, best_mcc = 0.5, 0.0
    for t in np.arange(0.25, 0.75, 0.005):
        m = matthews_corrcoef(y_true, (proba >= t).astype(int))
        if m > best_mcc:
            best_mcc, best_t = m, t
    return best_t, best_mcc


def shorten_feature_name(raw_name: str) -> str:
    """Map raw column names to shorter paper-friendly labels."""
    if raw_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[raw_name]
    for prefix in ("numeric__", "categorical__"):
        if raw_name.startswith(prefix):
            inner = raw_name[len(prefix):]
            return DISPLAY_NAMES.get(inner, inner)
    return raw_name


def load_data(path: Path):
    """Load dataset and build feature DataFrame + label array."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"": "missing", "None": "missing", "nan": "missing"})
    y = df["is_self_fixed"].values.astype(int)
    return records, df, y


# ── Phase 1: Optuna tuning ───────────────────────────────────────────────────

def run_optuna(df, y):
    """Tune LR, RF, XGBoost on 70/10/20 temporal split, 100 trials each."""
    N = len(df)
    split_70, split_80 = int(N * 0.7), int(N * 0.8)
    X_tr70, y_tr70 = df.iloc[:split_70], y[:split_70]
    X_val, y_val = df.iloc[split_70:split_80], y[split_70:split_80]
    pos70 = int(y_tr70.sum())
    neg70 = int(len(y_tr70) - pos70)

    tuned = {}

    def make_objective(model_name):
        def objective(trial):
            pre = make_preprocessor()
            if model_name == "logistic_regression":
                est = LogisticRegression(
                    C=trial.suggest_float("C", 0.01, 100, log=True),
                    solver="liblinear", max_iter=2000, random_state=SEED)
            elif model_name == "random_forest":
                est = RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
                    max_depth=trial.suggest_int("max_depth", 5, 30),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    class_weight="balanced", random_state=SEED, n_jobs=-1)
            elif model_name == "xgboost":
                est = XGBClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("lr", 0.005, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
                    scale_pos_weight=neg70 / pos70,
                    objective="binary:logistic", random_state=SEED, n_jobs=-1, verbosity=0)
            else:
                raise ValueError(model_name)
            pipe = Pipeline([("pre", pre), ("model", est)])
            pipe.fit(X_tr70, y_tr70)
            proba = pipe.predict_proba(X_val)[:, 1]
            _, mcc = optimal_mcc_threshold(y_val, proba)
            return mcc
        return objective

    for name in ["logistic_regression", "random_forest", "xgboost"]:
        print(f"\n  Tuning {name} (100 trials)...")
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(make_objective(name), n_trials=100, show_progress_bar=False)
        tuned[name] = study.best_params
        print(f"    Best val MCC: {study.best_value:.4f}")
        print(f"    Params: {study.best_params}")

    return tuned


# ── Phase 2: 9-split temporal validation ─────────────────────────────────────

def build_model(name, params, pos, neg):
    """Instantiate model with tuned params."""
    spw = neg / pos if pos > 0 else 1.0
    if name == "logistic_regression":
        return LogisticRegression(C=params.get("C", 1.0), solver="liblinear",
                                  max_iter=2000, random_state=SEED)
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 20),
            min_samples_split=params.get("min_samples_split", 5),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            class_weight="balanced", random_state=SEED, n_jobs=-1)
    elif name == "xgboost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("lr", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.6),
            reg_alpha=params.get("reg_alpha", 1.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            scale_pos_weight=spw,
            objective="binary:logistic", random_state=SEED, n_jobs=-1, verbosity=0)
    raise ValueError(name)


def run_temporal_validation(records, df, y, tuned_params):
    """Run 9-split expanding-window temporal validation for LR, RF, XGBoost + random baseline."""
    splits = make_temporal_splits(records)
    all_results = []

    for sp in splits:
        pct = sp.train_percentage / 100.0
        tr_idx = list(range(0, sp.train_size))
        te_idx = list(range(sp.train_size, sp.train_size + sp.test_size))
        df_train, y_train = df.iloc[tr_idx], y[tr_idx]
        df_test, y_test = df.iloc[te_idx], y[te_idx]
        pos, neg = int(y_train.sum()), int(len(y_train) - y_train.sum())
        print(f"\n  Split {sp.train_percentage}%: train={len(tr_idx)}, test={len(te_idx)}")

        # Random baseline
        dummy = DummyClassifier(strategy="stratified", random_state=SEED)
        dummy.fit(y_train.reshape(-1, 1), y_train)
        d_proba = dummy.predict_proba(y_test.reshape(-1, 1))[:, 1]
        d_pred = dummy.predict(y_test.reshape(-1, 1))
        all_results.append({
            "split_pct": pct, "model": "random_classifier",
            "auc": roc_auc_score(y_test, d_proba), "acc": accuracy_score(y_test, d_pred),
            "prec": precision_score(y_test, d_pred, zero_division=0),
            "rec": recall_score(y_test, d_pred), "f1": f1_score(y_test, d_pred),
            "mcc": matthews_corrcoef(y_test, d_pred),
        })

        for model_name in ["logistic_regression", "random_forest", "xgboost"]:
            pre = make_preprocessor()
            est = build_model(model_name, tuned_params[model_name], pos, neg)
            pipe = Pipeline([("pre", pre), ("model", est)])
            pipe.fit(df_train, y_train)
            proba = pipe.predict_proba(df_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(y_test, proba)
            mcc = matthews_corrcoef(y_test, pred)
            print(f"    {model_name:25s}: AUC={auc:.4f} MCC={mcc:.4f}")
            all_results.append({
                "split_pct": pct, "model": model_name,
                "auc": auc, "acc": accuracy_score(y_test, pred),
                "prec": precision_score(y_test, pred, zero_division=0),
                "rec": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
                "mcc": mcc,
            })

    return pd.DataFrame(all_results)


# ── Phase 3: SHAP on 80/20 split ─────────────────────────────────────────────

def run_shap(records, df, y, xgb_params):
    """Train XGBoost on 80% temporal split and compute SHAP."""
    splits = make_temporal_splits(records, train_percentages=[80])
    sp = splits[0]

    x_train = df.iloc[:sp.train_size].reset_index(drop=True)
    y_train = y[:sp.train_size]
    x_test = df.iloc[sp.train_size:].reset_index(drop=True)
    y_test = y[sp.train_size:]

    pre = make_preprocessor()
    pre.fit(x_train)
    raw_names = [n.split("__", 1)[1] if "__" in n else n for n in pre.get_feature_names_out()]
    display_names = [shorten_feature_name(n) for n in raw_names]

    x_train_proc = pd.DataFrame(pre.transform(x_train), columns=display_names)
    x_test_proc = pd.DataFrame(pre.transform(x_test), columns=display_names)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    model = XGBClassifier(
        n_estimators=xgb_params.get("n_estimators", 300),
        max_depth=xgb_params.get("max_depth", 5),
        learning_rate=xgb_params.get("lr", 0.05),
        subsample=xgb_params.get("subsample", 0.8),
        colsample_bytree=xgb_params.get("colsample_bytree", 0.6),
        reg_alpha=xgb_params.get("reg_alpha", 1.0),
        reg_lambda=xgb_params.get("reg_lambda", 1.0),
        scale_pos_weight=neg / pos,
        objective="binary:logistic", random_state=SEED, n_jobs=-1, verbosity=0,
    )
    print(f"\n  Training XGBoost ({len(display_names)} processed features)...")
    model.fit(x_train_proc.values, y_train)

    y_pred = model.predict(x_test_proc.values)
    y_score = model.predict_proba(x_test_proc.values)[:, 1]
    metrics = {
        "auc": roc_auc_score(y_test, y_score),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }
    print(f"  Test: AUC={metrics['auc']:.4f} MCC={metrics['mcc']:.4f}")

    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(x_test_proc)

    return model, x_test_proc, shap_values, metrics, display_names


# ── Phase 4: Generate figures ─────────────────────────────────────────────────

def generate_six_model_figure(struct_results_df, figures_dir):
    """Combine structured model results with existing text model results into six-model figure."""
    text_df = pd.read_csv(TEXT_MODEL_CSV)

    rows = []
    for _, r in text_df.iterrows():
        rows.append({
            "split_pct": r["train_percentage"] / 100.0,
            "model": r["model"],
            "auc": r["auc"], "f1": r["f1"], "mcc": r["mcc"],
        })
    text_results = pd.DataFrame(rows)

    combined = pd.concat([
        struct_results_df[["split_pct", "model", "auc", "f1", "mcc"]],
        text_results,
    ], ignore_index=True)

    model_order = ["random_classifier", "logistic_regression", "random_forest",
                    "xgboost", "textcnn", "bert"]
    model_labels = {
        "random_classifier": "Random baseline",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "textcnn": "TextCNN",
        "bert": "BERT",
    }
    model_colors = {
        "random_classifier": "#CFCECE",
        "logistic_regression": "#3775BA",
        "random_forest": "#8BCF8B",
        "xgboost": "#B64342",
        "textcnn": "#9A4D8E",
        "bert": "#42949E",
    }
    model_markers = {
        "random_classifier": "s",
        "logistic_regression": "D",
        "random_forest": "^",
        "xgboost": "o",
        "textcnn": "v",
        "bert": "P",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.2,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics_list = [("auc", "AUC"), ("f1", "F1"), ("mcc", "MCC")]

    for ax, (metric, ylabel) in zip(axes, metrics_list):
        for model in model_order:
            sub = combined[combined["model"] == model].sort_values("split_pct")
            if sub.empty:
                continue
            ax.plot(sub["split_pct"] * 100, sub[metric],
                    marker=model_markers.get(model, "o"), markersize=5,
                    linewidth=2, color=model_colors.get(model, "#333"),
                    label=model_labels.get(model, model))
        ax.set_xlabel("Training window (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(range(10, 100, 10))
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(w_pad=3)

    for ext in ["pdf", "png"]:
        out = figures_dir / f"six_model_temporal_comparison.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)

    for dest_ext in ["pdf", "png"]:
        src = figures_dir / f"six_model_temporal_comparison.{dest_ext}"
        dst = V4_DIR / f"six_model_temporal_comparison.{dest_ext}"
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            print(f"  Copied to {dst}")


def plot_ordered_shap_summary(x_test_df, shap_values, top15, output_path):
    """Draw a deterministic SHAP beeswarm using the same feature order as Figure 11."""
    feature_order = [str(f) for f in top15["feature"].tolist()]
    feature_to_idx = {feature: idx for idx, feature in enumerate(x_test_df.columns)}
    ordered_indices = [feature_to_idx[feature] for feature in feature_order if feature in feature_to_idx]
    if len(ordered_indices) != len(feature_order):
        missing = [feature for feature in feature_order if feature not in feature_to_idx]
        raise ValueError(f"Top-15 SHAP features missing from processed test data: {missing}")

    ordered_shap = shap_values[:, ordered_indices]
    ordered_values = x_test_df.iloc[:, ordered_indices].to_numpy()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    rng = np.random.default_rng(SEED)
    cmap = SHAP_CMAP

    for display_y, (feature, col_idx) in enumerate(zip(feature_order, range(len(feature_order)))):
        y = len(feature_order) - 1 - display_y
        x_vals = ordered_shap[:, col_idx]
        value_vals = ordered_values[:, col_idx]
        finite_values = value_vals[np.isfinite(value_vals)]
        if finite_values.size:
            v_min, v_max = np.nanpercentile(finite_values, [1, 99])
        else:
            v_min, v_max = 0.0, 1.0
        if np.isclose(v_min, v_max):
            colors = np.full_like(value_vals, 0.5, dtype=float)
        else:
            colors = np.clip((value_vals - v_min) / (v_max - v_min), 0, 1)

        order = np.argsort(np.abs(x_vals))
        jitter = rng.normal(0, 0.078, size=len(x_vals))
        ax.scatter(
            x_vals[order],
            y + jitter[order],
            c=colors[order],
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=10,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )

    ax.axvline(0, color=SHAP_AXIS_COLOR, linewidth=1.15, alpha=0.9, zorder=2)
    ax.grid(axis="x", color=SHAP_GRID_COLOR, linewidth=0.75, alpha=0.75)
    ax.grid(axis="y", color="#EEF3F5", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(len(feature_order))[::-1])
    ax.set_yticklabels(feature_order)
    ax.set_xlabel("SHAP value (impact on model output)", color=SHAP_AXIS_COLOR)
    ax.set_ylim(-0.8, len(feature_order) - 0.2)
    ax.tick_params(axis="both", colors=SHAP_AXIS_COLOR)
    ax.spines["left"].set_color(SHAP_AXIS_COLOR)
    ax.spines["bottom"].set_color(SHAP_AXIS_COLOR)

    x_abs = np.nanpercentile(np.abs(ordered_shap), 99.7)
    ax.set_xlim(-max(0.65, x_abs * 1.12), max(0.65, x_abs * 1.12))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.035, fraction=0.045)
    cbar.set_label("Feature value", rotation=270, labelpad=16, color=SHAP_AXIS_COLOR)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(colors=SHAP_AXIS_COLOR, length=0)
    cbar.outline.set_visible(False)

    fig.tight_layout(pad=0.45)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def generate_shap_figures(x_test_df, shap_values, metrics, figures_dir):
    """Generate SHAP summary dot plot and top-15 bar chart."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
    })

    n_feat = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)

    # Mean absolute SHAP table
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    pos_frac = (shap_values > 0).mean(axis=0)
    importance = pd.DataFrame({
        "feature": x_test_df.columns,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
        "positive_fraction": pos_frac,
    }).sort_values("mean_abs_shap", ascending=False)

    csv_path = RESULTS_DIR / "xgboost_shap_55feat_mean_abs.csv"
    importance.to_csv(csv_path, index=False, encoding="utf-8-sig")

    total_shap = float(importance["mean_abs_shap"].sum())
    top15 = importance.head(15)
    top15_pct = float(top15["mean_abs_shap"].sum()) / total_shap * 100 if total_shap > 0 else 0.0

    # Summary dot plot, explicitly locked to the same ranked top-15 as the bar chart.
    out_dot = figures_dir / "xgboost_shap_temporal.pdf"
    plot_ordered_shap_summary(x_test_df, shap_values, top15, out_dot)

    # Top-15 bar chart
    fig, ax = plt.subplots(figsize=(9.4, 6.8))
    # Shade bars by global SHAP contribution so higher-ranked features appear darker.
    bar_values = top15["mean_abs_shap"][::-1].to_numpy()
    bar_norm = plt.Normalize(vmin=float(bar_values.min()), vmax=float(bar_values.max()))
    ax.barh(
        top15["feature"][::-1],
        bar_values,
        color=SHAP_BAR_CMAP(bar_norm(bar_values)),
        edgecolor=SHAP_BAR_EDGE,
        linewidth=0.65,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12, color=SHAP_AXIS_COLOR)
    ax.set_title(
        f"Top-15 Feature Importance (XGBoost, {n_feat} features, temporal 80/20)",
        fontsize=9.5,
        color=SHAP_AXIS_COLOR,
        pad=8,
    )
    ax.grid(axis="x", color=SHAP_GRID_COLOR, linestyle="--", linewidth=0.75, alpha=0.85)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=SHAP_AXIS_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SHAP_AXIS_COLOR)
    ax.spines["bottom"].set_color(SHAP_AXIS_COLOR)
    fig.tight_layout()
    out_bar = figures_dir / "xgboost_shap_temporal_global_importance.pdf"
    fig.savefig(out_bar, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Copy publication figures to both the legacy V4 folder and the current JSS manuscript.
    import shutil
    for out_dir in [V4_DIR, JSS_MANUSCRIPT_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["xgboost_shap_temporal.pdf", "xgboost_shap_temporal_global_importance.pdf"]:
            src = figures_dir / fname
            dst = out_dir / fname
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied to {dst}")
        legacy_bar = out_dir / "xgboost_shap_global_shap_importance.pdf"
        if out_bar.exists():
            shutil.copy2(out_bar, legacy_bar)
            print(f"  Copied to {legacy_bar}")

    # Markdown summary
    lines = [
        f"# XGBoost SHAP – {n_feat} Features (no target-derived), Temporal 80/20",
        "",
        f"- Features: {n_feat} ({len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical).",
        f"- Removed: 6 features derived from is_self_fixed labels.",
        "",
        "## Test Metrics", "",
        "| AUC | Accuracy | Precision | Recall | F1 | MCC |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['auc']:.3f} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
        f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['mcc']:.3f} |",
        "", "## Top-15 Features", "",
        "| Rank | Feature | Mean |SHAP| | Share | Signed mean | % positive |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top15.itertuples(index=False), start=1):
        share = row.mean_abs_shap / total_shap * 100 if total_shap > 0 else 0.0
        lines.append(
            f"| {rank} | {row.feature} | {row.mean_abs_shap:.4f} | {share:.1f}% | "
            f"{row.mean_signed_shap:+.4f} | {row.positive_fraction * 100:.1f}% |"
        )
    lines.append(f"\nTop-15 cumulative: **{top15_pct:.1f}%** of total mean |SHAP|.")
    md_path = RESULTS_DIR / "xgboost_shap_55feat_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  SHAP summary: {md_path}")

    return importance


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== 55-Feature Pipeline (removed 6 target-derived features) ===")
    print(f"Numeric: {len(NUMERIC_FEATURES)}, Categorical: {len(CATEGORICAL_FEATURES)}, "
          f"Total: {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}")

    print("\n[1/4] Loading data...")
    records, df, y = load_data(DEFAULT_DATASET)
    print(f"  {len(records)} records loaded.")

    print("\n[2/4] Optuna hyperparameter tuning (LR, RF, XGBoost)...")
    tuned_params = run_optuna(df, y)

    params_path = RESULTS_DIR / "optuna_55feat_params.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(tuned_params, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved params: {params_path}")

    print("\n[3/4] 9-split temporal validation...")
    results_df = run_temporal_validation(records, df, y, tuned_params)

    csv_path = RESULTS_DIR / "temporal_55feat_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved results: {csv_path}")

    # Print summary tables
    print("\n  === 80/20 Split Results ===")
    r80 = results_df[results_df["split_pct"] == 0.8]
    for _, row in r80.iterrows():
        print(f"    {row['model']:25s}: AUC={row['auc']:.4f} MCC={row['mcc']:.4f} F1={row['f1']:.4f}")

    print("\n  === Mean Across 9 Splits ===")
    mean_df = results_df.groupby("model")[["auc", "mcc", "f1", "acc", "prec", "rec"]].mean()
    for model, row in mean_df.iterrows():
        print(f"    {model:25s}: AUC={row['auc']:.4f} MCC={row['mcc']:.4f} F1={row['f1']:.4f}")

    print("\n[4/4] SHAP analysis (80/20 split)...")
    model, x_test_df, shap_values, shap_metrics, display_names = run_shap(
        records, df, y, tuned_params["xgboost"])

    print("\n  Generating figures...")
    generate_six_model_figure(results_df, FIGURES_DIR)
    generate_shap_figures(x_test_df, shap_values, shap_metrics, FIGURES_DIR)

    print("\n=== Pipeline Complete ===")
    print(f"  Total features: {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}")
    print(f"  Results: {csv_path}")
    print(f"  SHAP: {RESULTS_DIR / 'xgboost_shap_55feat_summary.md'}")


if __name__ == "__main__":
    main()
