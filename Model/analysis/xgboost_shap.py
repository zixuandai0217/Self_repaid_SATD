import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    f1_score
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap

# === Data Loading ===
MERGED_DATA = '../../Dataset/data/merged_data_40501_updated copy 2.json'
with open(MERGED_DATA, 'r', encoding='utf-8') as f:
    records = json.load(f)
df = pd.DataFrame(records)

# === Basic Column Definitions ===
target_col = "is_self_fixed"
drop_cols = ["satd_id", "project_name", "survival_days"]

# === Read Feature Removal Files ===
VIF_DROP_FILE = 'vif_drop.txt'
if os.path.exists(VIF_DROP_FILE):
    with open(VIF_DROP_FILE, 'r', encoding='utf-8') as f:
        vif_dropped_features = [line.strip() for line in f if line.strip()]
    drop_cols += vif_dropped_features
else:
    print("Proceeding without VIF feature removal.")

NO_SIGNIF_FILE = 'no_significant_features.txt'
if os.path.exists(NO_SIGNIF_FILE):
    with open(NO_SIGNIF_FILE, 'r', encoding='utf-8') as f:
        nonsig_features = [line.strip() for line in f if line.strip()]
    drop_cols += nonsig_features
else:
    print("Proceeding without no_significant_features.txt feature removal.")

drop_cols = list(set(drop_cols))

# === Feature Column Classification ===
feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
cat_cols = [c for c in ["type", "quality_score", "add_is_weekend_or_night"] if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

X = df[feature_cols]
y = df[target_col]

# === Preprocessor ===
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

# === Model Definition ===
model = Pipeline([
    ("preproc", preprocessor),
    ("xgb", XGBClassifier(
        learning_rate=0.01,
        max_depth=30,
        n_estimators=500,
        scale_pos_weight=np.float64(1.4916301458438106),
        colsample_bytree=0.4,
        subsample=1.0,
        eval_metric="logloss",
        random_state=42,
    ))
])

base = os.path.splitext(os.path.basename(__file__))[0]
out_txt = f"{base}.txt"

# === StratifiedKFold Cross-Validation ===
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_list, acc_list, prec_list, recall_list, f1_list = [], [], [], [], []

with open(out_txt, "w", encoding="utf-8") as fout:
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pos_count = int((y_test == 1).sum())
        neg_count = int((y_test == 0).sum())

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        auc_list.append(auc)
        acc_list.append(acc)
        prec_list.append(prec)
        recall_list.append(recall)
        f1_list.append(f1)

        fout.write(
            f"Fold {fold}, Positive:{pos_count}, Negative:{neg_count}, "
            f"AUC:{auc:.4f}, ACC:{acc:.4f}, PREC:{prec:.4f}, RECALL:{recall:.4f}, F1:{f1:.4f}\n"
        )
        print("-" * 60)
        print(f"Fold {fold} results: auc:{auc:.4f}, acc:{acc:.4f}, prec:{prec:.4f}, recall:{recall:.4f}, f1:{f1:.4f}")

    avg_auc = np.mean(auc_list)
    avg_acc = np.mean(acc_list)
    avg_prec = np.mean(prec_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    fout.write(
        f"\nAverage metrics: AUC:{avg_auc:.4f}, ACC:{avg_acc:.4f}, PREC:{avg_prec:.4f}, "
        f"RECALL:{avg_recall:.4f}, F1:{avg_f1:.4f}\n"
    )

print(f"10-fold cross-validation completed, results saved to: {out_txt}")

# === SHAP Global Interpretation ===
preprocessor = model.named_steps['preproc']
new_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_features = num_cols + list(new_cat_cols)

X_test_processed = preprocessor.transform(X_test)
X_test_df = pd.DataFrame(X_test_processed, columns=all_features, index=X_test.index)

xgb_model = model.named_steps['xgb']
X_train_proc = preprocessor.transform(X_train)
X_train_df = pd.DataFrame(X_train_proc, columns=all_features)

explainer = shap.TreeExplainer(xgb_model, X_train_df, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test_df)

plt.figure()
shap.summary_plot(shap_values, X_test_df, plot_type='dot', show=False)
plt.tight_layout()
plt.savefig(f"{base}.pdf", bbox_inches='tight')
plt.close()

# === Calculate Feature Importance ===
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "feature": X_test_df.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

# === Validate Top15 Contribution Rate ===
total_shap = shap_importance_df["mean_abs_shap"].sum()
if total_shap == 0:
    top15_contrib_ratio = 0
else:
    top15_shap = shap_importance_df["mean_abs_shap"].head(15).sum()
    top15_contrib_ratio = top15_shap / total_shap

with open(out_txt, "a", encoding="utf-8") as fout:
    fout.write(f"\nTop15 features SHAP contribution rate: {top15_contrib_ratio:.2%}\n")
    if top15_contrib_ratio < 0.80:
        fout.write("Warning: Contribution rate <80%, recommend analyzing more features\n")
    else:
        fout.write("Conclusion: Top 15 features sufficiently cover model decision logic\n")

# === Save Results ===
shap_importance_df.to_csv(f"{base}_mean_abs_shap.csv", index=False, encoding="utf-8-sig")

# === Plot Top15 Feature Importance ===
plt.figure(figsize=(12, 8))
top_features = shap_importance_df.head(15)
plt.barh(
    top_features["feature"][::-1],
    top_features["mean_abs_shap"][::-1],
    color=plt.cm.coolwarm(np.linspace(0, 1, 15))
)
# plt.title(f"Top15 (: {top15_contrib_ratio:.1%})", fontsize=14)
plt.xlabel("Mean(|SHAP value|)", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{base}_global_shap_importance.pdf", bbox_inches='tight')
plt.close()

print(f"""
Complete results saved to:
- Cross-validation metrics: {out_txt}
- SHAP values: {base}_mean_abs_shap.csv
- Visualization: {base}.pdf
- Feature importance plot: {base}_global_shap_importance.pdf
""")
