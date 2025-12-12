import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, f1_score, recall_score,
    classification_report
)
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# 1. 
MERGED_DATA = '../../Dataset/data/merged_data_40501.json'
with open(MERGED_DATA, 'r', encoding='utf-8') as f:
    records = json.load(f)

df = pd.DataFrame(records)

# 2. 
target_col = "is_self_fixed"
drop_cols = ["satd_id", "project_name", "satd_survival_days"]
feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
cat_cols = [c for c in ["satd_type", "satd_quality_score", "satd_add_is_weekend_or_night"] if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

X = df[feature_cols]
y = df[target_col]

# 3.  60:20:20 
#  train_val (80%)  test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#  train_val  75:25  train (60%)  val (20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# ：/
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos


# preprocessor，

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_cols), # minmax
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols), # one-hot
    ]
)

# 
model = Pipeline([
    ("preproc", preprocessor),
    ("rf", RandomForestClassifier(
        random_state=42
    ))
])

param_grid = {
    'rf__n_estimators': [100, 300, 500],          # 
    'rf__max_depth': [10, 20, 30],          # 
    'rf__min_samples_split': [2, 5, 10],          # 
    'rf__min_samples_leaf': [1, 2, 4],            # 
    'rf__max_features': ['auto', 'sqrt', 0.6],    # 
    'rf__bootstrap': [True, False]                # 
}

# 6. Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# 
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print(f"Best cross-val AUC: {grid_search.best_score_:.3f}")

# 
base = os.path.splitext(os.path.basename(__file__))[0]
with open(f"{base}_best_params.txt", 'w', encoding='utf-8') as f:
    f.write(f"Best parameters: {grid_search.best_params_}\n")
    f.write(f"Best cross-val AUC: {grid_search.best_score_:.3f}\n")

# 7. 
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
y_val_proba = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)

with open(f"{base}_validation_report.txt", 'w', encoding='utf-8') as f:
    f.write(f"Validation AUC = {val_auc:.3f}\n")
    f.write("Validation Classification Report:\n")
    f.write(classification_report(y_val, y_val_pred))

# 8. 
y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)
with open(f"{base}_test_report.txt", 'w', encoding='utf-8') as f:
    f.write(f"Test AUC = {test_auc:.3f}\n")
    f.write("Test Classification Report:\n")
    f.write(classification_report(y_test, y_test_pred))

# 9.  ROC （）
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Test AUC = {test_auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{base}_roc.pdf', bbox_inches='tight')
plt.close()
print(f"Saved ROC curve to: {base}_roc.pdf")