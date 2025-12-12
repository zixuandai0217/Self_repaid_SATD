import os
import json
from numpy import rec
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import (
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    f1_score
)

from xgboost import XGBClassifier

MERGED_DATA = '../../Dataset/data/merged_data_40501_updated.json'
with open(MERGED_DATA, 'r', encoding='utf-8') as f:
    records = json.load(f)
df = pd.DataFrame(records)

# 
target_col = "is_self_fixed"
drop_cols = ["satd_id", "satd_survival_days"]
feature_cols = [c for c in df.columns if c not in drop_cols + [target_col, "project_name"]]
cat_cols = [c for c in ["satd_type", "satd_quality_score", "satd_add_is_weekend_or_night"] if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

X      = df[feature_cols]
y      = df[target_col]

groups = df["project_name"] #project_name

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
    ("xgb", XGBClassifier(
        learning_rate=0.01, # : 0.01
        max_depth=30, # : 30
        n_estimators=500, # : 100
        scale_pos_weight=np.float64(1.4916301458438106), # : np.float64(1.4916301458438106)
        colsample_bytree=0.4, # 
        subsample= 1.0,
        eval_metric="logloss",
        random_state=42,
    ))
])

base    = os.path.splitext(os.path.basename(__file__))[0]
out_txt = f"{base}.txt"

# 
LOGO = LeaveOneGroupOut()

# 
auc_list, acc_list, prec_list, recall_list, f1_list = [], [], [], [], []

with open(out_txt, "w", encoding="utf-8") as fout:
    for fold, (train_idx, test_idx) in enumerate(LOGO.split(X, y, groups=groups), start=1): # 
        print(f"Fold {fold}")
        project_left_out = groups.iloc[test_idx].unique()[0]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 
        pos_count = int((y_test == 1).sum())
        neg_count = int((y_test == 0).sum())
        
        #  & 
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = model.predict(X_test)
        
        # 
        auc   = roc_auc_score(y_test, y_proba)
        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        
        # 
        auc_list.append(auc)
        acc_list.append(acc)
        prec_list.append(prec)
        recall_list.append(recall)
        f1_list.append(f1)
        
        # ，
        print("-" * 60)
        fout.write(
            f"{fold}（：{project_left_out}），"
            f":{pos_count}，"
            f":{neg_count}，"
            f"AUC:{auc:.4f}，"
            f"Accuracy:{acc:.4f}，"
            f"Precision:{prec:.4f}，"
            f"Recall:{recall:.4f}，"
            f"F1:{f1:.4f}\n"
        )
        print(f"{fold} {pos_count} {neg_count} ：auc:{auc:.4f}, acc:{acc:.4f}, prec:{prec:.4f}, recall:{recall:.4f}, f1:{f1:.4f}")
    
    # 
    avg_auc  = sum(auc_list) / len(auc_list)
    avg_acc  = sum(acc_list) / len(acc_list)
    avg_prec = sum(prec_list) / len(prec_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_f1   = sum(f1_list) / len(f1_list)
    
    # 
    fout.write(
        f"，"
        f"AUC:{avg_auc:.4f}，"
        f"Accuracy:{avg_acc:.4f}，"
        f"Precision:{avg_prec:.4f}，"
        f"Recall:{avg_recall:.4f}，"
        f"F1:{avg_f1:.4f}\n"
    )
    print(f"：AUC:{avg_auc:.4f}，Accuracy:{avg_acc:.4f}，Precision:{avg_prec:.4f}， Recall: {avg_recall:.4f}， F1:{avg_f1:.4f}")

print(f"，：{out_txt}")