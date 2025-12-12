import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

MERGED_DATA = '../../Dataset/data/merged_data_40501_updated.json'
with open(MERGED_DATA, 'r', encoding='utf-8') as f:
    records = json.load(f)
df = pd.DataFrame(records)

# Define feature columns
target_col = "is_self_fixed"
drop_cols = ["satd_id", "project_name", "satd_survival_days"]
feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
cat_cols = [c for c in ["satd_type", "satd_quality_score", "satd_add_is_weekend_or_night"] if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

df_copy = df.copy()

# One-hot encode categorical variables
if cat_cols:
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    ohe_features = ohe.fit_transform(df_copy[cat_cols])
    new_cat_cols = ohe.get_feature_names_out(cat_cols)
    df_ohe = pd.DataFrame(ohe_features, columns=new_cat_cols, index=df_copy.index)

    # Merge encoded columns
    df_copy = df_copy.drop(columns=cat_cols)
    df_copy = pd.concat([df_copy, df_ohe], axis=1)

    # Update feature columns
    feature_cols = [c for c in df_copy.columns if c not in drop_cols + [target_col]]

X = df_copy[feature_cols]

# Calculate initial VIF values
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

base = os.path.splitext(os.path.basename(__file__))[0]

with open(f'{base}.txt', 'w', encoding='utf-8') as f:
    f.write("Initial VIF Values:\n")
    f.write(vif_data.sort_values(by="VIF", ascending=False).to_string())
    f.write("\n\n")

# Iteratively remove features with VIF greater than 5
threshold = 5

drop_file_path = f'{base}_drop.txt'

# Clear or create file
    with open(drop_file_path, 'w', encoding='utf-8') as f:
        pass  # Clear file content or create new file
round = 1

while vif_data.loc[vif_data["VIF"] > threshold, :].shape[0] > 0:
    max_vif_idx = vif_data.loc[vif_data["VIF"] > threshold, "VIF"].idxmax()
    removed_feature = vif_data.loc[max_vif_idx, "Feature"]

    # Update feature matrix
    X = X.drop(columns=[removed_feature])
    with open(f'{base}_drop.txt', 'a', encoding='utf-8') as f:
        f.write(f"{removed_feature}\n")

    # Recalculate VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    with open(f'{base}.txt', 'a', encoding='utf-8') as f:
        f.write(f"round {round}: Removed feature '{removed_feature}'\n\n")
        f.write("-" * 60 + "\n")
        f.write(vif_data.sort_values(by="VIF", ascending=False).to_string())
        f.write("\n\n")
    round += 1

# Final feature list
final_features = list(X.columns)

# Plot heatmap
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(24, 20))
sns.heatmap(
    X.corr(),
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "VIF Value"}
)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
output = f'{base}.pdf'
plt.savefig(output, dpi=300, bbox_inches='tight')
plt.close()