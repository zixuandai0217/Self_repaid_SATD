import json
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

MERGED_DATA = '../../Dataset/data/merged_data_40501_updated.json'
with open(MERGED_DATA, 'r', encoding='utf-8') as f:
    records = json.load(f)
df = pd.DataFrame(records)

# Define feature columns
target_col = "is_self_fixed"
drop_cols = ["satd_id", "project_name"]
feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
cat_cols = [c for c in ["satd_type", "satd_quality_score", "satd_add_is_weekend_or_night"] if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

results = []

# Loop through feature columns
for col in feature_cols:
    if col in num_cols:  # Numerical features: Mann-Whitney U test
        group0 = df[df[target_col] == 0][col].dropna()
        group1 = df[df[target_col] == 1][col].dropna()

        if len(group0) == 0 or len(group1) == 0:
            p_val = np.nan
            test_type = 'Mann-Whitney U'
            effect_size = np.nan
            stats_group0 = [np.nan] * 5
            stats_group1 = [np.nan] * 5
        else:
            #  Mann-Whitney U  Cliff's Delta
            u_stat, p_val = mannwhitneyu(group0, group1, alternative='two-sided')
            test_type = 'Mann-Whitney U'
            n0, n1 = len(group0), len(group1)
            cliff_delta = (2 * u_stat - n0 * n1) / (n0 * n1)
            effect_size = cliff_delta

            # Calculate statistics for positive and negative samples
            stats_group0 = [
                group0.mean(),       # Mean
                group0.median(),     # Median
                group0.std(),        # Std Dev
                group0.quantile(0.25),  # 25% Quartile
                group0.quantile(0.75)   # 75% Quartile
            ]
            stats_group1 = [
                group1.mean(),
                group1.median(),
                group1.std(),
                group1.quantile(0.25),
                group1.quantile(0.75)
            ]

    else:  # Categorical features: construct contingency table
        contingency_table = pd.crosstab(df[col], df[target_col])
        if contingency_table.shape == (2, 2):  # 2x2 contingency table
            try:
                chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
            except Exception as e:
                p_val = np.nan
                test_type = 'Chi-square'
                effect_size = np.nan
            else:
                if np.any(expected < 5):  # Fisher's exact test
                    _, p_val = fisher_exact(contingency_table)
                    test_type = 'Fisher Exact'
                    # Odds Ratio
                    table = contingency_table.values.copy()
                    if np.any(table == 0):  # Haldane-Anscombe correction
                        table = table + 0.5
                    a, b = table[0, 0], table[0, 1]
                    c, d = table[1, 0], table[1, 1]
                    odds_ratio = (a * d) / (b * c)
                    effect_size = odds_ratio
                else:
                    p_val = p_chi2
                    test_type = 'Chi-square'
                    # Cramer's V
                    n = contingency_table.values.sum()
                    r, c_ = contingency_table.shape
                    min_dim = min(r-1, c_-1)
                    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
                    effect_size = v
        else:  # Non-2x2 contingency table
            try:
                chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                n_total = contingency_table.values.sum()
                if (np.sum(expected < 5) / expected.size > 0.2) or (np.any(expected < 1)):
                    _, p_val, _, _ = chi2_contingency(
                        contingency_table, 
                        simulate_p_value=True,
                        replicates=10000
                    )
                    test_type = 'Chi-square (simulated)'
                else:
                    test_type = 'Chi-square'

                r, c_ = contingency_table.shape
                min_dim = min(r-1, c_-1)
                effect_size = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else np.nan
            except Exception as e:
                p_val = np.nan
                test_type = 'Chi-square'
                effect_size = np.nan

        stats_group0 = [np.nan] * 5  # Categorical features, do not calculate statistics
        stats_group1 = [np.nan] * 5

    significant = 'Yes' if p_val < 0.05 else 'No'

    # Add results
    results.append({
        'feature': col,
        'test_type': test_type,
        'p_value': p_val,
        'significant': significant,
        'effect_size': effect_size,
        **{
            f'group0_{stat}': val for stat, val in zip(
                ['Mean', 'Median', 'Std_Dev', 'Q25', 'Q75'], stats_group0
            )
        },
        **{
            f'group1_{stat}': val for stat, val in zip(
                ['Mean', 'Median', 'Std_Dev', 'Q25', 'Q75'], stats_group1
            )
        }
    })

# Organize results into DataFrame
results_df = pd.DataFrame(results)

# Sort
results_df['first_letter'] = results_df['feature'].str[0]
results_df['abs_effect_size'] = results_df['effect_size'].abs()
results_df = results_df.sort_values(
    ['first_letter', 'abs_effect_size'],
    ascending=[True, False]
).drop(columns=['first_letter', 'abs_effect_size'])

# Write to output file
base = os.path.splitext(os.path.basename(__file__))[0]
output = f'{base}.txt'
with open(output, 'w', encoding='utf-8') as f:
    f.write(results_df.to_string(
        index=False,
        formatters={
            'p_value': lambda x: f'{x:.2e}' if pd.notna(x) else 'NaN',
            'effect_size': lambda x: f'{x:.3f}' if pd.notna(x) else 'NaN',
            **{
                f'group{g}_{stat}': lambda x: f'{x:.3f}' if pd.notna(x) else ''
                for g in [0, 1] for stat in ['Mean', 'Median', 'Std_Dev', 'Q25', 'Q75']
            }
        }
    ))
no_sig = results_df.loc[results_df['significant'] == 'No', 'feature']
with open('no_significant_features.txt', 'w', encoding='utf-8') as f:
    for feat in no_sig:
        f.write(f"{feat}\n")