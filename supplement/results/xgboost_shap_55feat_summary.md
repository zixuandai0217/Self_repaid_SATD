# XGBoost SHAP – 55 Features (no target-derived), Temporal 80/20

- Features: 55 (49 numeric + 6 categorical).
- Removed: 6 features derived from is_self_fixed labels.

## Test Metrics

| AUC | Accuracy | Precision | Recall | F1 | MCC |
|---:|---:|---:|---:|---:|---:|
| 0.731 | 0.655 | 0.753 | 0.606 | 0.672 | 0.326 |

## Top-15 Features

| Rank | Feature | Mean |SHAP| | Share | Signed mean | % positive |
|---:|---|---:|---:|---:|---:|
| 1 | ownership | 0.1765 | 7.4% | -0.0022 | 63.1% |
| 2 | ownership_x_active_commits | 0.1702 | 7.2% | -0.0260 | 40.6% |
| 3 | repay_ratio | 0.1297 | 5.5% | +0.0197 | 43.2% |
| 4 | active_developers | 0.1278 | 5.4% | -0.0873 | 36.6% |
| 5 | file_frequency | 0.1265 | 5.3% | -0.0051 | 63.4% |
| 6 | repaid_satd_count | 0.1009 | 4.2% | +0.0461 | 37.6% |
| 7 | last_repay_satd_days | 0.0930 | 3.9% | -0.0114 | 59.9% |
| 8 | is_top_committer | 0.0859 | 3.6% | -0.0294 | 82.1% |
| 9 | p_active_days | 0.0849 | 3.6% | +0.0473 | 44.6% |
| 10 | d_active_commits | 0.0812 | 3.4% | +0.0207 | 18.5% |
| 11 | p_active_commits | 0.0777 | 3.3% | +0.0073 | 67.6% |
| 12 | d_last_commit_days | 0.0603 | 2.5% | -0.0224 | 52.2% |
| 13 | files | 0.0600 | 2.5% | +0.0498 | 84.7% |
| 14 | past_feature_ratio | 0.0587 | 2.5% | +0.0192 | 78.8% |
| 15 | file_authors | 0.0581 | 2.4% | +0.0090 | 62.4% |

Top-15 cumulative: **62.7%** of total mean |SHAP|.