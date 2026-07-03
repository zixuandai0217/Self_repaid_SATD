# 身份标签敏感性分析

## 实验设计

该实验从 `code/data_preparation/data/satd.db` 中重建 add/remove commit 身份，
再通过 `satd_id + remove_date` 与清洗后的 40,501 行 V4 数据集合并。
该实验不使用人工标注。

标签变体如下：

- `original`：作者名完全匹配 OR 作者邮箱完全匹配，与 V4 数据集一致。
- `email_only`：标准化后的邮箱地址必须匹配。
- `strict_normalized`：标准化后的邮箱 OR 标准化后的全名匹配。
- `expanded_normalized`：`strict_normalized` 匹配 OR 非通用邮箱 local part 匹配。

## 标签分布与一致性

| 变体 | Self count | Self rate | Flips vs original | Positive->negative | Negative->positive | Agreement | Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 23199 | 0.573 | 0 | 0 | 0 | 1.000 | 1.000 |
| email_only | 22483 | 0.555 | 716 | 716 | 0 | 0.982 | 0.964 |
| strict_normalized | 23202 | 0.573 | 3 | 0 | 3 | 1.000 | 1.000 |
| expanded_normalized | 23282 | 0.575 | 83 | 0 | 83 | 0.998 | 0.996 |

## 身份风险指标

| 指标 | Count | Rate |
|---|---:|---:|
| identity_bot_or_noreply | 2803 | 0.069 |
| identity_email_missing | 1 | 0.000 |
| db_exact_label_disagreement | 6 | 0.000 |

## Survival 摘要

| 变体 | 组别 | Count | Median days | Mean days |
|---|---|---:|---:|---:|
| original | self | 23199 | 18.0 | 147.0 |
| original | non_self | 17302 | 206.0 | 603.1 |
| email_only | self | 22483 | 16.0 | 128.7 |
| email_only | non_self | 18018 | 214.0 | 607.8 |
| strict_normalized | self | 23202 | 18.0 | 147.0 |
| strict_normalized | non_self | 17299 | 206.0 | 603.2 |
| expanded_normalized | self | 23282 | 18.0 | 148.4 |
| expanded_normalized | non_self | 17219 | 206.0 | 603.3 |

## XGBoost temporal sensitivity

| 变体 | 80/20 AUC | 80/20 MCC | Mean AUC | Mean MCC | Mean F1 |
|---|---:|---:|---:|---:|---:|
| original | 0.731 | 0.326 | 0.691 | 0.266 | 0.636 |
| email_only | 0.736 | 0.337 | 0.683 | 0.259 | 0.623 |
| strict_normalized | 0.731 | 0.326 | 0.691 | 0.266 | 0.637 |
| expanded_normalized | 0.734 | 0.328 | 0.691 | 0.265 | 0.638 |

## 解释

`email_only` 是最严格的 lower-bound 标签定义，因为它会丢弃不同邮箱地址下的同名匹配。
标准化变体估计在大小写、标点、重音符号以及 GitHub noreply 标准化之后，
exact-match 规则会发生多大变化。如果性能或 survival 结果发生较大变化，
说明论文结论对身份匹配规则敏感；如果变化较小，则支持结论的稳健性。
