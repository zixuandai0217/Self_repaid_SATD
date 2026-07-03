# Log-rank 检验（RQ2 survival comparison）

## 设置

脚本：`scripts/07_logrank_survival.py`

数据：`data/merged_data_40501_temporal_sorted_by_add_date.json`（N = 40,501）。
所有记录都有 `remove_date`，因此不存在右删失；每条观测都是一个事件。
该检验使用标准两样本 log-rank 统计量比较 self-repaid SATD
（`is_self_fixed = 1`，N = 23,199）和 non-self-repaid SATD
（`is_self_fixed = 0`，N = 17,302）的 survival 分布，方差按
hypergeometric at-risk 公式计算。

## 结果

| 指标 | 值 |
|---|---:|
| χ²(1) | 7,163.32 |
| p-value | < 1 × 10⁻³⁰⁰（IEEE-754 下等效为 0） |
| Observed events, self-repaid | 23,199 |
| Expected events, self-repaid (under H₀) | 15,358.7 |
| Observed − Expected (self-repaid) | +7,840.3 |
| Variance of (O − E) | 8,581.3 |

## 与论文中已报告的 Mann-Whitney U 的交叉验证

Mann-Whitney U = 1.0098 × 10⁸，p ≈ 0（同样出现下溢模式）。
两个检验都以相同方向拒绝原假设：self-repaid SATD 被偿还得更快，
且显著性强度一致。由于数据集中没有删失，基于 `satd_survival_days`
的 log-rank 检验和 Mann-Whitney U 检验预期应保持一致。
这里补充 log-rank 结果，是为了提供 survival analysis 领域更标准的检验，
与论文中已报告的 Mann-Whitney U 检验互为补充。

## 可直接写入论文的表述

> We further confirmed this difference using a two-sample log-rank test~\cite{peto1972asymptotically},
> which is the conventional significance test for comparing survival distributions.
> The log-rank statistic was $\chi^2 = 7{,}163.32$ on 1 degree of freedom with $p < 10^{-300}$,
> in agreement with the Mann-Whitney U test reported above. Because every SATD instance
> in our dataset has an observed repayment date (no right-censoring), the two tests are
> expected to lead to the same conclusion, and both strongly reject the null hypothesis
> that the two groups share the same survival distribution.
