# LLM 派生特征敏感性分析

## 实验设计

该敏感性模型从最终 55 特征 XGBoost 设置中移除六个由 DeepSeek-V3 派生的特征：

- `satd_quality_score` / 论文名 `quality_score`
- `developer_past_feature_ratio` / 论文名 `past_feature_ratio`
- `developer_past_bugfix_ratio` / 论文名 `past_bugfix_ratio`
- `developer_past_refactor_ratio` / 论文名 `past_refactor_ratio`
- `developer_past_cleanup_ratio` / 论文名 `past_cleanup_ratio`
- `project_readme_score` / 论文名 `readme_score`

实验复用同一组调好的 XGBoost 超参数和同一套时间顺序切分定义。

## 汇总结果

| 设置 | 80/20 AUC | 80/20 MCC | Mean AUC | Mean MCC |
|---|---:|---:|---:|---:|
| Full 55-feature XGBoost | 0.731 | 0.326 | 0.691 | 0.266 |
| No LLM-derived features (49) | 0.731 | 0.324 | 0.689 | 0.261 |

## 各切分结果

| 训练窗口 | Full AUC | No-LLM AUC | Full MCC | No-LLM MCC |
|---:|---:|---:|---:|---:|
| 10% | 0.657 | 0.655 | 0.214 | 0.218 |
| 20% | 0.634 | 0.621 | 0.179 | 0.153 |
| 30% | 0.687 | 0.683 | 0.246 | 0.244 |
| 40% | 0.687 | 0.685 | 0.257 | 0.250 |
| 50% | 0.706 | 0.707 | 0.287 | 0.288 |
| 60% | 0.704 | 0.705 | 0.289 | 0.288 |
| 70% | 0.725 | 0.721 | 0.332 | 0.318 |
| 80% | 0.731 | 0.731 | 0.326 | 0.324 |
| 90% | 0.689 | 0.688 | 0.262 | 0.269 |
