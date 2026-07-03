# 图 12 SHAP Waterfall 摘要

- 模型：55 特征 XGBoost，时间 80/20 切分。
- 实例选择：在正确预测为 self-repaid 的 hold-out 正例中，选择预测概率最接近正确正例中位数的样本。
- 测试集索引：4940
- 全局数据集索引：37340
- 真实标签：1
- 预测标签：1
- 预测概率：0.6986
- 基准值：-0.0010 log-odds
- 最终值：0.8409 log-odds

## 局部贡献

| 排名 | 特征 | 处理后取值 | SHAP 贡献 |
|---:|---|---:|---:|
| 1 | repaid_satd_count | 0.5020 | +0.3029 |
| 2 | ownership_x_active_commits | 0.1673 | +0.2565 |
| 3 | ownership | 1.0000 | +0.1727 |
| 4 | p_active_commits | 0.1942 | -0.1601 |
| 5 | file_frequency | 0.0039 | +0.1420 |
| 6 | file_authors | 0.0120 | +0.1293 |
| 7 | comment_keyword_type_OTHER | 1.0000 | -0.1220 |
| 8 | satd_quality_score_1 | 1.0000 | -0.1188 |
| 9 | p_active_days | 0.8791 | +0.0929 |
| 10 | last_repay_satd_days | 0.0003 | +0.0912 |
| 11 | length | 0.0359 | -0.0890 |
| 12 | active_developers | 0.0472 | +0.0790 |
| 13 | Other 58 features |  | +0.0654 |
