# SATD 人工验证

## 输入

- 人工验证 CSV：`Self Repaid SATD/code/modeling/analysis/366_human_eval_sample_384.csv`
- 总体规模：40,501
- 样本规模：384

## 抽样依据

验证样本量按照有限总体比例估计公式确定，设置为 95% 置信水平、
5% 误差范围，并采用保守比例 `p = 0.5`。当 `N = 40,501` 时，
所需样本量约为 380.56。因此，384 条记录的样本满足该抽样设计。

## 结果

- 人工确认的真实 SATD 记录：366/384
- Precision：95.3%
- Wilson 95% CI：92.7% 到 97.0%

## 可直接写入论文的表述

为评估最终 SATD 实例的数据质量，我们从 40,501 条实例中随机抽取了
384 条记录。样本量基于 95% 置信水平、5% 误差范围以及保守总体比例
`p = 0.5` 确定。人工检查确认 384 条样本中有 366 条是真实 SATD 实例，
得到 95.3% 的 precision，Wilson 95% 置信区间为 92.7% 到 97.0%。
该结果表明，最终 SATD 数据集在本文实证分析中具有较高 precision。

## 建议放置位置

- Data Quality 小节：报告抽样依据和 precision。
- Threats to Validity 小节：说明人工验证降低了对最终数据集中 SATD false positive 的担忧，同时承认该 precision-oriented 样本没有估计 recall。
