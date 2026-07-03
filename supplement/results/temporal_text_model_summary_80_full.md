# 时间切分文本模型摘要

## 实验设置

本实验使用 `f_comment` 作为文本输入，使用 `is_self_fixed` 作为标签。切分方式与结构化特征的时间验证一致，均遵循全局时间顺序。

TextCNN 复用原始超参数：MAX_WORDS=20000, MAX_LEN=100, EMB_DIM=300, FILTER_SIZES=(1, 2, 3, 4, 5), NUM_FILTERS=200, DROPOUT=0.5, LR=2e-05, BATCH=32。

BERT 复用原始超参数：MODEL=/root/satd_text_models/models/bert-base-uncased, MAX_LEN=128, EMBED_DIM=300, DROPOUT=0.3, LR=2e-05, BATCH=32。

TextCNN 词表只在每个时间切分的训练窗口上拟合，不使用未来测试集评论。

## 结果

| 模型 | 训练比例 | 训练 n | 验证 n | 测试 n | AUC | F1 | MCC | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| textcnn | 80% | 29,160 | 3,240 | 8,101 | 0.534 | 0.694 | 0.029 |  |
| bert | 80% | 29,160 | 3,240 | 8,101 | 0.535 | 0.703 | 0.046 |  |
