# 特征组消融实验

## 实验设计

本实验训练三个 XGBoost 变体。三者使用**完全相同的超参数**
（来自 55 特征 Optuna 调参结果），并采用同一套时间顺序验证协议。

| 组别 | 特征 | 说明 |
|-------|----------|-------------|
| A | Developer + Cross-dimensional (21) | 行为惯性信号：ownership、开发者活跃度、SATD 历史 |
| B | Code + Comment + Project (34) | 上下文信号：代码结构、评论文本属性、项目上下文 |
| C | All 55 features | 完整模型，作为参考 |

## 主要 80%/20% 时间顺序切分结果

| 特征组 | AUC | Accuracy | Precision | Recall | F1 | MCC |
|---------------|-----|----------|-----------|--------|----|----|
| Developer+Cross (21) | 0.706 | 0.640 | 0.767 | 0.546 | 0.638 | 0.316 |
| Code+Comment+Project (34) | 0.694 | 0.633 | 0.779 | 0.514 | 0.619 | 0.316 |
| All 55 features | 0.731 | 0.655 | 0.753 | 0.606 | 0.672 | 0.326 |

## 9 个 expanding-window 切分的平均结果

| 特征组 | Mean AUC | Mean MCC | Mean F1 |
|---------------|----------|----------|---------|
| Developer+Cross (21) | 0.677 | 0.261 | 0.611 |
| Code+Comment+Project (34) | 0.633 | 0.195 | 0.592 |
| All 55 features | 0.691 | 0.266 | 0.636 |

## 解释

该消融实验量化了行为惯性特征（developer history features）与上下文信号
（code/comment/project features）的相对贡献。如果 A 组
（Developer+Cross）的性能接近完整模型，说明模型主要依赖过往行为模式，
而不是主要依赖结构性上下文。
