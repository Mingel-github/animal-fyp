# IDEA-068：年龄敏感声学归纳偏置与 AST 融合

## 1. 研究问题

冻结 AudioSet-AST 在 MeowAgeNet 上已经提供较强的全局表示，也能在 CatMeows 上迁移未见猫的情境信息；但犬年龄外部筛选显示，它对发育阶段的跨个体表示较弱。IDEA-068 检验一个与数据集标签调配不同的问题：显式加入与发声生理相关的 F0、周期性和声质轨迹，能否稳定补充 AST 偏重情境与声音事件的表示。

本轮只做内部 train→validation 机制筛选，不访问任何 outer-test 预测。历史结果已知，因此它是探索性、事后提出的模块实验，不作为独立确认性证据。

## 2. 年龄敏感声学特征

所有特征逐叫声提取，不使用年龄标签，也不按类别调整参数。音频统一为 16 kHz 单声道；使用 `librosa.pyin`，固定 `fmin=60 Hz`、`fmax=2000 Hz`、`frame_length=1024`、`hop_length=160`。

固定的 20 维特征包括：

1. log-F0 的 q10、中位数、q90、IQR、标准差和 MAD；
2. log-F0 的时间斜率与相邻有效帧绝对变化中位数；
3. 周期变化代理量；
4. 有声帧比例、平均 voicing probability 及其标准差；
5. 基于归一化自相关的周期性中位数与 HNR 中位数；
6. 有声帧 RMS 的相邻变化代理、中位数和 IQR；
7. 有声帧谱倾斜的中位数和 IQR；
8. 有声帧谱平坦度中位数。

这里的周期和幅度变化是稳健代理量，不将其表述为严格的逐周期 Praat jitter/shimmer。缺失值只用每个训练角色中的中位数填补；均值和标准差也只由训练角色估计，避免验证信息泄漏。

## 3. 固定管线

三条管线共享相同的 768→128→3 AST 分类头、初始化、猫批次、call-level 类别平衡交叉熵、优化器、batch 顺序和基于验证猫级 CE 的 checkpoint 规则。

- `A0_ast_only`：仅使用冻结 AST call embedding。
- `A1_age_residual`：20 维年龄声学特征经 20→32→128 分支形成 residual，加到 AST 的 128 维隐藏表示；末层投影零初始化，使初始 logits 与 A0 一致。
- `A2_confidence_gated_age_residual`：与 A1 相同，但 residual 乘以 F0 可靠度与 128 维可学习 gate。可靠度固定为 `sqrt(voiced_fraction × mean_voicing_probability)`，在 F0 不可靠时抑制年龄分支。

本轮不同时加入 Hybrid Call+Cat loss，以便隔离年龄声学分支本身的作用。如果某条年龄管线达到晋级条件，再冻结其结构，在下一阶段单独检验 Hybrid 目标是否提供额外收益。

## 4. 内部评估

- 数据：MeowAgeNet analysis view，792 calls、111 cats；
- 表示：锁定的 768 维标准 AST call embedding；
- 角色：formal-v2 nested roles 的 train 和 validation；
- 独立单位：cat ID；
- 范围：3 repeats × 4 folds × 3 pipelines，共 36 fits；
- base seed：17；
- outer test：不读取、不预测、不评价。

主要比较为 A1−A0 和 A2−A0。完整报告同时保留每个 fold 的正负结果、最差 fold、senior recall、CE 和 Brier，不因 gate 未通过而删除平均正向信号。

## 5. 预设晋级条件

候选管线必须同时满足：

1. 平均 fold Macro-F1 相对 A0 至少提高 0.005；
2. 12 个 folds 中至少 8 个为正；
3. 合并验证集 CE 不劣于 A0；
4. 合并验证集 Brier 不劣于 A0；
5. 最差 fold 的 Macro-F1 差值不低于 -0.03；
6. 每个 repeat 的 senior recall 差值不低于 -0.02。

达到条件只表示可以进入下一轮内部 Hybrid 组合或外部冻结验证，并不直接构成正式效果声明。若平均结果为正但未过 gate，则保留为探索性方法模块和后续消融依据，不围绕当前 folds 事后搜索 F0 范围、特征子集、隐藏维度或 gate 初值。

## 6. 解释边界

F0 和声质同时受情境、性别、体型、品种、录音条件及发声类型影响，不能被视为年龄的唯一决定因素。本模块的目标是提供年龄相关的结构先验，并由 AST 保留情境和整体频谱信息；它不是用单一平均 F0 替代 AST，也不预设年龄与 F0 存在跨物种单调关系。
