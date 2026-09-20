# IDEA-082：MeowAgeNet 年龄声学机制分组消融

## 研究问题

IDEA-068 至 IDEA-076 已经表明：完整 20 维年龄声学分支在同一 MeowAgeNet 数据上的平均方向为正，但未通过稳定性与安全门槛。IDEA-082 不再重复“完整 C1 是否有效”，而是在不看新结果的前提下，检验正向信号主要来自哪一类声学机制，并用等参数 shuffled-feature 对照区分“真实声学信息”与“新增参数/优化路径”。

## 冻结分组

三个主分组互斥且穷尽 IDEA-068 的 20 维特征；组合组是预先声明的重叠：

- G1（F0 level/contour，5 维）：`log_f0_q10`、`log_f0_median`、`log_f0_q90`、`log_f0_iqr`、`log_f0_slope`。
- G2（source stability/periodicity/voicing/HNR，10 维）：`log_f0_std`、`log_f0_mad`、`log_f0_abs_delta_median`、`period_variation_proxy`、`voiced_fraction`、`voiced_probability_mean`、`voiced_probability_std`、`periodicity_median`、`hnr_db_median`、`amplitude_variation_proxy`。
- G3（spectral-energy/voice-quality，5 维）：`log_rms_median`、`log_rms_iqr`、`spectral_tilt_median`、`spectral_tilt_iqr`、`spectral_flatness_median`。
- G12（F0 + stability，15 维）：G1 与 G2 的并集。

`log_f0_std`、`log_f0_mad` 和 `log_f0_abs_delta_median` 同时会响应有意的音高变化和不稳定性；为避免 G1 预先混入待检验的稳定性机制，本实验将它们锁入 G2。未执行的 8/7/5 taxonomy 只作为边界敏感性说明，不构成可在结果后切换的方案。

## 模型与容量

融合算子冻结为 C1：

`h' = h + 0.25 * stopgrad(RMS(h)) * tanh(W2 GELU(W1 a))`

完整 C1 的年龄支路预算为 9,068 个参数。按输入维数透明地选择最接近该预算的整数宽度：G1=67（9,106）、G2=64（9,024）、G3=67（9,106）、G12=62（9,056）。最大偏差 44 个参数，占完整年龄支路预算 0.49%。同组 real 与 shuffled 的结构、参数量和初始状态完全相同。

九条管线为：A0；G1/G2/G3/G12 各自的 real 与 shuffled。完整 C1 不重跑，只作为 IDEA-076 的历史参考。

## Shuffled-feature 容量对照

每个 repeat/fold 内，train 与 validation 分别生成确定性、标签盲、无固定点的调用级置换；不跨 role，不触碰 test。所有组在同一 role/cell 复用同一行映射，因此 real/shuffled 的训练特征多重集以及训练折统计量严格一致。runner 记录目的行到源行映射的 SHA-256 与固定点数。

## 统计身份与拟合数

新 base seeds 为 8694、5378、5945；每个 seed 有 3 repeats × 4 folds。历史 A0 的 seed 身份无法与这些新 seed 严格配对，因此只重跑每个 cell 一次 A0，并由八条分组管线共同复用：36 个 A0 加 288 个分组拟合，共 324 fits。

主汇总单位是 9 个等权 seed×repeat 估计，每个估计先池化四个 validation folds；split-cell 汇总为 12 个 repeat×fold 单元，每个单元先对三个 base seed 取均值。调用级和重复动物出现只作描述，不能当作独立样本。

## 预注册判定

每个主组分别检验两个对照：信息对照 real−shuffled，以及实用对照 real−A0。两者都须满足：Macro-F1 均值至少 +0.005；至少 2/3 base-seed 均值为正；至少 6/9 seed×repeat 为正；至少 8/12 split cells 非负；最差 split cell 不低于 −0.03；平均 animal CE 与 Brier 不劣；平均 balanced accuracy 不劣。实用对照还要求每个 base seed 的 senior-recall 差值不低于 −0.02。只有信息门与实用门同时通过，才标记该来源得到支持。

G12 同样按上述门槛独立判断；`G12−G1`、`G12−G2` 与 `G12−G1−G2+A0` 只作预注册描述，不据此事后选择或重定义分组。

## 执行边界

- checkpoint 继续使用 animal-level validation CE；训练损失继续使用既有 globally class-balanced call CE。
- 所有缺失值填补与标准化只用当前 training cats 拟合，再原样用于 validation。
- 不生成、不读取 outer-test prediction 或指标。
- CPU preflight 通过后仍不得启动正式实验；正式运行需要显式 `--director-authorized`。
- 任何门槛失败均如实记录，不修改分组、宽度、seed、置换、聚合或阈值。

