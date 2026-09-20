# IDEA-070：有界年龄条件 AST 调制（BACM）

## 研究动机

IDEA-068/069 的 A1 年龄声学残差在四个基础种子上都形成正向平均信号，但独立种子复现仍有 5/12 个共同 split 为负，最差 split 为 `-0.0386`，pooled cross-entropy 也变差。A1 的结构为无条件加法：年龄分支可以向 128 维 AST 隐状态写入任意方向和幅度的向量。即使某条叫声的 F0／周期性线索受到情境、性别、体型或录音条件干扰，这个残差仍会直接改变分类表示。

本实验改变“如何融合”，不重新选择特征、不调整 F0 范围，也不改变训练数据。目标是检验：把年龄分支限制为对 AST 已有通道的小幅条件缩放，能否保留平均收益，同时降低 split 敏感性和概率劣化。

## 三条严格配对管线

设标准化 AST embedding 为 `x`，锁定的 20 维年龄声学特征为 `a`：

```text
h = ReLU(W_ast x)
c = GELU(W1 a)
```

- `A0_ast_only`：`h` 直接进入共享的 BatchNorm、dropout 和分类层。
- `A1_age_residual`：`h + W2 c`，即 IDEA-068/069 的原始加法残差。
- `B1_bounded_age_modulation`：

```text
s = 0.25 * tanh(W2 c)
h_B1 = h * (1 + s)
```

`W1: 20→32`、`W2: 32→128`，因此 B1 与 A1 的年龄分支参数量完全相同。B1 的 `W2` 权重和偏置均为零初始化，训练开始前 `s=0`，A0、A1、B1 logits 必须逐元素完全一致。

B1 每个通道的缩放范围固定为 `[0.75, 1.25]`。它可以按年龄声学条件增强或抑制 AST 已经存在的证据，但不能翻转 ReLU 后通道的符号，也不能凭空写入任意新的隐藏向量。`0.25` 是运行前锁定的结构预算，本轮不搜索其他幅度。

## 未使用种子与预算

基础种子由固定字符串 `IDEA-070-bounded-age-conditioned-film-v1` 的 SHA-256 digest 前三个大端 uint32 对 10000 取模得到：

```text
SHA-256 = 13179406c930b92e12db568abb716a965b9e8c15f16e1d233a6fc8eb9e555236
base seeds = [2326, 3550, 4426]
```

这些种子未用于 A1，且在 IDEA-070 结果产生前锁定。每个基础种子继续使用固定的 3 repeats × 4 folds；完整种子为 `base_seed + 10000×repeat + 100×fold`。

总预算为 `3 pipelines × 3 seeds × 3 repeats × 4 folds = 108 fits`。每个 `(seed, repeat, fold)` 内三条管线共享数据角色、AST 主头初始化、训练 RNG 重置、猫 batch 顺序、损失、优化器和 checkpoint 规则。

## 预设判定

B1 相对 A0 的所有条件必须同时满足：

1. 9 个 seed×repeat 的平均 Macro-F1 增益至少 `+0.005`；
2. 3/3 基础种子的平均增益严格大于 0；
3. 至少 6/9 个 seed×repeat 增益严格大于 0；
4. 至少 8/12 个共同 split-cell 的跨种子平均增益不小于 0；
5. 最差共同 split-cell 的平均增益不低于 `-0.03`；
6. pooled cross-entropy 与 Brier 均不劣于 A0；
7. 每个基础种子的 pooled senior recall 增益不低于 `-0.02`。

为避免“比基线略好、却比原 A1 明显差”仍被误称为改进，另要求：

8. B1−A1 的平均 seed×repeat Macro-F1 不低于 `-0.005`；
9. B1 pooled cross-entropy 不劣于 A1。

全部条件通过才称 B1 为 A1 的稳定化改进。否则完整保留正负结果，不在同一结果上调整缩放上限、特征、宽度、种子或门槛。

## 审计边界

- 只使用 formal roles 中的 train/validation，禁止生成 outer-test 预测；
- 锁定 roles、AST embedding、AST fbank、IDEA-068 年龄特征与提取摘要的 SHA-256；
- 运行时核对 792 calls、111 cats 和 36 个互不重复的完整种子；
- 每组三条管线记录优化前 logits 最大差，必须为 0；
- 共同训练 epoch 的 cat order 与 call coverage 哈希必须一致；
- 断点续跑重新核验 fit 身份、预测文件及 SHA-256；
- 36 个逐 fold 差值和 pooled animal occurrences 均为重复观察，只作描述，不当作独立样本。

## 后续边界

本轮只改变年龄—AST 融合算子。既有多层 naive fusion、普通 Adapter、LoRA 和无条件 top-block 更新已有不稳定或负向历史结果，不在本轮重复。若 B1 通过，再另行冻结“年龄条件选择 AST 深度／进入 AST 上层 block”的内部结构实验。

