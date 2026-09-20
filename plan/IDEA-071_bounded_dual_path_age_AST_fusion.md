# IDEA-071：有界双路径年龄—AST 融合

## 研究问题

IDEA-068/069/070 表明两件事同时成立：

1. 原始 A1 加法残差能够稳定提供较强的 Macro-F1 增益，说明年龄声学分支需要向 AST 补充新的表示方向；
2. BACM 的有界通道缩放取得三者最佳 pooled CE 与 Brier，说明限制年龄分支扰动有助于概率校准，但纯缩放过于保守，不能替代 A1。

本轮检验一个模块级假设：年龄分支同时承担“重标定 AST 已有证据”和“补充 AST 缺失方向”，并对两种作用分别设定结构上限，能否兼顾 A1 的分类增益与 BACM 的校准收益。

## 四条配对管线

共同 AST 主路径为：

```text
h = ReLU(W_ast x)
```

其中 `x` 是训练角色标准化后的 768 维冻结 AST embedding。

### A0：AST-only

```text
h_A0 = h
```

### A1：原始加法残差

```text
c = GELU(W1 a), W1: 20→32
r = W_r c, W_r: 32→128
h_A1 = h + r
```

年龄分支增加 4,896 个参数。

### C1：容量与约束匹配的宽加法对照

```text
c_wide = GELU(W_wide a), W_wide: 20→60
rms = stopgrad(sqrt(mean_j(h_j^2) + 1e-8))
r_wide = 0.25 * rms * tanh(W_r_wide c_wide), W_r_wide: 60→128
h_C1 = h + r_wide
```

年龄分支增加 9,068 个参数。C1 同时匹配 D1 的容量、有界 `tanh` 和相对 AST 隐状态 RMS 的 shift 尺度；因此 D1−C1 主要回答“多出的通道缩放路径是否有价值”，而不是把容量或正则化混进机制差异。

### D1：有界双路径仿射融合

```text
c = GELU(W1 a), W1: 20→32
s = 0.25 * tanh(W_s c), W_s: 32→128
rms = stopgrad(sqrt(mean_j(h_j^2) + 1e-8))
b = 0.25 * rms * tanh(W_b c), W_b: 32→128
h_D1 = h * (1 + s) + b
```

D1 年龄分支增加 9,120 个参数，只比 C1 多 52 个参数。`s` 把每个 AST 通道缩放到原值的 `[0.75, 1.25]`；`b` 能写入新方向，但每维幅度不超过当前样本 AST 隐状态 RMS 的 25%。`rms` 停止梯度，避免模型通过主动放大主路径来松动 shift 约束。两个输出头都零初始化，训练开始时 D1 与 A0 完全一致。

本轮固定 scale cap 与 shift cap 均为 `0.25`，不做幅度搜索。两者使用独立输出头，因为缩放已有通道和补充新方向是不同机制；共享同一个 32 维年龄编码器，避免不受控地扩大前端容量。

## 参数量与公平性

| 管线 | 总可训练参数 | 年龄分支参数 |
|---|---:|---:|
| A0 | 99,075 | 0 |
| A1 | 103,971 | 4,896 |
| C1 | 108,143 | 9,068 |
| D1 | 108,195 | 9,120 |

C1 与 D1 的总参数差只有 52。所有残差／调制输出层均为零初始化；同一 `(seed, repeat, fold)` 的四条管线共享 AST 主头初始化、训练 RNG 重置、数据角色、猫 batch 顺序、损失、优化器和 checkpoint 规则。另记录验证集 `||年龄扰动|| / ||h||`，检查有界公式在实际训练后的作用强度。

## 新种子与预算

基础种子由固定字符串 `IDEA-071-bounded-dual-path-affine-v1` 的 SHA-256 digest 前三个大端 uint32 对 10000 取模得到：

```text
SHA-256 = c6787a5c673b16c7bb778880d35306ccab80426918c1a83426b49118e9b117ad
base seeds = [4412, 5703, 3120]
```

它们在 IDEA-071 结果产生前锁定，未用于 A1、BACM 或本候选。固定使用 3 repeats × 4 folds；完整种子为 `base_seed + 10000×repeat + 100×fold`。

总预算：`4 pipelines × 3 seeds × 3 repeats × 4 folds = 144 fits`。

## 预设 gate

Gate 分为“基础有效性”和“机制／替代性”两层，D1 必须同时满足：

1. D1−A0 的 9 个 seed×repeat 平均 Macro-F1 至少 `+0.005`；
2. 3/3 基础种子的 D1−A0 平均值严格大于 0；
3. 至少 6/9 个 seed×repeat 的 D1−A0 严格为正；
4. 至少 8/12 个共同 split-cell 的跨种子平均 D1−A0 不小于 0；
5. 最差共同 split-cell 的平均 D1−A0 不低于 `-0.03`；
6. pooled CE 与 Brier 均不劣于 A0；
7. 每个基础种子的 pooled senior recall 增益不低于 `-0.02`；
8. D1−A1 的平均 seed×repeat Macro-F1 不低于 `-0.005`；
9. D1 pooled CE 不劣于 A1；
10. D1−C1 的平均 seed×repeat Macro-F1 严格大于 0；
11. 至少 2/3 基础种子的 D1−C1 平均值不小于 0；
12. D1 pooled CE 不劣于 C1。

前七项是 D1 对 A0 的基础有效性；第八、九项检验能否替代 A1；最后三项在容量、约束匹配后检验 scale+shift 是否优于单独 shift。只有全部通过才能称为“双路径机制晋级”；若只通过基础层，只能称 D1 整体模块有效。

## 审计与停止边界

- 只使用 inner train/validation roles，禁止生成 outer-test 预测；
- 锁定 roles、AST embedding、AST fbank、年龄声学特征和提取摘要哈希；
- 核对 792 calls、111 cats、36 个互不重复的完整种子；
- 36 组四管线优化前 logits 最大差必须为 0；
- 共同 epoch 的 cat order 与 call coverage 哈希必须一致；
- 断点续跑核验 fit 身份、预测文件和 SHA-256；
- 无论通过与否都保留所有正、平、负结果，不在本结果上调整两个 cap、宽度、种子或 gate。
