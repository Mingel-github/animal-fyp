# IDEA-072：C1 有界年龄加法的独立种子确认

## 研究问题

IDEA-071 在三个新基础种子上发现：原始无界 A1 相对 A0 的平均 Macro-F1 为 `-0.018123`，而有界宽加法 C1 相对 A1 为 `+0.016909`，基本恢复到 A0（C1−A0 `-0.001214`）。但 C1 尚未形成绝对净增益，而且 C1 比 A1 更宽，不能区分其保护作用来自额外容量还是有界公式。

本轮完全冻结 C1 的结构与 `0.25` 上限，使用新的、结果产生前锁定的种子，并加入同参数量的无界宽加法 U1。目标是同时回答：

1. C1 能否稳定超过 A0；
2. C1 能否继续优于原 A1；
3. 在参数量和宽度完全匹配后，有界 RMS 相对残差能否优于无界残差。

## 四条配对管线

共同 AST 主路径：

```text
h = ReLU(W_ast x)
```

其中 `x` 是只按训练角色标准化的 768 维冻结 AST embedding，`a` 是锁定的 20 维年龄声学特征。

### A0：AST-only

```text
h_A0 = h
```

### A1：原始无界加法

```text
c = GELU(W1 a), W1: 20→32
r = W_r c, W_r: 32→128
h_A1 = h + r
```

### U1：容量匹配的无界宽加法

```text
c = GELU(W1 a), W1: 20→60
r = W_r c, W_r: 60→128
h_U1 = h + r
```

### C1：锁定的有界宽加法

```text
c = GELU(W1 a), W1: 20→60
rms = stopgrad(sqrt(mean_j(h_j^2) + 1e-8))
r = 0.25 * rms * tanh(W_r c)
h_C1 = h + r
```

U1 与 C1 使用相同宽度、相同参数量和相同零初始化输出头；二者只相差有界 RMS 相对融合公式。因此 C1−U1 检验有界化本身，而不是容量差异。

## 参数与公平性

| 管线 | 总可训练参数 | 年龄分支参数 |
|---|---:|---:|
| A0 | 99,075 | 0 |
| A1 | 103,971 | 4,896 |
| U1 | 108,143 | 9,068 |
| C1 | 108,143 | 9,068 |

所有年龄输出头零初始化，四条管线优化前 logits 必须完全一致。同一 `(seed, repeat, fold)` 共享 AST 主头初始化、训练 RNG 重置、数据角色、猫 batch 顺序、损失、优化器和 checkpoint 规则。

## 新种子与预算

固定字符串：

```text
IDEA-072-C1-independent-seed-confirmation-v1
```

其 SHA-256 为：

```text
7b38bf32fa4607fa54b7d5561a664296320a5d8733105a7a6cecc2fa4c9ea142
```

按 digest 中连续的大端 uint32 对 10000 取模、跳过已选冲突值的规则，前三个值无冲突：

```text
base seeds = [6530, 3562, 3846]
```

字符串按 UTF-8 编码。这些种子未用于 IDEA-068 至 IDEA-071。固定使用 3 repeats × 4 folds，完整种子为 `base_seed + 10000×repeat + 100×fold`。这里的“独立种子确认”仅指新初始化；仍复用相同 111 只猫和内部角色，不是独立数据集确认。

总预算：`4 pipelines × 3 seeds × 3 repeats × 4 folds = 144 fits`。

## 预设 gate

### 基础有效性：C1 对 A0

1. 9 个 seed×repeat 的平均 C1−A0 Macro-F1 至少 `+0.005`；
2. 3/3 基础种子的平均 C1−A0 严格大于 0；
3. 至少 6/9 个 seed×repeat 的 C1−A0 严格大于 0；
4. 至少 8/12 个共同 split-cell 的跨种子平均 C1−A0 不小于 0；
5. 最差共同 split-cell 不低于 `-0.03`；
6. 9 个 seed×repeat 单元等权平均的 CE 与 Brier 均不劣于 A0；
7. 每个基础种子的 pooled senior recall 增益不低于 `-0.02`。

### A1 替代性

8. 平均 seed×repeat C1−A1 不低于非劣界 `-0.005`；
9. 至少 2/3 基础种子的平均 C1−A1 不小于 0；
10. C1 的 seed×repeat 平均 CE 与 Brier 均不劣于 A1；
11. 每个基础种子的 pooled senior recall 相对 A1 不低于 `-0.02`。

若平均 C1−A1 严格大于 0，另报告为 F1 优越；替代 gate 本身只要求非劣。

### 有界机制：C1 对 U1

12. 平均 seed×repeat C1−U1 至少 `+0.005`；
13. 至少 2/3 基础种子的平均 C1−U1 不小于 0；
14. 至少 5/9 个 seed×repeat 的 C1−U1 严格大于 0；
15. C1 的 seed×repeat 平均 CE 与 Brier 均不劣于 U1。

只有三层全部通过，才能称 C1 在新种子上晋级且其优势不能由宽度／容量单独解释。若只通过 A1 替代层，只能称 C1 对无界 A1 有保护作用；若基础有效性通过但机制层失败，只能称 C1 整包模块有效，不能归因于有界公式。

## 审计与停止边界

- 只使用 inner train/validation roles，不生成 outer-test 预测；
- 锁定 roles、AST embedding、AST fbank、年龄特征和提取摘要哈希；
- 核对 792 calls、111 cats、36 个互不重复的完整种子；
- 36 组四管线优化前 logits 最大差必须为 0；
- 共同 epoch 的猫顺序与 call coverage 哈希必须一致；
- Macro-F1、CE、Brier 的 gate 均以 9 个 seed×repeat 单元等权汇总；12 个 split-cell 先跨三个基础种子平均；36 folds 和 612 个重复 animal occurrences 只作描述；
- 记录 C1 最终 checkpoint 的验证集 `||r||/||h||`；
- 无论通过与否都保留所有结果，不在本结果上调整宽度、cap、RMS 规则、种子或 gate。
