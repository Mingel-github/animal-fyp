# IDEA-074：C1 × Hybrid Call+Cat 二乘二归因实验

## 研究问题

IDEA-071 至 IDEA-073 的跨轮描述性证据支持把 C1 有界宽年龄残差设为当前主要候选。IDEA-066 的固定 Hybrid Call+Cat 目标也取得正向平均 Macro-F1、CE 和 Brier，但没有达到 split 稳定性门槛。两者分别改变表示融合和监督层级，理论上具有正交性。

本轮不直接把两个模块堆在一起后只与 A0 比较，而采用完整二乘二设计，回答：

1. C1 在全新种子上是否继续超过匹配 A0；
2. 固定 Hybrid 目标是否为 C1 提供稳定的额外收益；
3. Hybrid 对 C1 的增量是否大于它对 A0 的增量，即是否存在超加性协同，而非单纯复现 Hybrid 主效应。

本轮仍使用同一 111 只猫与 inner train/validation roles，不是独立数据集确认。禁止生成或访问 outer-test 预测。

## 二乘二管线

共同 AST 主路径：

```text
h = ReLU(W_ast x)
```

其中 `x` 为只使用训练角色统计量标准化的768维冻结 AST embedding；`a` 为已锁定的20维 F0、周期性和声质特征。

### 表示轴

#### A0：AST-only

```text
h_A0 = h
```

#### C1：有界宽年龄残差

```text
c = GELU(W_age a), W_age: 20→60
q = stopgrad(sqrt(mean_j(h_j²) + 1e-8))
r = 0.25 × q × tanh(W_r c), W_r: 60→128
h_C1 = h + r
```

C1 完全复用 IDEA-071～073 的公式、宽度、`0.25` cap、训练折内缺失值填补与标准化规则，不在本轮调整。

### 监督轴

令每条叫声的类别平衡交叉熵为 `L_call`。对同一只猫的所有 call softmax 概率作算术平均，使用训练角色猫类别频率的逆频率权重计算猫级交叉熵 `L_cat`。每只猫的全部叫声必须在同一个 set sample 中，不能跨 batch 拆分；`cat_batch_size=4`。call 权重只由训练 calls 计算，cat 权重只由训练 cats 计算，两项损失均在每个 batch 内按“加权和／权重和”归一。猫级损失严格使用 mean(call softmax) 的真实类别概率，并在 `1e-7` 处 clamp 后取负对数。

```text
Call-only: L_C = L_call
Hybrid:    L_H = 0.5 × L_call + 0.5 × L_cat
```

`0.5/0.5` 严格复用 IDEA-066，不搜索混合系数。

四条管线都计算 `L_call` 与 `L_cat` 并记录审计值，只由固定系数组合为反向传播的 total loss，避免 call-only 与 Hybrid 走不同的数据或计算路径。

四条管线为：

| 管线 | 表示 | 训练目标 |
|---|---|---|
| `A0_C_call_only` | A0 | `L_call` |
| `A0_H_hybrid` | A0 | `0.5 L_call + 0.5 L_cat` |
| `C1_C_call_only` | C1 | `L_call` |
| `C1_H_hybrid` | C1 | `0.5 L_call + 0.5 L_cat` |

## 参数与初始化公平性

| 管线 | 可训练参数 |
|---|---:|
| A0-C | 99,075 |
| A0-H | 99,075 |
| C1-C | 108,143 |
| C1-H | 108,143 |

- A0-C 与 A0-H 在同一完整种子下初始 `state_dict` 必须逐张量相同；
- C1-C 与 C1-H 的初始 `state_dict` 必须逐张量相同；
- 四条管线共享的 AST 主干张量必须相同，C1-C 与 C1-H 的全部年龄分支张量必须相同；
- C1 输出头零初始化，因此四条管线的优化前 logits 必须完全相同；
- 模型构造后重置训练 RNG，使四条管线共享猫 batch 顺序、dropout/AMP 随机流和 call coverage；
- 两种 loss 不增加模型参数；差异只来自监督目标。

## 新种子与预算

固定 UTF-8 字符串：

```text
IDEA-074-C1-hybrid-factorial-v1
```

SHA-256：

```text
91d52a6aa12c7c8b7e37da71d0ece239c97564e6f3ae912ce92d60a9a07ab94a
```

按连续大端 uint32 对10000取模并跳过 IDEA-065 至 IDEA-073 所有相关实验已使用基础种子的预设规则，前三个值均无冲突：

```text
base seeds = [6346, 7243, 9617]
```

每个基础种子使用3 repeats × 4 folds，完整种子为 `base_seed + 10000×repeat + 100×fold`。

总预算：

```text
4 pipelines × 3 base seeds × 3 repeats × 4 folds = 144 fits
```

## 汇总单位

- Macro-F1、CE 和 Brier 的主要单位为9个 seed×repeat；每个单元先合并四折，再等权平均；
- 12个共同 split-cell 由相同 repeat×fold 在三个基础种子上先平均；
- 36个逐 fold 对比和每条管线612个重复 animal occurrences只作描述，不视为独立样本；
- 二乘二交互项在每个 seed×repeat 内计算：

```text
I = (C1-H − C1-C) − (A0-H − A0-C)
```

## 预设 gate

### 第一层：C1 主候选复现

比较 `C1-C − A0-C`：

1. seed×repeat 平均 Macro-F1 差值至少 `+0.005`；
2. 3/3 基础种子平均差值严格大于0；
3. 至少6/9个 seed×repeat 严格为正；
4. 至少8/12个共同 split-cell 不小于0；
5. 最差共同 split-cell 不低于 `-0.03`；
6. C1-C 的等权平均 CE 不劣于 A0-C；
7. C1-C 的等权平均 Brier 不劣于 A0-C；
8. 每个基础种子的 pooled senior recall 差值不低于 `-0.02`。

本层通过时，可称 C1 在本轮新种子上完成内部复现；失败时不删除历史正向结果，但不能把本轮写成确认成功。

### 第二层：Hybrid 是否应并入 C1

比较 `C1-H − C1-C`：

1. seed×repeat 平均 Macro-F1 差值至少 `+0.005`；
2. 3/3 基础种子平均差值严格大于0；
3. 至少6/9个 seed×repeat 严格为正；
4. 至少8/12个共同 split-cell 不小于0；
5. 最差共同 split-cell 不低于 `-0.03`；
6. C1-H 的等权平均 CE 不劣于 C1-C；
7. C1-H 的等权平均 Brier 不劣于 C1-C；
8. 每个基础种子的 pooled senior recall 差值不低于 `-0.02`；
9. C1-H 相对 A0-H 的 seed×repeat 平均 Macro-F1 至少 `+0.005`；
10. C1-H−A0-H 在3/3基础种子上均严格为正；
11. C1-H 的等权平均 CE 与 Brier 均不劣于 A0-H；
12. 每个基础种子的 C1-H−A0-H pooled senior recall 不低于 `-0.02`。

只有本层全部通过，才把 Hybrid 合入主要候选。否则继续保留纯 C1-C，Hybrid 仍作为独立正向消融。`C1-H−A0-C` 仍完整报告，但不作为独立 gate，因为在第一层和本层均通过时，其均值与基础种子方向已大体由简单效应蕴含。

### 第三层：超加性协同解释

对交互项 `I`：

1. seed×repeat 平均严格大于0；
2. 至少2/3基础种子的平均交互项不小于0；
3. 至少5/9个 seed×repeat 交互项不小于0；
4. 至少8/12个共同 split-cell 交互项不小于0；
5. 最差共同 split-cell 交互项不低于 `-0.03`。

第三层只控制“Hybrid 与 C1 存在超加性协同”的机制表述，不单独决定 C1-H 是否替代 C1-C。若交互约为0，表示两项收益近似可加兼容，不表示两个实验轴“不正交”。若第二层通过而第三层失败，可称 Hybrid 对 C1 有直接增益，但不能称两机制协同；若第三层通过而第二层失败，也不能升级组合，因为交互没有转化为足够的绝对收益。

必须完整报告四个简单效应：`C1-C−A0-C`、`A0-H−A0-C`、`C1-H−C1-C`、`C1-H−A0-H`。CE、Brier 和 senior recall 的交互项只作描述；其中 CE/Brier 使用“改善量差”的符号，使正值表示 Hybrid 在 C1 上带来的概率质量改善大于在 A0 上的改善。

## 审计与停止边界

- 核对792 calls、111 cats、36个互不重复的完整种子；
- 核对四管线优化前 logits 差为0，以及两组 loss 配对模型的初始 `state_dict` 相同；
- 所有共同 epoch 的猫顺序与 call coverage 哈希必须一致；
- checkpoint 重载概率差必须为0；
- 每个 fit 记录 call loss、cat loss、total loss，但 checkpoint 仍统一按最低验证动物 CE 选择；
- 核对所有预测只覆盖对应 inner-validation 角色；
- 无论结果如何，不在本轮调整 Hybrid 权重、C1 cap、年龄宽度、特征、种子、聚合或 gate；
- 不生成 outer-test 预测。
