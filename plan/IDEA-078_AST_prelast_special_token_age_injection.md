# IDEA-078：在 AST 最后编码块前向双 special tokens 注入年龄声学条件

## 状态与优先级

- 状态：结果产生前的冻结计划。
- 数据优先级：MeowAgeNet 为最高优先级；其他猫数据次之；犬数据最低。IDEA-075 的犬结果不进入本轮 gate。
- IDEA-076 已冻结：不增加其种子、不修改 C1、不用本轮结果回写 IDEA-076。
- IDEA-077 已完成且固定 last-4 全局 LayerMix 失败；本轮不是事后挑选层权重，而是新的、预先固定位置的条件注入机制。
- 本轮只使用 formal-v2 nested train/validation 角色，`outer_test_accessed=false`。

## 研究问题

现有 A1/C1 把 20 维年龄敏感声学特征加到 AST 已完成全局池化之后的分类器隐藏层。它们能改变分类边界，却不能让冻结 AST 的最后一个 self-attention block 在年龄条件下重新读取 patch tokens。

IDEA-078 检验一个更窄的问题：只在最后一个 AST encoder block 前，把由 20 维锁定年龄特征产生的同一个条件向量加到 CLS 与 distillation 两个 special tokens，保持 144 个 patch tokens 不变，再让冻结的最后 block、最终 LayerNorm 与原 pooler 处理该变化，能否比同路径 A0 稳定提高 MeowAgeNet 动物级 Macro-F1。

## 为什么必须创建新的无标签 token cache

现有逐层缓存形状为 `792 calls × 12 layers × 768`。它在每层已经执行：最终 LayerNorm、CLS/distillation 均值、call 内 segment 均值。因此它已经丢失：

1. CLS 与 distillation 两个 token 的分别状态；
2. 144 个 patch tokens；
3. 843 个 segment 到 792 个 call 的未平均表示。

最后一个 Transformer block 的 self-attention 需要完整的 `146 × 768` token 序列，所以现有 cache 不能支持本轮。必须新增 label-free cache：保存每个 segment 在第 11 个 block 输出、进入第 12 个 block之前的完整 float32 tokens。

锁定 cache 定义：

- 输入：现有 `ast_fbank_128.npz` 中 843 个标准 AST segments；
- 模型：锁定 checkpoint 与 revision，标准 10×10 stride，12×12 patch grid；
- 缓存位置：one-based block 11 输出／one-based block 12 输入；
- 形状：`843 × 146 × 768`，其中位置 0/1 是 CLS/distillation，位置 2–145 是 144 个 patch tokens；
- dtype：float32；
- token 文件：未压缩 `.npy`，允许 memory map；
- index：只保存 `segment_call_indices`、`segment_counts`、`call_ids` 与几何；不保存 labels、cat_id 或任何 split/role；
- label use：false；feature extraction 不读取年龄标签或角色。

token 数据本体为 94,523,904 个 float32，即 378,095,616 bytes（360.58 MiB；另加很小的 `.npy` header）。单纯使用 float16 虽可减半，但会让 A0 精确重建与锁定 pooler 的误差解释复杂化，因此本轮固定 float32。

## 固定表示路径

令 `S11 ∈ R^(146×768)` 为一个 segment 在冻结 AST block 11 后的 token 序列，`B12` 为冻结的最后一个 AST block，`LN` 为冻结的最终 LayerNorm。锁定的 segment pooler 为：

```text
Pool(S) = 0.5 × [LN(B12(S))_CLS + LN(B12(S))_DIST]
```

call 表示是该 call 全部 segments 的 `Pool(S)` 算术平均。A0 不直接读取旧 final embedding 作为模型输入；它必须从同一 pre-last cache 经 `B12 → LN → 双 special mean → segment mean` 重建。旧 final embedding 只作为只读锚点，preflight 必须验证重建结果与锁定 `pooler_output` 匹配。

## 年龄条件分支

20 维年龄特征继续使用 IDEA-068 锁定的 label-free 特征。每个 fit 只用当前训练角色计算逐维中位数、均值和标准差：先以训练中位数填补非有限值，再标准化。

瓶颈固定为 10：

```text
ã = train-role standardize_and_impute(a)
u = GELU(W1 ã + b1),              W1: 20 → 10
d = W2 u + b2,                    W2: 10 → 768
```

`W2` 与 `b2` 全零初始化，所以所有候选在优化前对 A0 的影响严格为零。`W1/b1` 使用相同完整种子的默认 PyTorch 初始化；构建后重置训练 RNG，避免不同构造路径改变 dropout 与 batch 顺序。

年龄分支参数：

```text
(20×10 + 10) + (10×768 + 768) = 8,658
```

这比旧 A1 的 4,896 个年龄分支参数多 3,762，比 C1 的 9,068 个少 410。它最接近 C1，但没有结果后填充无作用参数来伪造精确相等。

## 三条预注册管线

### A0：同路径 token 重建基线

```text
z_A0 = mean_segments Pool(S11)
```

之后使用原训练角色标准化的 `768 → 128 → 3` 分类头。

### P1：参数匹配的 post-pool 直接注入对照

```text
z_P1 = mean_segments Pool(S11) + d(a_call)
```

P1 与 T1 使用完全相同的 20→10→768 年龄分支和参数量。它控制“多 8,658 个参数、可以直接利用年龄特征”本身，避免把普通 post-pool 残差误写成内部 token 机制。

### T1：pre-last 双 special-token 注入

```text
S'_CLS  = S11_CLS  + d(a_call)
S'_DIST = S11_DIST + d(a_call)
S'_PATCH = S11_PATCH                 # 144 个 patch tokens 完全不变
z_T1 = mean_segments Pool(S')
```

最后一个 AST block、最终 LayerNorm 与所有 AST 参数保持冻结；只训练年龄分支与原分类头。

参数量固定为：

| 管线 | 分类头 | 年龄分支 | 总可训练参数 |
|---|---:|---:|---:|
| A0 | 99,075 | 0 | 99,075 |
| P1 | 99,075 | 8,658 | 107,733 |
| T1 | 99,075 | 8,658 | 107,733 |

因此 P1 是必要的容量与直接注入机制对照；不再增加其他管线。

## 与旧 A1/C1 的关系

- A1：20→32→128，无界加到**可训练分类器的 128 维隐藏层**；参数 103,971。
- C1：20→60→128，以 `0.25×stopgrad(RMS(h))×tanh` 有界加到**128 维分类器隐藏层**；参数 108,143。
- P1：20→10→768，无界加到**冻结 AST 最终 pooled representation**，随后才进入分类头；参数 107,733。
- T1：20→10→768 加到**最后 AST block 前的两个 special tokens**，允许冻结 self-attention/MLP 用未改动 patch tokens 重新变换条件；参数 107,733。

所以 T1 不与 A1/C1 重复。P1 是与 T1 同参数、同输出维度的直接注入对照；只有 T1 相对 P1 通过机制 gate，才能把收益归因于“最后 block 内的 token 交互”，而不是年龄分支容量。

## 种子与预算

固定 UTF-8 字符串：

```text
IDEA-078-Meow-AST-prelast-special-token-age-injection-v1
```

SHA-256：

```text
3a68ffd3fb356fbe000c0cde18fa1439d108b439b0927a1048e519c96052e67b
```

按 successive non-overlapping big-endian uint32、模 10000、跳过 IDEA-068 至 IDEA-077 base/full seed 冲突的规则，前三个候选均合格：

```text
9763, 3230, 9726
```

完整种子规则继续为 `base_seed + 10000×repeat + 100×fold`。3 个基础种子 × 3 repeats × 4 folds 产生 36 个唯一完整种子，和 IDEA-068 至 IDEA-077 的 base/full seeds 均无碰撞。

正式预算：

```text
3 pipelines × 3 base seeds × 3 repeats × 4 folds = 108 fits
```

其中主比较 A0/T1 共 72 fits。

## 训练与配对规则

- 角色：`meowagenet_formal_v2_nested_roles.csv`；每个 cell 只使用 train 与 validation。
- loss：训练 calls 上全局类别平衡 call cross-entropy。
- checkpoint：最低未加权 inner-validation animal cross-entropy。
- optimizer：Adamax，learning rate `0.006`，epsilon `1e-7`，gradient clip `1.0`。
- dropout：`0.44571035356880917`；cat batch size 4；最多 50 epochs；patience 8。
- 同一完整种子的 A0/P1/T1 共享分类头初始状态、角色、cat batch 顺序、call/segment coverage 和训练 RNG 重置。
- P1/T1 的年龄分支完整初始状态相同。
- 三管线优化前 logits 和 loss 必须严格相同。
- 冻结 AST tail 永远保持 eval mode，参数梯度为 false；T1 梯度只穿过其计算图到达年龄分支，不更新 AST。
- outer-test 预测、选择或调参一律禁止。

## preflight 与分阶段授权

### 阶段一：CPU 可行性 preflight

在 cache 尚未创建时，使用锁定 fbank 的少量完整 calls 在 CPU 上执行：

1. 标准 AST 完整 forward 得到 canonical pooler；
2. 手工执行 embeddings + blocks 1–11，得到真实 `S11`；
3. 手工执行 block 12 + final LayerNorm + 双 special mean；
4. 逐元素比较手工路径与同一次 canonical pooler；
5. 将 segments 平均为 call 后，在容差内比较锁定 GPU final embedding；
6. 检查 A0/P1/T1 零影响初始化、P1/T1 完整年龄分支状态相同、T1 只改 token 0/1、梯度可达、参数量和内存预算。

该阶段只可给出 `GO_FOR_LABEL_FREE_CACHE_EXTRACTION`，不能授权正式训练。

### 阶段二：cache 后 CPU formal preflight

新 cache 完成并记录来源、模型和输出哈希后，必须在 CPU 上对全部 843 segments 重建 A0，并与 792 个锁定 final embeddings 比较。只有 cache 几何、ID/mapping、全量 A0 重建、角色、哈希、初始化、公平性和预算全部通过，才可给出 `GO_FOR_FORMAL_GPU_RUN`。

## 结果汇总与 gate

主统计单位为 9 个等权 `base_seed × repeat` 单元；每个单元先合并 4 个 validation folds。12 个共同 split-cell 先在三个基础种子上平均。36 个 fold 对比与重复 animal occurrences 只作描述。

### 主 gate：T1 相对 A0

所有条件必须同时满足：

1. 平均 T1−A0 Macro-F1 ≥ `+0.005`；
2. `3/3` 基础种子均值严格为正；
3. 至少 `6/9` seed×repeat 严格为正；
4. 至少 `8/12` split-cell 非负；
5. 最差 split-cell ≥ `-0.03`；
6. T1 平均 CE 不劣于 A0；
7. T1 平均 Brier 不劣于 A0；
8. 每个基础种子的 pooled senior-recall 差值 ≥ `-0.02`。

同时完整报告 balanced accuracy，但不在看结果后把它加入或移出 gate。

### 机制 gate：T1 相对 P1

只有主 gate 通过时才可解释；所有条件必须同时满足：

1. 平均 T1−P1 Macro-F1 严格为正；
2. 至少 `2/3` 基础种子均值严格为正；
3. 至少 `5/9` seed×repeat 严格为正；
4. 至少 `8/12` split-cell 非负；
5. 最差 split-cell ≥ `-0.03`；
6. T1 平均 CE 不劣于 P1；
7. T1 平均 Brier 不劣于 P1。

P1−A0 的 Macro-F1、balanced accuracy、CE、Brier、senior recall 与 split 稳定性全部报告，但不单独授权 T1。

## 决策规则

- 主 gate 通过、机制 gate 通过：保留 T1，并允许解释最后 block 内 token 交互提供了超出同参数直接注入的增益。
- 主 gate 通过、机制 gate 未通过：可保留 T1 作为有效年龄条件模型，但不能把增益归因于内部 token 机制。
- 主 gate 未通过：记录所有正向与负向结果，停止该固定注入层、token 范围、瓶颈和公式；不得事后改注入层、只选 CLS、加入 patch tokens、改瓶颈、种子或 gate。

## 资源边界

- cache 磁盘：约 360.58 MiB token 数据，另加 index/summary。
- 最坏 4-cat 静态上界：当前数据中 segment 数最多的四只猫合计 157 segments；其 token 输入 float32 约 67.2 MiB。
- batch 32/64 的 token 输入分别约 13.69/27.38 MiB；最后 block 的 attention/activation 是主要显存项。
- 8GB RTX 4060 Ti 可行；正式 runner 固定 AMP，若实测 OOM 只能减小 `cat_batch_size` 并作为新的协议版本处理，不能在本协议结果中静默改变。

## 与 IDEA-077 的边界

IDEA-077 的固定／全局可学习 LayerMix 已关闭。IDEA-078 不混合历史层，不依据 IDEA-077 权重选择 block；注入位置固定为最后 block 前，范围固定为两个 special tokens。IDEA-077 只提供“A0 final 仍是表示对照”和现有 call-level cache 不足的实现背景。
