# IDEA-039｜Animal-grouped 训练增强结果

> 执行日期：2026-09-16
> 状态：Stage C 首轮完成；当前有限增强池阶段性收尾
> 锁定代码提交：`fb9e09efb517f9432f44ec9a724e434bf026212c`

## 1. 本轮实验回答什么

本轮保持 MIT AST AudioSet checkpoint、冻结 backbone、`768 → 128 → 3` 分类头、111-cat
分组 splits、animal-level checkpoint selection 和 animal-level complete-OOF 评价不变，只改变训练
输入。核心问题分成两条：

1. H039-A：固定的轻量 SpecAugment 能否稳定提高 frozen AST；
2. H039-B：每个 outer fold 在训练侧独立选择增强策略，能否比固定增强更有效。

所有增强视图继承原 call 的 `cat_id`，同一只猫的 calls、segments 和增强视图始终位于同一数据
角色。Validation 与 test 使用 identity 输入，每个训练 epoch 为每条 call 生成一个增强视图。

## 2. 数据、候选与执行规模

- 数据：111 只猫、792 条 call、843 个 segment；
- 类别：Kitten 15 只、Adult 62 只、Senior 34 只；
- 评价单位：animal；每条 pipeline 在每个 repeat 形成覆盖 111 只猫的 complete OOF；
- 内层选择：4 个 policy × 3 条 augmentation RNG streams × 3 repeats × 4 folds，合计
  144 个 inner-only fit；
- 外层评价：4 条 pipeline × 3 repeats × 4 folds，合计 48 个 outer fit；
- 最终评价：4 条 pipeline × 3 组 complete OOF，合计 12 组 111-cat 结果。

有限候选池如下：

| Policy | 训练输入变化 | 作用 |
| --- | --- | --- |
| `P0_identity` | 保持 fbank 数值不变 | 在线执行路径对照 |
| `P1_specaugment_light` | 以 0.5 概率遮蔽至多 4 帧时间区和 4 个频率 bin | 固定轻量时频遮蔽；A1 使用此策略 |
| `P2_gain_noise_light` | 归一化增益偏移 `[-0.1, 0.1]`，并以 0.5 概率加入标准差 0.015 的高斯噪声 | 检验能量和轻噪声鲁棒性 |
| `P3_shift_combo_light` | 有效时间区循环平移至多 3 帧，并使用减半的增益扰动 | 检验轻时间偏移与增益组合 |

时间 mask 宽度同时受“至多有效帧的 12.5%”约束。675/792 条 call 的时长不超过 1 秒，
有效 fbank 帧中位数约为 70，因此首轮采用窄 mask，并将 pitch shift 和 time stretch 留在候选池之外。

## 3. Smoke、执行路径与审计

- `P0_identity` 输入逐位不变；所有策略保持 padding 区逐位不变；
- 同一 RNG stream 的随机增强逐位复现，不同 streams 生成不同增强视图；
- smoke fold 的 P0/P1 batch order 相同：517 条训练 call、65 个 8-call microbatch、17 次
  optimizer step；
- frozen AST 在线输出与历史缓存 embedding 的最大绝对差为
  `2.1576881408691406×10⁻⁵`；
- 144 个选择 fit 完整结束，选择记录为 `outer_test_accessed=false`；
- 12 个 `(repeat, outer fold)` lock 全部唯一；11/12 个 lock 满足预设 stream-stability 条件；
- execution lock 在读取 outer test 前固定 protocol、runner、环境和 12 个 policy locks；
- R0 Macro F1 精确复现历史 strict reference：`0.7570206188850608`；
- 原始预测清单包含 120 个文件、1,898,700 bytes，聚合 SHA-256 为
  `256813341d557f1fac2b3e04f7d0272c207a0b9c31303d371b57973660415673`。

### 3.1 执行调整记录

最初 smoke 将在线/缓存 embedding 最大差阈值设为 `1×10⁻⁵`，实测差值为
`2.1577×10⁻⁵`。该差值来自冻结 AST 在不同 batch shape 下的 GPU 浮点执行顺序。Outer test
仍封闭时，阈值调整为 `3×10⁻⁵`，并将每个 epoch 的在线编码按 32 个 segment 批量执行，再按原
8-call microbatch 顺序训练分类头。分类头的 BatchNorm、dropout、call weight、optimizer step 和
batch order 保持原定义。最终 runner 与 protocol 随后提交并进入 execution lock。

调整前的慢速 smoke 与两个 inner fit 已转移到
`runs/meowagenet_idea039_grouped_augmentation_v1_superseded_online_microbatch_2026-09-16`，最终
选择和外层结果全部来自锁定 runner。

## 4. 每折 policy lock

| Repeat | Fold | 选中 policy | Epoch | Inner Macro F1 | Inner animal CE | Top-2 streams | 稳定 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | `P3_shift_combo_light` | 12 | 0.6718 | 0.8310 | 2 | 是 |
| 0 | 1 | `P2_gain_noise_light` | 9 | 0.8491 | 0.4834 | 3 | 是 |
| 0 | 2 | `P1_specaugment_light` | 16 | 0.8713 | 0.3445 | 3 | 是 |
| 0 | 3 | `P0_identity` | 7 | 0.7738 | 1.1684 | 3 | 是 |
| 1 | 0 | `P1_specaugment_light` | 5 | 0.7000 | 0.7933 | 3 | 是 |
| 1 | 1 | `P0_identity` | 24 | 0.7519 | 0.5436 | 3 | 是 |
| 1 | 2 | `P2_gain_noise_light` | 11 | 0.8148 | 0.8361 | 3 | 是 |
| 1 | 3 | `P3_shift_combo_light` | 5 | 0.7387 | 0.7341 | 2 | 是 |
| 2 | 0 | `P1_specaugment_light` | 16 | 0.6923 | 0.6848 | 2 | 是 |
| 2 | 1 | `P0_identity` | 7 | 0.8643 | 0.7689 | 1 | 否 |
| 2 | 2 | `P3_shift_combo_light` | 6 | 0.7874 | 0.7080 | 3 | 是 |
| 2 | 3 | `P3_shift_combo_light` | 1 | 0.7460 | 0.8532 | 2 | 是 |

选中次数为：P0 3 折、P1 3 折、P2 2 折、P3 4 折。第一名与第二名的 inner Macro F1
平均差为 `0.0232`，范围为 `0–0.0791`；3/12 个 folds 的首位 Macro F1 精确并列，由 animal
CE 完成决胜。四种策略均被选中，说明内层偏好随猫分组变化；11/12 的流稳定性说明排序没有由
单一 augmentation stream 主导。

## 5. Complete-OOF 主结果

| Pipeline | Animal Macro F1 | Balanced accuracy | QWK | Accuracy | Animal CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 cached frozen AST | 0.7570 ± 0.0086 | 0.7645 | 0.6721 | 0.7568 | **0.7160** |
| C0 online identity | 0.7557 ± 0.0155 | 0.7615 | 0.6693 | 0.7568 | 0.7685 |
| A1 fixed SpecAugment | **0.7578 ± 0.0405** | **0.7671** | **0.6830** | 0.7568 | 0.7443 |
| A2 nested policy | 0.7556 ± 0.0231 | 0.7624 | 0.6658 | 0.7538 | 0.7802 |

A1 相对 R0 的平均变化为：Macro F1 `+0.0008`、Balanced Accuracy `+0.0027`、QWK
`+0.0109`、Accuracy `0.0000`、animal CE `+0.0284`。它在平均分类平衡和年龄顺序相关错误上
略有改善，并保持完全相同的平均准确率；同时 repeat 间 Macro F1 波动由 `0.0086` 增至
`0.0405`。因此，固定轻 SpecAugment 形成弱正的副指标信号，主指标增益很小且跨 split 波动较大。

A2 相对 R0 的平均变化为：Macro F1 `−0.0014`、Balanced Accuracy `−0.0021`、QWK
`−0.0064`、Accuracy `−0.0030`、animal CE `+0.0642`。A2 相对 A1 的 Macro F1 为
`−0.0022`，QWK 为 `−0.0173`。内层按 fold 选择确实找到了不同 policy，但这种差异没有转化为
更好的外层 complete-OOF 结果。

### 5.1 Repeat-level 配对结果

| Repeat | R0 | C0 | A1 | A2 | A1−R0 | A2−R0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7659 | 0.7548 | 0.7792 | 0.7708 | +0.0133 | +0.0049 |
| 1 | 0.7565 | 0.7716 | 0.7832 | 0.7670 | +0.0267 | +0.0106 |
| 2 | 0.7487 | 0.7406 | 0.7111 | 0.7290 | −0.0376 | −0.0197 |
| **平均** | **0.7570** | **0.7557** | **0.7578** | **0.7556** | **+0.0008** | **−0.0014** |

A1 和 A2 相对 R0 都在 2/3 repeats 中为正；repeat 2 的下降抵消了前两轮收益。按猫配对的
5,000 次 cluster bootstrap 给出 A1−R0 Macro F1 95% 区间 `[-0.0301, +0.0293]`，
A2−R0 区间 `[-0.0311, +0.0296]`，A2−A1 区间 `[-0.0289, +0.0261]`。三个区间均跨越
0，和 repeat 方向共同显示当前增益处于小幅混合状态。

### 5.2 C0 路径门与同路径比较

C0−R0 的 Macro F1 为 `−0.00134`，落在预设 `±0.005` 范围内；平均预测一致率为
`95.80%`，低于预设 `99%`。每个 repeat 平均有 4.67/111 只猫改变类别，其中平均新增与损失
的正确猫均为 2.33 只，因此平均 Accuracy 完全相同。结果说明在线 frozen-AST 路径保持了总体
性能水平，同时边界附近个体的类别发生了交换，路径门状态为 `passed=false`。

同路径辅助比较更直接地反映增强内容：

| 对比 | Macro F1 | Balanced accuracy | QWK | Accuracy | Animal CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1−C0 | +0.0022 | +0.0056 | +0.0137 | 0.0000 | **−0.0242** |
| A2−C0 | −0.0001 | +0.0009 | −0.0035 | −0.0030 | +0.0116 |

A1 相对 C0 在 repeats 0/1 分别提高 `+0.0244`、`+0.0116`，repeat 2 下降 `−0.0295`。
固定 SpecAugment 在同路径下同时改善平均 QWK、Balanced Accuracy 和 animal CE，但平均 Macro
F1 增益仍为 `+0.0022`，低于 `+0.005` 继续门。A2 与 C0 的平均 Macro F1 基本相同，并在
1/3 repeats 中领先，当前 selector 没有形成额外收益。

### 5.3 类别表现

| Pipeline | Kitten recall | Adult recall | Senior recall | Kitten F1 | Adult F1 | Senior F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.8000 | 0.7581 | **0.7353** | 0.7914 | 0.7853 | **0.6943** |
| C0 | 0.8000 | **0.7688** | 0.7157 | 0.7905 | **0.7876** | 0.6890 |
| A1 | **0.8222** | 0.7634 | 0.7157 | 0.7980 | 0.7833 | 0.6923 |
| A2 | **0.8222** | **0.7688** | 0.6961 | **0.8056** | 0.7851 | 0.6762 |

A1 相对 R0 将 Kitten recall 提高 `0.0222`、Adult recall 提高 `0.0054`，Senior recall
降低 `0.0196`。A2 同样提高 Kitten 与 Adult recall，同时 Senior recall 降低 `0.0392`。
这解释了 A1 的 Balanced Accuracy/QWK 小幅改善，以及 A2 的总体收益被 Senior 类代价抵消。

### 5.4 Call-count 分层

以下解释性分析将三次 repeat 的预测合并为 333 个“猫×repeat”观察，并按每只猫的 call 数量
分成接近等规模的三组：1–2 calls、3–8 calls、9+ calls。

| Pipeline | 1–2 calls F1 | 3–8 calls F1 | 9+ calls F1 |
| --- | ---: | ---: | ---: |
| R0 | 0.6671 | **0.8179** | 0.7862 |
| C0 | **0.6744** | 0.8001 | 0.7938 |
| A1 | 0.6641 | 0.8162 | 0.7967 |
| A2 | 0.6570 | 0.8058 | **0.8015** |

A1 与 A2 的正向差异集中在 9+ calls 的猫，分别比 R0 高约 `0.0105` 和 `0.0153`；1–2
calls 与 3–8 calls 组持平或略低。拥有更多 calls 时，animal probability aggregation 可以平均
多个增强训练后形成的 call-level 变化；calls 较少时，单条预测改变对猫级结果的影响更大。这个
分层结果用于解释 repeat 波动，完整 111-cat OOF 继续承担主结论。

## 6. Gate 与阶段结论

H039-A 的六个性能检查中五项通过：2/3 positive repeats、Balanced Accuracy、QWK、animal
CE 和逐类别 recall 代价均在预设范围；平均 Macro F1 增益为 `+0.0008`，低于 `+0.005`。
H039-B 的性能检查中，平均 Macro F1 增益和 animal CE 两项未通过；stream stability 达到
11/12，满足预设的至少 8/12。C0 的预测一致率同时未达到路径门。

本轮形成四条可直接进入论文实验章节的结论：

1. 固定轻 SpecAugment 保持了 frozen AST 的总体水平：A1 Macro F1 为
   `0.7578 ± 0.0405`，R0 为 `0.7570 ± 0.0086`；其 QWK 提高 `0.0109`，说明年龄顺序相关
   错误略有改善，同时跨 splits 的波动明显增加；
2. 每折 nested selector 选出了四种不同策略，11/12 折达到流稳定标准；A2 Macro F1 为
   `0.7556 ± 0.0231`，相对 A1 为 `−0.0022`，当前内层策略偏好没有带来外层增益；
3. 增强对拥有 9 条及以上 calls 的猫显示较积极的解释性信号，对 calls 较少的猫收益较弱；该
   分层差异与 repeat 波动共同指向数据量和 animal aggregation 的交互；
4. 在线路径与缓存路径的平均性能接近，预测一致率为 95.80%；后续训练侧输入实验应保留同路径
   identity control，并把同路径差异作为增强净作用的主要辅助证据。

按照冻结规则，H039-A、H039-B 和路径门均为 `passed=false`，seeds 43/101 不进入扩展。本轮
有限增强池以“固定增强弱正副指标信号、nested selector 未形成增益”阶段性收尾。下一阶段按既定
路线进入 Stage D 的低成本 nuisance-variable 诊断，先分析时长、有效帧比例和能量是否与现有
OOF 预测、置信度和错误方向系统相关，再决定是否建立新的方法 IDEA。

## 7. 计算成本

| Pipeline | 可训练参数 | 总参数 | 平均选中 epoch | 平均 inner 时间/fit | 平均 outer 时间/fit | 峰值 VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 99,075 | 99,075（缓存 head） | 10.17 | 2.28 s | 1.40 s | 19.3 MiB |
| C0 | 99,075 | 85,466,115 | 9.75 | 11.15 s | 4.11 s | 591.9 MiB |
| A1 | 99,075 | 85,466,115 | 9.42 | 85.34 s | 20.02 s | 591.9 MiB |
| A2 | 99,075 | 85,466,115 | 9.92 | 65.95 s | 15.43 s | 591.9 MiB |

在线三条 pipeline 都冻结 AST backbone，只训练 99,075 个分类头参数。增强增加了每个 epoch
重新生成 fbank 视图和执行 frozen AST 前向传播的成本；GPU 峰值仍低于 0.6 GiB。

## 8. 关键文件

- 研究计划：`plan/IDEA-039_grouped_augmentation_policy.md`；
- Protocol：`configs/protocol/meowagenet_idea039_grouped_augmentation_v1.json`；
- Runner：`scripts/run_meowagenet_idea039_grouped_augmentation.py`；
- Smoke：`runs/meowagenet_idea039_grouped_augmentation_v1/smoke/summary.json`；
- 每折 policy locks：
  `runs/meowagenet_idea039_grouped_augmentation_v1/selection/per_fold_policy_locks.json`；
- Execution lock：`runs/meowagenet_idea039_grouped_augmentation_v1/execution_lock.json`；
- 机器可读结果：
  `metadata/experiments/meowagenet_idea039_grouped_augmentation_v1_results.json`；
- 下一阶段记录：`plan/POST_IDEA039_stage_update.md`。
