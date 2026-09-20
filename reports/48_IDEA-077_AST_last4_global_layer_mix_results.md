# IDEA-077：AST 最后四层全局 LayerMix 结果

## 结论

预注册的 `L1_global_layermix` 没有优于精确 final-layer 基线 `A0_final`，主 gate 未通过。九个等权 `base_seed × repeat` 单元中，L1 的平均 Macro-F1 比 A0 低 `0.01022`；三个新 base seeds 的平均差值全部为负，只有 `3/9` 个 seed×repeat 为正。CE、Brier、split 稳定性和 senior recall 保护条件也未通过。

因此，本轮不支持把“AST 最后四层的全局可学习混合”保留为当前 MeowAgeNet 的表示模块候选。固定均值 `M0_uniform_last4` 同样没有优于 A0；不得依据这些结果改选层数、删层、改 gate 或追加结果驱动种子。

## 实验边界与完成状态

- 数据：MeowAgeNet 792 calls、111 cats。
- 角色：formal-v2 nested roles；只使用 train/validation，`outer_test_accessed=false`。
- 管线：A0 final、M0 固定 last-4 均值、L1 零影响全局 LayerMix。
- 新 base seeds：`6917, 1398, 5934`。
- 预算：`3 pipelines × 3 seeds × 3 repeats × 4 folds = 108 fits`；全部完成。
- L1 初始化：`alpha=0`、`gamma=0`，优化前 A0/L1 logits 和 loss 差均为 `0`。
- 参数量：A0/M0 各 `99,075`；L1 `99,080`，仅增加 5 个全局参数。
- 执行环境：Python 3.10.12、PyTorch 2.2.2+cu121、NVIDIA GeForce RTX 4060 Ti。

## 主要结果

以下指标先在每个 `base_seed × repeat` 单元合并四个 validation folds，再对九个单元等权平均。

| Pipeline | Macro-F1 | Balanced accuracy | Animal CE | Animal Brier |
|---|---:|---:|---:|---:|
| A0 final | 0.74157 | 0.77037 | 0.68537 | 0.40674 |
| M0 uniform last-4 | 0.72261 | 0.76481 | 0.67458 | 0.41184 |
| L1 global LayerMix | 0.73135 | 0.76574 | 0.68598 | 0.41306 |

配对 Macro-F1：

| Comparison | Mean delta | Positive / tied / negative | Worst | Best |
|---|---:|---:|---:|---:|
| L1 − A0 | -0.01022 | 3 / 0 / 6 | -0.07043 | +0.03539 |
| M0 − A0 | -0.01897 | 3 / 1 / 5 | -0.08744 | +0.06576 |
| L1 − M0 | +0.00874 | 5 / 0 / 4 | -0.07043 | +0.06649 |

M0 的平均 CE 比 A0 低 `0.01079`，但 Macro-F1 低 `0.01897`，Brier 反而高 `0.00509`；这不是一致的概率质量或分类收益。L1 相对 M0 有小幅平均优势，但两者都低于 A0，所以不能据此支持可学习 LayerMix。

## 主 gate 审计

`L1 − A0` 的八项条件全部失败。

| 条件 | 预注册阈值 | 观察值 | 结果 |
|---|---:|---:|---|
| 平均 Macro-F1 增益 | ≥ +0.005 | -0.01022 | Fail |
| 正向 base seeds | 3/3 | 0/3 | Fail |
| 正向 seed×repeat | ≥ 6/9 | 3/9 | Fail |
| 非负 split-cells | ≥ 8/12 | 5/12 | Fail |
| 最差 split-cell | ≥ -0.03 | -0.08853 | Fail |
| Animal CE 不劣 | L1 ≤ A0 | 0.68598 > 0.68537 | Fail |
| Animal Brier 不劣 | L1 ≤ A0 | 0.41306 > 0.40674 | Fail |
| 每个 seed senior recall | ≥ -0.02 | 最差 -0.03333 | Fail |

三个 base-seed 的 L1−A0 Macro-F1 均值分别为：

- seed 6917：`-0.01540`
- seed 1398：`-0.00737`
- seed 5934：`-0.00790`

对应 senior recall 差值为 `-0.03333、-0.01667、+0.03333`。12 个共同 split-cells 为 4 正、1 平、7 负，范围 `-0.08853～+0.05165`。

## LayerMix 行为

36 个 L1 最优 checkpoint 的 signed residual gate 全部保持为小的正值：均值 `0.08257`、标准差 `0.04724`、范围 `0.00121～0.17838`。最后四层的平均权重为：

| AST block | Mean weight | SD |
|---|---:|---:|
| 9 | 0.25140 | 0.01026 |
| 10 | 0.26324 | 0.01422 |
| 11 | 0.24946 | 0.01318 |
| 12 | 0.23590 | 0.01315 |

模型确实离开了零 gate，且对第 10 层略有偏重，但权重整体仍接近均匀。结合负的主结果，更合理的解释是：最后四层含有可被轻微利用的互补信息，但当前全局、样本无关的混合不足以稳定改善年龄分类；它偶尔改变错误边界，也会在部分 split 上造成明显退化。

## 缓存与表示审计

本轮没有重新运行 AST，也没有新建 hidden-state cache。既有缓存的几何为 `792×12×768`；每层都使用共同最终 LayerNorm、CLS/蒸馏 token 均值以及 call 内 segment 均值。缓存 block 12 与锁定 `pooler_output` 的平均绝对差为 `1.36479e-6`、最大为 `1.81198e-5`；L1 的第 12 层位置直接替换为精确锁定 final embedding，所以零 gate 初态与 A0 完全相同。

## 独立审计与可复现性

- `--resume` 两次完整重读后，汇总 SHA-256 保持 `7d0c974a3e8825426ee92e79b5492717352866bcc19ab2903ff833b7cdd9fa9b`，run summary 哈希保持 `0afa907f12c7c52dc7aa4491f7968ef4233cbd2997be4344620add2891e33eca`。
- 独立审计检查 108 个 fit summaries、216 个预测文件、12,015 条 call 预测和 1,836 条 animal 预测。
- 从 call 预测重新聚合到 animal 的最大数值差为 `1.11e-16`。
- 所有 checkpoint 重载后的最大概率差为 `0`。
- 独立重算的均值、配对差、split-cell、senior recall 与八项 gate 均和锁定 runner 完全一致。

完整产物位于 `runs/meowagenet_idea077_ast_last4_layer_mix_v1/`；独立审计为 `independent_audit.json`。

## 决策

按照预注册 fail action：停止当前固定 last-4 全局 LayerMix 公式，不把它纳入当前 MeowAgeNet 主候选，也不进行结果驱动的层数、gate 或种子修补。A0 final representation 仍是本轮表示对照中最稳健的选择。M0/L1 的正向 split 和 learned gate/weights 可以保留为未来提出不同、事前有明确机制的新 IDEA 时的背景证据，但不能被解释为本轮成功。
