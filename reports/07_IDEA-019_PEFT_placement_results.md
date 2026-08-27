# IDEA-019 PEFT Placement 诊断实验报告

> 日期：2026-08-26
> 状态：12 层 probe、七种条件、四折 GPU pilot 与 2,000 次 paired bootstrap 已完成
> 主指标：animal-level macro F1

> 2026-08-27 路线更新：本报告提名的 Probe-guided AST adapter 已在
> `reports/08_IDEA-048_stage_checkpoint.md` 中保留为当前参考候选；后续重复实验、
> 新 idea 与候选替换保持开放。Head-only、Distributed 和 Random 保留为必要消融。

## 1. 结论摘要

IDEA-019 获得了**有性能信号的部分支持**。同样大小的 adapter 放在不同 AST 层，
会明显改变猫龄分类结果；probe-guided placement 是本轮最高方案，同时 random
placement 与它非常接近，因此“layer probe 能独特定位最佳适配层”的机制主张仍需
限定。

| Pipeline | Macro F1 | Balanced accuracy | QWK | 正确猫数 |
|---|---:|---:|---:|---:|
| Head-only frozen AST | 0.7260 | 0.7696 | **0.6791** | 81 / 111 |
| Adapter early | 0.7204 | 0.7243 | 0.5963 | 79 / 111 |
| Adapter middle | 0.7445 | 0.7617 | 0.6360 | 82 / 111 |
| Adapter late / last-k | 0.7298 | 0.7678 | 0.6606 | 80 / 111 |
| Adapter distributed | 0.7501 | 0.7769 | 0.6616 | **84 / 111** |
| Adapter random | 0.7545 | 0.7556 | 0.6419 | **84 / 111** |
| **Adapter probe-guided** | **0.7575** | **0.7795** | 0.6755 | 83 / 111 |

核心结果有四点：

1. Probe-guided 相对 head-only 提高 **0.0315 macro F1**，点估计刚超过预设的
   `δ_task = 0.03`；balanced accuracy 同时提高 0.0099，QWK 只变化 -0.0036。
2. Probe-guided 相对固定 late placement 提高 0.0277，并在四个 outer folds 中
   三胜一平，说明“惯用最后两层”不是当前任务的最佳规则。
3. Probe-guided 相对 random 只高 0.0030；random 与 distributed 也取得 0.7545 和
   0.7501。位置确实重要，但当前结果同时支持“宽分布/优化路径也能带来收益”。
4. 每个 adapter pipeline 只增加 99,904 个可训练 adapter 参数，总可训练参数为
   198,979。Probe-guided 高于此前 last-2 fine-tuning 的 0.7338 和 full fine-tuning
   的 0.7251，同时比它们少约 72 倍和 430 倍可训练参数。

因此，probe-guided adapter 值得作为当前 **provisional AST performance candidate**
保留；random 与 distributed 作为关键 ablation 保留。IDEA-019 的“PEFT 有效且
placement 会影响结果”得到支持，“probe 分数能够唯一解释 placement utility”得到
部分支持。

## 2. IDEA-019 在研究什么

PEFT 是 parameter-efficient fine-tuning，即参数高效微调。它冻结大模型主体，只在
少量位置加入可训练模块。本轮固定一种 family：**block-output residual bottleneck
adapter**。

每个 adapter 执行：

`hidden → Dense(32) → GELU → Dense(768) → 加回原 hidden`

第二个 Dense 使用零初始化，所以训练开始时 adapter 是严格恒等映射。三个抽查
placements 的初始 logits 最大差为 0；训练后的差异来自 adapter 被放置的位置和其
对应梯度路径。

本轮的核心问题不是“adapter 是否比 LoRA 好”，而是：

> 当 PEFT family 和可训练参数完全一致时，插入 AST 的哪些层会更好地适配猫龄任务，
> frozen layer probe 能否事前预测这些层？

## 3. 冻结的 placement 矩阵

AST 有 12 个 Transformer blocks。下表使用 1-based 层号：

| Placement | 层 | 含义 |
|---|---|---|
| Early | 1 + 2 | 适配最早的声学处理层 |
| Middle | 6 + 7 | 适配中间表征层 |
| Late / last-k | 11 + 12 | 惯用的最后两层适配 |
| Distributed | 3 + 10 | 在深度轴上分散两个 adapter |
| Random | 2 + 10 | seed 20260826 事前生成的随机对照 |
| Probe-guided | 每个 outer fold 动态选择 | 由 outer-train 内 layer probes 选得分最高两层 |

六种 adapter 条件均为：

- 两个 width-32 adapters；
- 99,904 个 adapter 参数；
- 198,979 个总可训练参数（含统一 MLP head）；
- 相同 optimizer、learning rate、batch、early stopping 和初始化；
- AST checkpoint、geometry、输入、segment pooling 和 cat-ID folds 完全一致。

## 4. Layer probe 如何选层

先从 frozen standard AST 提取每个 block 的表示：对该层输出应用最终 LayerNorm，
平均 CLS 与 distillation tokens，再平均同一 call 的 segments。正式 FP32 层表示包含：

- 843 个 segments；
- 792 个 calls；
- 111 只 cats；
- 12 层 × 768 维。

第 12 层与锁定 `pooler_output` 的平均绝对差为 `1.36e-6`、最大差为 `1.81e-5`，
说明 probe 表示与既有 frozen AST 对齐。

对每个 outer fold，layer probe 只读取该 fold 的 inner-train cats，再进行三折 cat-level
CV。12 层使用相同 StandardScaler 与 balanced logistic probe，按平均 animal macro F1
排名，选择前两层。outer-test cats 不参与选层。

四折选层结果：

| Outer fold | Probe-guided layers |
|---|---|
| Fold 0 | 5 + 8 |
| Fold 1 | 7 + 10 |
| Fold 2 | 10 + 12 |
| Fold 3 | 2 + 7 |

第 7 层和第 10 层各被选择两次，但没有一组层在四折中固定重复。这个结果说明猫龄
可解码信息常位于中后层，同时 probe ranking 对训练动物组成较敏感。

## 5. 训练协议与资源

- Backbone：MIT AudioSet standard AST，12 blocks；
- 数据：792 calls、843 segments、111 cats；
- 划分：固定四个 cat-ID-disjoint outer folds；
- Segment：1.28 秒窗口、0.64 秒 hop；
- Call representation：同一 call 的 AST `pooler_output` 算术平均；
- Head：768→128、ReLU、BatchNorm、Dropout、3 classes；
- Adapter learning rate：`1e-3`；head learning rate：`0.003109800273709165`；
- Adamax、micro-batch 8、gradient accumulation 4、FP16；
- 最大 50 epochs，inner `val_loss` patience 8；
- 每折根据 inner validation 选择 epoch，再用 outer train+validation 重训；
- Primary：animal macro F1；secondary：balanced accuracy、QWK、per-class recall。

## 6. 四折表现

| Pipeline | Fold 0 | Fold 1 | Fold 2 | Fold 3 | 合并 F1 |
|---|---:|---:|---:|---:|---:|
| Head-only | **0.8125** | 0.6537 | 0.8100 | 0.6251 | 0.7260 |
| Early | 0.7737 | 0.5889 | 0.7573 | 0.7581 | 0.7204 |
| Middle | **0.8125** | 0.6643 | 0.7649 | 0.7341 | 0.7445 |
| Late | 0.7737 | 0.6323 | 0.8100 | 0.7122 | 0.7298 |
| Distributed | 0.7737 | **0.7167** | 0.7856 | 0.7242 | 0.7501 |
| Random | 0.7737 | 0.6693 | 0.7856 | **0.7838** | 0.7545 |
| Probe-guided | 0.7737 | 0.6458 | **0.8472** | 0.7643 | **0.7575** |

相对 head-only，probe-guided 在 fold 0、1 下降，在 fold 2、3 提高。合并增益主要来自
fold 2 的层 10+12 和 fold 3 的层 2+7，因此它达到了实践阈值点估计，但方向尚未在
四折全部重复。

相对 late，probe-guided 的差值分别为 `0、+0.0135、+0.0372、+0.0521`，方向更整齐。
这支持 probe-guided 比固定 last-k 更适合当前任务。相对 random，它在 fold 2 领先，
在 fold 1、3 落后，fold 0 持平；这限制了更强的 layer-semantic 解释。

## 7. Paired bootstrap

使用同一组 111 只 outer-test cats、按真实年龄类别分层重采样 2,000 次：

| 对比 | ΔMacro F1 | 95% bootstrap 区间 |
|---|---:|---:|
| Probe-guided − Head-only | **+0.0315** | `[-0.0287, +0.0915]` |
| Probe-guided − Late | +0.0277 | `[-0.0090, +0.0702]` |
| Probe-guided − Random | +0.0030 | `[-0.0552, +0.0667]` |
| Random − Head-only | +0.0285 | `[-0.0427, +0.0923]` |
| Distributed − Head-only | +0.0241 | `[-0.0407, +0.0857]` |

Probe-guided 的 +0.0315 是本轮唯一越过 `δ_task=0.03` 的点估计。111-cat bootstrap
区间仍覆盖 0，说明真实增益幅度可能较小，也可能达到约 9 个百分点。它作为性能候选
有明确理由保留；其效果强度将在后续冻结评价中继续确认。

## 8. Probe 分数是否预测 adapter 收益

把每个 fold 的六个 adapter placements 按平均 layer-probe score 排序，再与该
placement 的 outer macro-F1 gain 对应：

| Fold | Spearman ρ |
|---|---:|
| Fold 0 | 0.131 |
| Fold 1 | 0.314 |
| Fold 2 | **0.812** |
| Fold 3 | 0.657 |
| 24 个 placement-fold observations 合并 | **0.489** |

Fold 2、3 的 probe–adaptation 对应较强，fold 0、1 较弱。合并探索性相关为 0.489，
说明高 probe utility 与较好的 placement gain 存在中等正对应；这些 observations
共享 folds 和模型，不作为 24 个独立实验解释。

结合 primary contrasts，最合适的机制判断是：

- layer probe 能提供有用的 placement 排序信息；
- 固定 last-k 会错过部分中间/分散层机会；
- random 也很强，说明梯度路径、优化稳定性和分散覆盖同样参与了收益；
- 当前还不能把收益全部解释成“某层包含独特年龄机制”。

## 9. 类别变化

Head-only：

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 13 | 2 | 0 |
| Adult | 7 | 42 | 13 |
| Senior | 1 | 7 | 26 |

Probe-guided：

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 14 | 1 | 0 |
| Adult | 3 | 47 | 12 |
| Senior | 1 | 11 | 22 |

Probe-guided 多识别对 1 只 kitten 和 5 只 adult，少识别对 4 只 senior，总正确数从
81 增至 83。Kitten recall 从 0.8667 提高到 0.9333，adult 从 0.6774 提高到 0.7581，
senior 从 0.7647 降到 0.6471。

Macro F1 与 balanced accuracy 因 kitten/adult 改善而提高；QWK 基本保持，是因为新增的
senior→adult 相邻错误抵消了其他类别收益。这说明 adapter 带来的主要变化是重新分配
中高年龄边界，而不是简单提高所有类别。

## 10. 参数与计算效率

| 状态 | 总可训练参数 | 相对 adapter | 四折训练与预测时间 | 峰值显存 |
|---|---:|---:|---:|---:|
| Head-only | 99,075 | 0.50× | 8.9 秒 | 0.02 GiB |
| Probe-guided adapter | 198,979 | 1× | 81.2 秒 | 0.91 GiB |
| Last-2 fine-tuning（既有） | 14,276,355 | 71.7× | 80.0 秒 | 0.69 GiB |
| Full fine-tuning（既有） | 85,466,115 | 429.5× | 164.4 秒 | 2.31 GiB |

Adapter 的参数效率非常明显；wall-clock 没有按参数量同比下降，因为每个 batch 仍需
完成完整 AST forward，而且较早 adapter 的梯度需要穿过后续 frozen blocks。相同参数
下，early 峰值显存约 0.97 GiB，late 只有约 0.43 GiB，说明 placement 同时影响反向
图长度和资源消耗。

## 11. 与已有模型的关系

| Pipeline | Macro F1 |
|---|---:|
| 锁定 VGGish + MLP baseline | 0.6846 |
| Head-only Frozen AST（同 PyTorch recipe） | 0.7260 |
| Last-2 fine-tuning | 0.7338 |
| Full fine-tuning | 0.7251 |
| **Probe-guided adapter** | **0.7575** |

Probe-guided 比同 recipe 的 head-only 高 0.0315，比 last-2 高 0.0237，比 full 高 0.0324。
它以最低的 encoder adaptation 参数取得当前 AST 系列最高点值。

它也比锁定 VGGish baseline 高 0.0729；其中 0.0414 来自新的 PyTorch AST head recipe
相对旧 frozen-AST recipe 的变化，adapter 在同 recipe 下贡献的可归因增量是 0.0315。
项目仍保留唯一的锁定 VGGish baseline，不新增第二个 baseline。

## 12. 路线判断

本轮形成四条可用于论文的结论：

1. **低参数适配有效。** Probe-guided adapter 用约 10 万 encoder-adaptation 参数取得
   当前 AST 系列最高 macro F1 0.7575。
2. **Placement 会影响泛化。** Early、middle、late、distributed 的 F1 范围为
   0.7204–0.7501，相同参数预算不能消除位置差异。
3. **Probe-guided 优于惯用 last-k。** 四折三胜一平、合并 +0.0277，说明 probe
   提供了有效选层信息。
4. **Probe 机制获得部分支持。** Random 与 probe-guided 只差 0.0030，优化路径和
   分散层覆盖仍是强竞争解释。

当前决策是：**保留 probe-guided adapter 为 provisional AST performance candidate，
保留 random/distributed 为必要 ablations；IDEA-019 在本阶段完成，Probe-guided
进入 IDEA-048 阶段性 checkpoint，作为可被后续新证据替换的 current reference。**

## 13. 产物与完整性

- 正式汇总：`runs/idea019_peft_placement_v1/summary.json`
- Layer probes：`runs/idea019_peft_placement_v1/layer_probe_results.csv`
- Probe–placement 对应：`runs/idea019_peft_placement_v1/probe_placement_correspondence.csv`
- 七种 call/animal predictions：`runs/idea019_peft_placement_v1/`
- FP32 12-layer representations：`runs/ast_layer_embeddings_v2_float32/`
- 训练脚本：`scripts/run_idea019_peft_placement.py`
- 层表示脚本：`scripts/extract_ast_layer_embeddings.py`
- 机器元数据：`metadata/experiments/meowagenet_idea019_peft_placement_v1_results.json`

完整性核验确认：每种条件均包含 792 个唯一 calls、111 个唯一 cats；四折 call 数为
186/191/211/204；每只猫只进入一个 outer-test fold；所有概率有限、非负且和为 1，
最大概率和误差小于 `1.33e-7`；六种 adapter placement 的 adapter 参数均严格为 99,904，
训练前抽查 logits 完全一致。Head-only 指标逐值复现既有 frozen control。
