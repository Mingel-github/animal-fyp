# IDEA-048 最终候选冻结记录

> 日期：2026-08-27
> Freeze ID：`meowagenet-idea048-candidate-freeze-v1`
> 状态：**Stage C candidate freeze 已完成；Stage D 最终比较已就绪**

## 1. 冻结结论

IDEA-048 的最终 performance candidate 冻结为：

> **MIT standard AST + outer-train-only layer probe + two width-32 residual
> bottleneck adapters + 768→128 三分类 MLP head**

简称 **Probe-guided AST adapter**。

项目继续保留唯一的正式 baseline：**锁定 VGGish + MLP**。Head-only AST 是 adapter
净增益对照；Random 和 Distributed adapters 是必要 placement ablations，不包装成
额外 baseline。

本次 freeze 固定候选身份、选层规则、训练流程、评估协议和对照角色。它不重新训练
模型，也不改变已经完成的 IDEA-019 结果。机器可读配置位于
`configs/protocol/meowagenet_idea048_candidate_freeze_v1.json`。

## 2. 冻结时的性能依据

| Pipeline | Animal macro F1 | Balanced accuracy | QWK | 论文角色 |
|---|---:|---:|---:|---|
| VGGish + MLP | 0.6846 | 0.6754 | 0.5263 | 唯一正式 baseline |
| Head-only frozen AST | 0.7260 | 0.7696 | **0.6791** | 同 recipe 的 adapter 净效应对照 |
| Distributed adapter | 0.7501 | 0.7769 | 0.6616 | 必要 placement ablation |
| Random adapter | 0.7545 | 0.7556 | 0.6419 | 必要 probe-guidance ablation |
| **Probe-guided AST adapter** | **0.7575** | **0.7795** | 0.6755 | **最终性能候选** |

当前候选相对锁定 VGGish baseline 的 macro F1 提高 **0.0729**，超过事前保留的
实践差异阈值 `δ_task = 0.03`；相对同一 PyTorch AST recipe 的 Head-only control
提高 **0.0315**。因此，IDEA-048 已经得到清晰的性能提升信号，Probe-guided AST
adapter 进入最终比较是当前结果支持的直接决策。

Random 与 Probe-guided 仅相差 0.0030，所以最终论文把主要性能贡献表述为
**低参数、跨层 AST adaptation 带来的完整 pipeline 提升**；probe-guided selection
取得最高点估计并优于惯用 late placement，其独特机制优势由消融表限定说明。

## 3. 冻结的候选规则

### 3.1 数据与评估

- 使用 `meowagenet-locked-v1`：792 段唯一 calls、111 只分析猫、三年龄类别；
- 固定四个 cat-ID-disjoint outer folds 和既有 nested roles；
- outer-test cats 只用于该折最终预测；
- 同一猫的 call 概率取算术平均后得到 animal prediction；
- Primary：animal-level macro F1；
- Secondary：balanced accuracy、QWK、逐类 recall、混淆矩阵与计算成本；
- 实践差异阈值固定为 `0.03 macro F1`。

### 3.2 AST 与输入

- Checkpoint：`MIT/ast-finetuned-audioset-10-10-0.4593`；
- Revision：`f826b80d28226b62986cc218e5cec390b1096902`；
- 使用 standard AST geometry：frequency/time stride 均为 10；
- 音频为 16 kHz，1.28 秒窗口、0.64 秒 hop；
- 位置网格使用既有 bilinear interpolation，patch projection 权重保持预训练值；
- 同一 call 的多个 segment `pooler_output` 取算术平均。

### 3.3 Probe-guided placement

冻结的是**选层程序**，不是一组人工指定的全局层号：

1. 每个 outer fold 独立运行；
2. 只使用该折的 inner-train cats；
3. 对 AST 12 层 FP32 frozen call representations 分别运行三折 cat-level CV；
4. 每层使用相同的 StandardScaler 和 balanced logistic regression；
5. 按平均 animal macro F1 排名；
6. 选择前两层，完全同分时优先较浅层；
7. Probe seed 为 `19000 + outer_fold`。

当前冻结规则产生的审计结果是：Fold 0 选择 5+8，Fold 1 选择 7+10，Fold 2 选择
10+12，Fold 3 选择 2+7。层号是规则的输出；未来重复实验仍执行同一规则。

### 3.4 Adapter、head 与训练

- 两个 block-output residual bottleneck adapters；
- 每个 adapter 为 `768→32→768`，GELU，残差相加；
- Up projection 零初始化，两个 adapters 共 99,904 个可训练参数；
- 加上统一 MLP head 后总可训练参数为 198,979；
- Head：标准化 → Dense(128) → ReLU → BatchNorm → Dropout → Dense(3)；
- Optimizer：Adamax；adapter LR `1e-3`，head LR `0.003109800273709165`；
- Micro-batch 8，gradient accumulation 4，gradient clip 1.0；
- 最大 50 epochs，inner validation loss patience 8；
- 每折选择 best epoch 后，在 outer train+validation 上按该 epoch 数重训；
- 模型 seed 为 `42 + outer_fold`，CUDA mixed precision 保持启用。

## 4. 最终比较中的模型角色

### Primary table

最终主对照只有一项：

`Probe-guided AST adapter − locked VGGish+MLP`

它回答 IDEA-048 的论文主问题：升级后的完整 pipeline 是否提高猫年龄三分类性能。

### Required ablations

1. **Head-only AST**：量化相同 AST/head recipe 下 adapter 带来的直接增量；
2. **Distributed 3+10**：说明同参数预算下跨深度 placement 的表现；
3. **Random 2+10**：检验 probe guidance 相对固定随机 placement 的额外贡献。

Early、Middle、Late、full fine-tuning、last-2 fine-tuning、IDEA-013 和 IDEA-003 的
已有结果继续作为完整实验轨迹和支撑分析保留，不进入最终候选搜索。

## 5. 路线状态

| 阶段 | 状态 | 产物 |
|---|---|---|
| Stage A：baseline 与协议锁定 | 已完成 | manifest、checksum、cat-ID folds、VGGish baseline |
| Stage B：方法候选开发 | 已完成 | IDEA-012、fine-tuning、013、003、019 |
| Stage C：最终 candidate freeze | **本次完成** | `meowagenet-idea048-candidate-freeze-v1` |
| Stage D：IDEA-048 最终比较 | 下一步 | 冻结候选 vs 唯一 VGGish baseline |

后续重复实验属于额外稳定性证据。它们沿用本次 v1 配置并使用新的 run ID；结果无论
相同或有波动，都追加记录，不回写改变本次冻结决策。

## 6. 冻结依据与追踪文件

- Locked protocol：`configs/protocol/meowagenet_locked_v1.json`
- Candidate freeze：`configs/protocol/meowagenet_idea048_candidate_freeze_v1.json`
- Locked baseline results：`metadata/experiments/meowagenet_locked_v1_results.json`
- IDEA-019 results：`metadata/experiments/meowagenet_idea019_peft_placement_v1_results.json`
- IDEA-019 report：`reports/07_IDEA-019_PEFT_placement_results.md`
- Candidate implementation：`scripts/run_idea019_peft_placement.py`
