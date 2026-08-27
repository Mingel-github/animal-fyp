# IDEA-048 阶段性收尾记录

> 日期：2026-08-27
> Checkpoint ID：`meowagenet-idea048-stage-checkpoint-v1`
> 状态：**当前实验阶段已收尾；后续候选与新 idea 保持开放**

## 1. 阶段性决定

IDEA-048 当前阶段已经取得清晰的性能提升结果。Probe-guided AST adapter 以
animal-level macro F1 **0.7575** 成为目前表现最好的方案，保留为：

> **current reference candidate｜当前参考候选**

这是一项阶段性结论，不是毕业论文最终模型锁定。后续可以继续开展重复实验、提出
新的 idea、开发新的 pipeline，或者由更好的候选替代 Probe-guided AST。

项目当前稳定保留：

- 唯一正式 baseline：锁定 VGGish + MLP；
- 当前参考候选：Probe-guided AST adapter；
- 必要解释性消融：Head-only、Distributed、Random；
- 已完成的 IDEA-012、fine-tuning、IDEA-013、003、019 实验轨迹。

## 2. 当前阶段的性能总结

| Pipeline | Animal macro F1 | Balanced accuracy | QWK | 当前角色 |
|---|---:|---:|---:|---|
| VGGish + MLP | 0.6846 | 0.6754 | 0.5263 | 唯一正式 baseline |
| Head-only frozen AST | 0.7260 | 0.7696 | **0.6791** | Adapter 净效应对照 |
| Distributed adapter | 0.7501 | 0.7769 | 0.6616 | Placement ablation |
| Random adapter | 0.7545 | 0.7556 | 0.6419 | Probe-guidance ablation |
| **Probe-guided AST adapter** | **0.7575** | **0.7795** | 0.6755 | **当前参考候选** |

当前参考候选相对 VGGish baseline 的 macro F1 提高 **0.0729**，相对同一
PyTorch AST recipe 的 Head-only control 提高 **0.0315**。因此，本阶段可以明确记录：

> 当前开发得到的低参数 AST adaptation pipeline 已经超过复现的 VGGish baseline，
> IDEA-048 获得了有实际幅度的性能提升信号。

Random 与 Probe-guided 仅相差 0.0030，说明低参数跨层适配是当前较稳定的性能信息；
probe-guided selection 取得最高点估计，其独特机制优势继续由消融结果限定解释。

## 3. 哪些内容继续稳定

为了让以后新增实验仍然能够公平比较，以下基础保持为当前 `v1` 研究基准：

- 数据 manifest 与 checksum；
- 792 段唯一 calls、111 只分析猫和三年龄类别；
- 四个 cat-ID-disjoint outer folds 与 nested roles；
- animal-level probability aggregation；
- primary metric：animal-level macro F1；
- practical threshold：`δ_task = 0.03`；
- 唯一 VGGish + MLP baseline；
- 当前所有报告、预测和机器元数据作为历史证据保留。

这些稳定项让未来的新 idea 可以直接回答：它相对 VGGish 和当前 Probe-guided
reference 分别提高了多少。

## 4. 哪些内容保持开放

以下研究内容不在本次 checkpoint 中定死：

- 最终论文采用哪一个模型；
- Probe-guided AST 是否长期保持最佳；
- 是否引入新的 pretrained audio encoder；
- adapter、LoRA 或其他 PEFT 设计；
- 新的 pooling、loss、augmentation 或多尺度输入；
- 新的任务建模与组合方案；
- 后续重复次数、seed 设计与最终稳定性实验；
- 最终论文主表和最终模型命名。

只要新方案沿用可比的数据和 animal-level 评估，便可以进入下一阶段。确实需要改变
基础协议时，建立新的 protocol version，并同时保留 v1 结果作为历史参照。

## 5. 当前参考候选的可复现快照

为了便于未来比较，本阶段仍记录 Probe-guided AST 的完整 recipe，但这份 recipe 是
**参考快照**，不是不可替代的最终方案：

- MIT AudioSet standard AST checkpoint，固定 revision `f826b80d...`；
- 16 kHz，1.28 秒窗口、0.64 秒 hop；
- Standard AST geometry，frequency/time stride 均为 10；
- 每个 outer fold 只用 inner-train cats 完成 12 层 probe；
- 三折 cat-level CV，以 animal macro F1 选择前两层；
- 两个 width-32 residual bottleneck adapters；
- 统一 768→128→3 MLP head；
- Adamax，adapter LR `1e-3`，head LR `0.003109800273709165`；
- 最大 50 epochs，inner validation loss patience 8；
- 模型 seed 为 `42 + outer_fold`。

四折选层审计结果继续记录为：5+8、7+10、10+12、2+7。

## 6. 路线状态修订

| 阶段 | 状态 | 含义 |
|---|---|---|
| Baseline 与协议建立 | 已完成 | 数据、fold、VGGish baseline 可追溯 |
| 第一阶段候选开发 | 已完成 | AST、012、fine-tuning、013、003、019 已评估 |
| IDEA-048 阶段性 checkpoint | **本次完成** | 保存当前成果与当前最佳候选 |
| 后续候选开发与重复实验 | 保持开放 | 有新 idea 时继续，不受本次候选限制 |
| 论文最终 candidate freeze | 延后 | 在模型路线和论文范围成熟后执行 |
| 论文最终比较 | 延后 | 使用届时冻结的最终候选完成 |

因此，原先“Stage D 最终比较是紧接着的唯一下一步”修订为：

> **当前阶段先完整收尾。下一阶段既可以做重复实验，也可以开发新的改进 idea；最终
> candidate freeze 和最终比较留到论文方案成熟时执行。**

## 7. 后续实验如何接入

后续每个新方向采用新的 idea 或 run ID，并报告两类对照：

1. 新方案 vs 唯一 VGGish baseline：判断是否继续支持 IDEA-048；
2. 新方案 vs 当前 Probe-guided reference：判断是否形成新的最佳候选。

新的结果追加到项目记录，不覆盖本阶段结论。如果新方案超过 0.7575，current
reference candidate 可以自然更新；如果没有超过，Probe-guided 继续作为最强已知方案。

## 8. 追踪文件

- Stage checkpoint：`configs/protocol/meowagenet_idea048_stage_checkpoint_v1.json`
- Decision metadata：`metadata/experiments/meowagenet_idea048_stage_checkpoint_v1.json`
- Locked protocol：`configs/protocol/meowagenet_locked_v1.json`
- Locked baseline results：`metadata/experiments/meowagenet_locked_v1_results.json`
- IDEA-019 results：`metadata/experiments/meowagenet_idea019_peft_placement_v1_results.json`
- IDEA-019 report：`reports/07_IDEA-019_PEFT_placement_results.md`

本记录取代同日较早的“final candidate freeze”路线表述。较早提交仍保留在 Git 历史中，
用于透明记录决策如何从最终冻结调整为阶段性收尾。
