# AST 内部诊断与单模块验证计划

> 状态：active planning  
> 日期：2026-09-14  
> 前置阶段：IDEA-056 已完成并通过项目预设确认规则  
> 当前任务：先定位 AST 表示中的问题，再选择一个内部改进方向

## 1. 阶段决定

IDEA-056 已完成。后续实验统一采用当前 tuned frozen-AST reference：冻结 AST backbone，
使用既有分类头和 animal-ID-disjoint 数据划分，以 animal-level validation cross-entropy
选择 checkpoint。旧的 call-level checkpoint 规则保留为 IDEA-056 的匹配消融，不再作为
新方法实验的默认参考。

本阶段依次完成四件事：

1. 诊断 AST 对局部 patch、中间层表示和猫叫领域特征的利用情况；
2. 根据诊断证据只选择一个 AST 内部改进方向；
3. 将该方向实现为单一模块，与参数量匹配对照和原始 AST 参考流程比较；
4. 单模块获得稳定正信号后，再单独规划模块组合实验。

这是一份阶段计划。诊断尚未完成，因此当前不为新方法分配 IDEA 编号，也不预设 local
branch、layer fusion 或 domain adaptation 中的任何一个必然有效。

## 2. 研究目标

当前目标是回答：现有 AST 在家猫年龄分类中仍未充分利用的信息主要出现在哪里？候选解释
分为三个互相竞争、也可能部分共存的方向：

| 诊断轴 | 待检查的问题 | 支持该解释的观测形式 |
| --- | --- | --- |
| 局部 patch | AST 的全局输出是否压缩了短时、局部时间—频率年龄线索 | 局部 token 或受控 patch 区域在 final global embedding 之外提供可重复的类别信息；遮蔽相应区域会稳定损害猫级预测 |
| 中间层 | 年龄相关信息是否在某些 Transformer block 中较强，随后在最终层被减弱或改写 | 个别中间层在相同 probe 下稳定优于最后层，或对最后层错误提供可重复的互补信息 |
| 预训练领域差异 | AudioSet 预训练目标与亚秒猫叫年龄任务之间是否存在表示偏差 | 使用训练折内猫叫音频进行受控的领域适配后，表示可分性或 inner-validation 指标稳定改善，而局部/层级读出不足以解释该改善 |

诊断目标是缩小机制范围。单个相关性、单折最高分或一张 attention 图只形成线索，需要配对
结果或第二种诊断证据支持后，才能用于方向选择。

## 3. 已有证据与本轮边界

以下内容直接来自仓库中的既有实验：

- IDEA-056 的 animal-level checkpoint 规则在 6 组新配对中平均提高 Macro F1
  `+0.0059`，4/6 为正，已成为后续参考流程；
- 12-layer scalar fusion 低于 final-layer reference，训练后的层权重接近均匀，因此本轮不
  重复“全部层直接加权平均”的相同实现；
- IDEA-052 的 temporal mean residual 在 2/3 repeats 提高，但平均差为 `-0.0009`；
  temporal salience residual 平均差为 `-0.0139`。局部信息仍值得诊断，原有逐维 gate
  实现已经完成筛选；
- AVES-base-bio 是 IDEA-049 中最好的新增 frozen backbone，但 Macro F1 仍比 matched
  AST 低约 `0.0607`。该结果说明“动物声预训练”标签本身不能证明领域模型更适合本任务，
  领域差异需要在同一 AST 路径内进一步检验。

本阶段只使用公开 MeowAgeNet 数据和现有 animal-ID 独立划分。所有会影响方法选择的诊断
先在 inner-training / inner-validation 范围完成，outer-test 不用于挑选层、patch、模块或
超参数。

## 4. 第一阶段：AST 内部诊断

### 4.1 共同设置

- 以 IDEA-056 确定的 AST reference 作为共同起点；
- 固定音频预处理、数据划分、猫级概率聚合和主要指标定义；
- 诊断结果按 repeat、fold、年龄类别和 call 时长保留，避免只看总平均；
- 优先复用 frozen features，控制计算量；需要训练的诊断只访问当前训练角色中的数据；
- 每项诊断记录 source statement、observed evidence、inference 和 unresolved uncertainty。

### 4.2 局部 patch 诊断

目标是判断局部时间—频率区域是否包含 global embedding 未稳定保留的信息。允许采用的诊断
包括局部 token probe、受控 time/frequency patch masking、区域汇总后的互补性分析，以及
错误样本对局部区域的敏感性分析。

需要回答：

1. 局部信息的增益是否跨多个 animal-ID splits 出现；
2. 信号是否集中于某类时长、频带或年龄类别；
3. 观测能否解释 IDEA-052 中 temporal mean residual 的 2/3 正向与一次较大下降；
4. 结果是否主要由能量、静音比例、padding 或切分边界等简单因素驱动。

### 4.3 中间层诊断

目标是区分“某些层包含额外年龄信息”与“简单混合所有层破坏最终表示”。优先使用共享设置的
layer-wise probe、少量候选层的互补错误分析和 final-layer conditional gain。现有结果已提示
第 8、10、11、12 层值得优先检查，但候选层仍由 inner-only 诊断决定。

需要回答：

1. 某一层或小范围层组是否跨 splits 保持优势；
2. 中间层能否修正 final layer 的一组可重复错误；
3. 优势是否来自新增信息，或只是 probe 参数量、归一化与训练难度差异；
4. 能否形成比全 12 层 scalar fusion 更有针对性的机制。

### 4.4 预训练领域差异诊断

目标是检验 AST 的通用音频预训练表示是否需要面向猫叫作有限调整。可采用训练折内的无标签
猫叫预适配、受约束的上层表示更新或适配前后的表示可分性比较。具体训练方法在诊断实现前
登记，且所有预适配数据严格限制在当前训练角色内。

需要回答：

1. 领域调整的收益是否超过仅增加可训练参数带来的容量效应；
2. 收益是否跨 splits，并在 Macro F1、Balanced Accuracy 或类别 recall 中形成一致解释；
3. 改善是否来自减少设备、时长、音量等 nuisance variables 的影响；
4. 结果能否解释 AST 优于 AVES，同时仍可能从猫叫域适配中获益。

## 5. 方向选择规则

完成三条诊断轴后形成一份 diagnostic decision record，由团队或导师决定进入哪一个方向。
选择时分别记录以下依据，不把它们压缩成一个缺少解释的总分：

- 跨 animal-ID splits 的一致性；
- 对 AST 现有错误的解释能力；
- 与既有负面或混合结果的相容性；
- 预期对 Macro F1、Balanced Accuracy 和类别 recall 的影响；
- 单卡实验的可行性、附加参数量和复现难度；
- 参数量匹配对照是否能够清楚构造；
- 结果为零或负数时仍能回答的研究问题。

选择记录必须包含一个主方向、未选方向及其暂缓理由。若三条轴都缺少重复证据，本阶段结论
记为“诊断未支持 AST 内部扩展”，不为了增加实验数量强行指定模块。

## 6. 第二阶段：单模块验证

选定方向后，另建一个带新 IDEA 编号的 Idea Card，并在查看 outer-test 结果前写明假设、
竞争解释、实现、指标和继续条件。首轮核心比较保持为三条 pipeline：

| Pipeline | 作用 |
| --- | --- |
| R0：原始 AST reference | IDEA-056 确定的 tuned frozen-AST 流程，提供主要性能参考 |
| M1：单一诊断驱动模块 | 只加入所选机制，检验该机制能否提供增量 |
| C1：参数量匹配对照 | 使用相近数量的可训练参数或计算量，但不实现目标机制，用于区分机制收益与普通容量收益 |

“原始 AST”在本计划中指 AST backbone 与标准分类头的参考架构，并采用 IDEA-056 已确认的
训练轮次选择规则。这样可以将新模块增量与旧 checkpoint 选择规则的影响分开。

主要评价单位继续是 animal level，主要指标为 Macro F1；Balanced Accuracy、QWK、普通
accuracy、各类别 precision/recall、animal-level CE 和结果波动作为支持证据。参数量、训练
时间和显存占用同时报告。

## 7. 稳定正信号与模块组合

“稳定正信号”表示模块相对 R0 的改善在多个 animal-ID splits 和随机设置中重复出现，平均
效果达到新 Idea Card 预先声明的门槛，同时没有依赖单个类别或单次异常运行。具体数值门槛
应在模块和预计效应范围明确后冻结，本阶段计划不提前指定统一数字。

达到门槛后，模块进入独立确认，再讨论与已有模块组合。组合实验需要把单模块 M1 作为直接
对照，以检验组合带来的增量。未达到门槛时保留诊断和消融结果，返回尚未选择的诊断轴或
结束本轮，不在负信号上连续叠加模块。

## 8. 预期产物

1. AST 内部诊断报告及机器可读结果；
2. 三条诊断轴的 decision record，包括证据、推断、限制和团队选择；
3. 一个新的单模块 Idea Card 与冻结 protocol；
4. R0、M1、C1 的配对结果、参数量和资源报告；
5. 是否进入独立确认或模块组合的阶段决定。

计划文件保存在 `plan/`。完成后的实验结果进入 `reports/`，机器可读结果进入 `metadata/`，
运行审计进入 `runs/`。诊断和结果不会反向改写本计划。

## 9. 术语说明

| 术语 | 含义 |
| --- | --- |
| local patch（局部块） | AST 将 log-Mel spectrogram 划分成的小型时间—频率区域。局部 patch 诊断检查短时或特定频带的信息是否在全局汇总过程中被弱化。 |
| intermediate layer（中间层） | AST 最终输出之前的 Transformer block 表示。不同层可能偏向局部声学形态、组合模式或预训练任务中的高层特征。 |
| pretraining domain mismatch（预训练领域差异） | AST 主要从通用音频学习表示，而当前任务是短时家猫叫声年龄分类；两种数据分布和目标之间的差异可能限制直接迁移。 |
| probe（探针模型） | 在冻结表示上训练的简单分类器，用来测量某层或某类特征包含多少可用于当前标签的信息。probe 的高分属于诊断证据，不等同于新模块已经有效。 |
| nuisance variable（干扰变量） | 与年龄标签可能偶然相关、但不属于目标机制的因素，例如录音设备、音量、时长和背景噪声。模型依赖这些因素时，跨猫泛化可能下降。 |
| parameter-matched control（参数量匹配对照） | 与新模块具有相近可训练参数量或计算量、但缺少目标机制的对照，用来判断提升来自机制设计还是单纯增加容量。 |
| single-module test（单模块验证） | 一次只增加一个结构或训练机制，使性能变化能够归因于该模块。 |
| stable positive signal（稳定正信号） | 项目中的决策用语，指改进跨多个独立划分和随机设置重复出现，并达到预先写明的继续条件。它不是深度学习中的固定算法名称。 |

## 10. 决策日志

| 日期 | 类型 | 决定 | 依据 |
| --- | --- | --- | --- |
| 2026-09-14 | located evidence | IDEA-056 完成，animal-level CE checkpoint selection 成为后续 AST reference | 6 组新配对平均 Macro F1 `+0.0059`，4/6 为正；同时保留 Balanced Accuracy 的权衡 |
| 2026-09-14 | decision | 下一阶段先完成 AST 内部诊断，只选择一个方向进入单模块验证 | 团队确认的执行顺序；避免同时叠加模块导致归因困难 |
| 待填写 | decision | 局部 patch / 中间层 / 预训练领域差异中选择一个方向 | 由诊断报告和团队或导师判断填写 |

