# MeowAgeNet–AST Idea 空间与 Agent 工作流

## 1. 文件用途

本文件定义一套可复用的研究 idea 空间。它用于让不了解既往聊天记录的团队成员或
Agent，在相同项目边界下独立产生候选、核验依据、形成可证伪假设，并在获得执行指令后
生成实验计划和运行结果。

这里记录的是候选生成方法，不预设某个模块必然有效。每个输出需要标记为
`idea`、`assumption`、`prediction`、`located evidence` 或 `decision`。

## 2. 固定研究边界

- 研究对象：公开 MeowAgeNet 家猫叫声年龄分类数据。
- 原始 baseline：VGGish embedding + MLP。
- 当前强参考：tuned frozen AST + classification head。
- 研究重心：计算机方法、训练与评价，不扩展到重新采集大型动物声学数据。
- 划分单位：animal ID；同一只猫及其增强版本只能属于同一数据角色。
- 主要评价单位：cat/animal level，而不是单条 call。
- 主指标：animal-level Macro F1。
- 支持指标：Balanced Accuracy、QWK、accuracy、各类别 precision/recall、confusion matrix。
- 资源边界：以单张消费级 GPU 可复现为目标；大型 foundation-model 预训练不在当前范围。
- 方法比较：共享 split、数据角色、checkpoint selection 规则和评价代码；模型原生输入要求
  可以不同，但必须披露，不能把 preprocessing 差异写成纯 backbone 差异。

## 3. 当前 located evidence

以下内容来自现有项目报告，用于防止新 Agent 重复提出已经完成的同一实现：

- AST head-only 相对 VGGish+MLP 有明确正增益，是当前主要性能来源。
- Probe-guided adapter 相对 AST head-only 的增益较小并具有 split/seed 依赖性。
- 已测试的 Q/V LoRA 没有超过 matched tuned AST head-only。
- SSAST、PaSST、PANNs CNN14 和 AVES 的首轮 frozen-backbone screening 均低于 matched AST。
- 已测试的 time-fine patch geometry、temporal pooling、ordinal learning、全 12 层 scalar
  fusion 和 strict global cat balancing 没有形成稳定总体提升。
- AST 与 VGGish/CNN 保留了一部分 exclusive-correct animals，说明模型间可能存在互补信息。
- 数据包含 792 条 call、111 只猫，每只猫的 call 数量不均衡；当前最终预测通过对同一只猫
  的 call probabilities 取平均得到。

这些结果关闭的是具体实现，不是整个研究角度。重新开放某个角度时，Idea Card 必须说明
新机制与旧实现的差别，以及旧负面结果为何没有直接证伪新候选。

建议新 Agent 首先阅读：

- `reports/12_formal_v2_1_core_results.md`
- `reports/18_AST_head_and_adapter_hyperparameter_search.md`
- `reports/20_IDEA-050_AST_LoRA_initial_results.md`
- `reports/21_AST_cat_balancing_and_multilayer_fusion_results.md`
- `reports/22_AST_cat_balancing_seed_expansion_results.md`
- `reports/23_AST_cat_balance_global_weighting_results.md`
- `plan/AST_accuracy_enhancement_candidates.md`

## 4. Idea 空间：十个独立观察角度

同一个候选可以跨越两个角度，但必须指定一个主要角度。首轮发散时，每个角度独立生成，
避免所有想法都退化为“更换 backbone”或“再加一个 attention 模块”。

| 角度 | 核心问题 | 可改变的对象 | 需要观察的结果 |
| --- | --- | --- | --- |
| A. Task 与 label | 年龄类别是否得到最合适的数学表达？ | categorical、ordinal、continuous、hierarchical、multi-task、label uncertainty | 类别混淆、年龄距离错误、标签噪声敏感性 |
| B. Prediction unit | 训练单位和最终预测单位是否一致？ | call-level、animal-level、multi-instance/set learning、不同 call 的可靠性 | animal-level F1、不同 bag size 下的一致性 |
| C. Data contribution | 哪些动物、类别和样本主导梯度？ | sampling、loss weighting、hard-example policy、class/animal balance | 各类 recall、每猫表现、seed 稳定性 |
| D. Input representation | 输入是否保留任务所需的时间—频率信息？ | waveform、log-Mel、窗口长度、padding、patch、multi-resolution、learnable frontend | 按时长和频带分组的误差、局部线索保留情况 |
| E. Feature extraction | backbone 缺少什么表示，或已有表示如何被利用？ | local/global branch、CNN/Transformer composition、token/feature pooling、selected layers | residual information、层间互补、参数匹配后的增益 |
| F. Parameter adaptation | 哪些参数值得在小数据上更新？ | head-only、normalization/bias、scale-shift、adapter、LoRA、last blocks、full tuning | train–validation gap、跨 seed 方向、trainable parameters |
| G. Learning objective | loss 是否直接奖励目标指标和结构？ | CE、cost-sensitive、contrastive、metric/prototype、consistency、distillation | Macro F1、QWK、类间边界、embedding geometry |
| H. Knowledge combination | 不同模型或先验是否包含互补信息？ | feature/probability fusion、teacher–student、ensemble、acoustic descriptors | exclusive-correct animals、融合增益、学生模型保真度 |
| I. Optimization 与 inference | 性能损失是否来自训练轨迹或决策过程？ | optimizer、schedule、regularization、EMA/SWA、checkpoint averaging、calibration | 方差、最低性能、置信度、class-wise trade-off |
| J. Evaluation 与 robustness | 当前提升在哪些条件下成立？ | duration、call count、class、recording condition、compute budget、parameter count | subgroup performance、CI、failure cases、efficiency |

其中 A–I 可以产生方法候选；J 既可以检验其他候选，也可以形成 evaluation contribution。
单纯增加训练次数、参数量或搜索范围属于工程操作，需要与某个可检验机制结合后才能成为
研究 idea。

## 5. 从角度生成候选的组合规则

每个候选由下面五个槽位组成：

```text
Observed bottleneck
× intervention location
× proposed mechanism
× discriminating comparison
× expected measurable change
```

### 5.1 Observed bottleneck

只能从数据或现有结果中选择可以检查的瓶颈，例如：

- 小样本造成参数更新不稳定；
- 多 call 训练与 animal-level 评价存在单位差异；
- 短音频中的局部线索可能被 pooling 或 patching 稀释；
- 年龄类别具有顺序，但错误距离没有被充分建模；
- 类别或动物的样本贡献不均衡；
- 两个模型在不同 animals 上正确；
- checkpoint、seed 或 split 方差较大；
- 置信度与实际正确率不一致。

Agent 需要先给出诊断方法。缺少诊断证据时，将瓶颈标记为 `assumption`。

### 5.2 Intervention location

选择一个主要作用位置：

```text
waveform → spectrogram → patch/token → AST block/layer
→ pooled call embedding → classification head/loss
→ multi-call animal aggregation → calibrated decision
```

一个首轮候选默认只改变一个主要位置。跨位置组合在单模块产生稳定信号后再进入实验。

### 5.3 Proposed mechanism

说明候选通过什么因果或统计机制改善结果。以下陈述信息不足：

- “增加模块提高性能”；
- “使用更先进的模型”；
- “融合更多特征”；
- “调整超参数”。

合格机制需要说明：加入、保留、抑制或重新分配了什么信息；这个变化为何可能影响
unseen-cat 的预测。

### 5.4 Discriminating comparison

每个 idea 至少指定一个能区分竞争解释的对照：

- 参数量匹配的 wider head；
- feature-only 与 AST-only；
- 同参数量、不同作用位置；
- learned mechanism 与简单 mean/concatenation；
- 新 loss 与同结构 CE；
- 新 aggregation 与当前 arithmetic mean；
- 新 adaptation 与 matched head-only。

### 5.5 Expected measurable change

预测必须落到可观测量，例如：

- complete-OOF animal Macro F1 的配对变化；
- kitten/adult/senior recall 的方向；
- QWK 是否改善年龄顺序错误；
- 多 seed 方差或最差 repeat 是否改善；
- 不同 call 数、时长或 padding 分组之间的性能差是否缩小；
- 相同性能下的 trainable parameters、显存或训练时间是否下降。

## 6. 标准 Idea Card

每个新候选单独保存为 `plan/IDEA-XXX_<short_name>.md`，使用以下结构：

```markdown
# IDEA-XXX | 简短名称

## Provenance
- Origin: human / AI-assisted / literature-inspired / mixed
- Stage: independent / post-evidence / revised
- Date:

## Primary angle
- A–J 中的一个主要角度：
- 可选的一个支持角度：

## Located evidence
- 本地结果：
- 文献依据：
- 证据边界：

## Observed bottleneck
- 已观察事实：
- 尚待验证的 assumption：

## Idea claim
一句话说明候选方法、作用位置和预期解决的问题。

## Proposed mechanism
说明信息或梯度如何改变，以及为什么可能改善 unseen-cat 预测。

## Predictions
1. 主预测：
2. 支持预测：
3. 如果机制成立，应出现的 subgroup/error 变化：

## Rival explanations
1. 收益只来自参数量增加：
2. 收益只来自搜索或随机性：
3. 收益只来自 preprocessing 或类别先验：

## Disconfirming evidence
什么结果会使 idea 被判定为无效、信息不足或需要修改。

## Minimum discriminating experiment
- Matched reference：
- 唯一主要变量：
- 必要 ablations：
- Inner-only selection：
- Outer evaluation：
- 计算预算：

## Novelty location
- task / method / training / evaluation / efficiency：
- 与现有本地实现的差别：
- 最接近的 prior work：

## Status
- proposed / evidence-check / shortlisted / pilot / confirmed / closed
- Human decision owner：
```

## 7. 新 Agent 的标准工作流

### Phase 1：读取边界和事实

1. 阅读本文件与第3节列出的项目报告。
2. 建立三栏记录：`established`、`uncertain`、`closed implementation`。
3. 旧实验结果只能作为 located evidence；不得根据单个最高分推断稳定收益。

### Phase 2：独立发散

1. 在查看新增 AI 候选前，按 A–J 每个角度生成至少一个候选。
2. 每个候选只写一句 Idea claim 和一个可证伪结果。
3. 保留互相冲突的候选，不立即评分或合并。
4. 对与旧实验相近的候选，明确实现差异。

### Phase 3：结构化整理

1. 按主要角度、作用位置和预期机制聚类。
2. 合并重复措辞，保留机制、假设或对照不同的候选。
3. 为每个候选填写标准 Idea Card。

### Phase 4：局部证据核验

1. 每个候选单独建立 evidence sheet，不压缩全部论文笔记。
2. 使用现有单篇精读笔记和 paper-lookup 查找最接近工作、正面与负面结果。
3. 记录检索式、日期、论文标识和检索边界。
4. 文献中未找到直接相同工作时，写“在本次检索范围内未定位”，不写绝对首创。

### Phase 5：形成可证伪研究题目

对进入 shortlist 的候选使用 hypothesis-generation，形成：

- 明确的主假设；
- 至少一个 rival hypothesis；
- 区分两者的预测；
- 最小实验；
- measurement 与统计分析；
- 提前声明的停止或继续条件。

### Phase 6：adversarial review 与人工决策

审查下面的问题：

- 同样的正结果还能由什么原因产生？
- 是否存在 animal leakage、augmentation contamination 或 outer-test tuning？
- 新模块是否只增加了参数或训练次数？
- 输入窗口、模型原生 preprocessing 和 checkpoint selection 是否公平？
- 结果为零时是否仍能回答一个清楚的问题？

Agent 提供候选及审查记录；团队成员或导师决定进入实验的 idea。

### Phase 7：执行所选 idea

获得明确执行指令后：

1. 先写 `plan/` 文件，再写 protocol、配置和 runner。
2. 复用 animal-ID-disjoint split bank，并验证数据 hash。
3. 超参数与 module selection 只访问对应 outer-training role 内的 inner split。
4. 首轮保持一个主要变量，运行 matched reference 与必要 ablations。
5. 生成 complete OOF animal predictions，覆盖全部111只猫。
6. 报告 Macro F1、Balanced Accuracy、QWK、accuracy、分类别结果和 paired changes。
7. 首轮出现信号后再扩展 seeds；确认阶段使用预先锁定的 gate。
8. 将可读结论保存到 `reports/`，机器结果保存到 `metadata/experiments/`，运行审计保存到
   `runs/`。raw audio、checkpoints 和 per-sample predictions 遵守仓库数据策略。

探索阶段可以连续测试候选；确认阶段使用未根据结果修改的锁定 protocol。两阶段的结果和
用途需要分别标记。

## 8. 候选筛选维度

评分只能帮助排序，最终选择由团队或导师完成。建议分别记录分数、理由和置信度：

| 维度 | 高分含义 |
| --- | --- |
| Problem fit | 直接对应本地已观察到的问题 |
| Information gain | 正、负结果都能区分竞争解释 |
| Method contribution | 具有明确机制和可复用的方法表达 |
| Expected performance value | 有合理机会提高 Macro F1 或支持指标 |
| Feasibility | 单卡、公开数据和毕业设计周期内可执行 |
| Rigor | 能建立 matched control 并避免 leakage |
| Novelty evidence | 经限定文献检索后仍有清楚差异 |
| Null-result value | 无增益时仍能形成有边界的结论 |

不得用加权总分覆盖以下 gate：数据不可得、评价泄漏、算力不可承受、缺少有效对照或研究
问题无法被当前数据测量。

## 9. 可复制给新 Agent 的启动提示词

```text
你正在处理 MeowAgeNet–AST 家猫年龄分类项目。仓库根目录包含 plan、reports、configs、
scripts、splits 和 metadata。首先完整阅读
plan/AST_idea_space_and_agent_workflow.md，以及其中第3节指定的现有结果报告。

任务模式：[IDEATE / PLAN / RUN，用户填写]

IDEATE：沿 A–J 十个角度独立生成候选。每个候选标明 located evidence、assumption、
Idea claim、mechanism、prediction、rival explanation、disconfirming evidence 和最小对照。
保留候选之间的冲突，不自动选择最终题目。避免重复已经完成的同一实现。

PLAN：只处理用户指定的 IDEA。使用 hypothesis-generation 将其转化为可证伪假设，完成
局部文献核验、adversarial review、matched controls、ablation、指标和停止条件，并写入
plan/IDEA-XXX_<short_name>.md。计划阶段不读取 outer-test 结果来选择方法。

RUN：仅在用户已经明确选定 IDEA 并授权执行时开始。先冻结 protocol 和配置，复用
animal-ID-disjoint splits，保留 matched tuned AST reference，随后运行 smoke、inner-only
selection、complete-OOF evaluation 和必要的 seed 扩展。结果分别写入 reports、metadata
和 runs，完整报告负面结果与协议偏差。

所有结论区分 source statement、located evidence、inference 和 speculation。最终方法选择
由团队成员或导师决定。
```

## 10. 当前覆盖图的使用方式

新的 idea 登记后，在本节追加一行，记录角度和实现状态。这样可以看出项目在哪些角度投入
过多、哪些角度仍缺少候选，同时避免把某个具体负结果误写成整个方向已经无效。

| 已有工作 | 主要角度 | 当前证据状态 |
| --- | --- | --- |
| VGGish+MLP → AST head-only | E/F | 稳定正结果，当前强参考 |
| Probe-guided adapter | F/E | 相对 head-only 增益较弱 |
| Q/V LoRA | F | 当前实现低于 matched head-only |
| SSAST、PaSST、PANNs、AVES screening | E | 首轮候选均低于 AST |
| Time-fine patch geometry | D | 当前实现无总体提升 |
| Temporal pooling | E/B | 当前实现无总体提升 |
| Ordinal learning | A/G | 当前实现无总体提升 |
| 12-layer scalar fusion | E | 当前实现下降，权重接近均匀 |
| Cat-balanced loss | C/G | 修正复验缺少改进证据 |
| Cat-level set aggregation | B/E/C | hidden-mean 与 attention set 低于 matched call-probability mean |
| AST local acoustic residual | D/E | temporal mean 三次平均与 reference 接近且有 2/3 正向 repeats；salience residual 较低 |
| AST–VGGish probability fusion | E/F | 存在 21 次 VGGish-only 正确；inner-selected scalar fusion 平均低于 AST 0.0071 |
| LayerNorm / SSF / BitFit | F/E | 三种受约束校准均低于 frozen AST；LayerNorm 最接近且有 1/3 正向 repeat |
| Checkpoint ensemble / class-bias calibration | I/J | tail-3 平均与 A0 主指标接近，并改善 balanced accuracy 与 animal CE；类别 bias 的主指标较低 |

## 11. 当前优先探索导航

以下排序是团队在现阶段给出的 working priority，属于 `decision`，用于给后续 Agent 分配
探索顺序。它不表示相应方法已经得到验证，也不阻止文献核验或诊断结果改变排序。

| 优先级 | 方向 | 更适合解决的问题 | 大致思路 | 在研究中的角色 |
| ---: | --- | --- | --- | --- |
| 1 | Cat-level set aggregation | call-level 建模与最终 cat-level 预测之间的单位差异，以及每只猫拥有不同数量 call 的数据结构 | 将同一只猫的多条 call 视为一个整体，让模型学习多条观测之间的信息、差异和可靠性，再输出猫级年龄预测 | 问题定义最清楚，能够形成围绕 animal-level learning 的主要方法贡献 |
| 2 | AST + local acoustic residual branch | AST 全局表示可能弱化亚秒叫声中的局部时间—频率年龄线索 | 保留 pretrained AST 主路径，再用轻量分支补充局部声学信息，并在分类前进行受控融合 | 直接改造 AST，兼顾性能提升、结构消融和方法创新 |
| 3 | AST–VGGish fusion / distillation | AST 与 VGGish/CNN 在部分 animals 上存在不同的正确与错误模式 | 先检验模型输出或表示是否真正互补；若互补信号稳定，再研究融合或教师—学生知识转移 | 较适合快速寻找额外性能；单纯融合偏工程，稳定 distillation 可以增强方法表达 |
| 4 | LayerNorm / SSF / BitFit | adapter 和 LoRA 可能具有不合适的更新位置或有效容量，小数据更需要受约束的表示校准 | 只更新 normalization、feature scale/shift 或 bias 等少量参数，观察轻量校准能否比现有 PEFT 更稳定 | 补充 PEFT 证据，帮助解释“更新多少参数、更新哪里”比具体方法名称更重要 |
| 5 | Checkpoint averaging / calibration | 训练轨迹波动、单 checkpoint 偶然性和类别决策偏差可能影响最终指标 | 在模型主体确定后，对相邻 checkpoint 或输出概率进行受控整合与校准 | 最终性能优化和支持实验，适合提高稳定性、置信度或特定类别表现 |

### 11.1 排序的解释

第一和第二方向优先处理模型与任务结构之间的关系，最有机会形成清楚的研究问题。第三方向
优先利用现有模型的互补错误，通常更接近性能增强路线。第四方向延续现有 PEFT 研究，但把
重点从继续增加可训练模块转向更新位置和容量控制。第五方向作用于训练结果与最终决策，适合
在主模型确定后作为收尾实验。

### 11.2 给后续 Agent 的边界

- 首轮分别处理五个方向，不把多个方向同时组合成一个大模型。
- 每个方向先完成问题诊断和 Idea Card，再提出具体结构或运行方案。
- 排名表示当前探索顺序，不构成预期性能排名。
- 新证据可以改变优先级；调整时记录证据、理由、日期和决策人。

### 11.3 执行进度（2026-09-14）

- 优先级 1 已完成：IDEA-051 的 hidden-mean set 与 attention set 均低于 matched
  call-probability mean；animal-level checkpoint selection 保留为轻量正信号。
- 优先级 2 已完成：IDEA-052 的 temporal-mean residual 在 2/3 repeats 提高，平均
  macro-F1 差为 `−0.0009`；temporal-salience residual 平均差为 `−0.0139`。两条
  seed-expansion gate 均关闭。
- 优先级 3 的直接概率融合已完成：333 次配对评价中包含 21 次 VGGish-only 正确，说明
  互补空间存在；IDEA-053 的每折 inner-selected scalar fusion 平均 macro F1 为 0.7499，
  相对 AST 为 `−0.0071`，三个 repeats 中 1 个提高，seed-expansion gate 关闭。
- 优先级 4 已完成：IDEA-054 的 LayerNorm、block-output SSF、BitFit 平均 macro F1 分别为
  0.7429、0.7356、0.7372，低于 matched frozen AST 的 0.7570；三条 seed-expansion gate
  均关闭。LayerNorm 在 1/3 repeats 提高，保留为局部 PEFT 信号。
- 优先级 5 已完成：IDEA-055 的 tail-3 checkpoint 概率平均取得 0.7552 macro F1，
  相对 matched A0 的 0.7570 为 `−0.0018`；balanced accuracy 提高 0.0020，animal CE
  降低 0.0100。类别 bias 与 ensemble+bias 分别取得 0.7513 和 0.7457。三条
  seed-expansion gate 均关闭，P1 保留为概率质量与类别均衡方面的支持性结果。
- 当前五项优先探索形成阶段性收尾，A0 继续作为性能参考。该节点用于整理现有证据，
  不锁定最终模型，也不限制后续新 idea；feature-level、条件式融合和新的 AST 改进仍可
  按独立 Idea Card 进入下一轮。

### 11.4 五项路线完成后的确认决策（2026-09-14）

- IDEA-051 至 IDEA-055 的候选方法暂不扩展 base seeds 43/101。
- 新 A0 使用 animal-level validation cross-entropy 选择训练轮次，seed-17 相对旧 C0 的
  观察差为 `+0.0106`、2/3 repeats 为正。该结果另立为 IDEA-056 确认计划。
- IDEA-056 的主要判断只使用尚未查看的 seeds 43/101；seed 17 保留为历史探索性证据。
- IDEA-056 完成后再进入新的 AST architecture idea，避免把训练流程确认和模型结构改造
  混为同一个实验。

后续固定阶段顺序为：完成 IDEA-056 并确定 AST 参考流程；诊断局部 patch、中间层和预训练
领域差异；依据诊断只选择一个 AST 内部方向；完成单模块、参数量匹配和原始 AST 对照；
出现稳定正信号后再进入模块组合。
