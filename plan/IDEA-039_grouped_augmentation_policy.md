# IDEA-039｜Animal-grouped augmentation policy

> 日期：2026-09-15
> 状态：当前活动计划；等待 executable protocol
> 决策人：团队成员或导师
> 目标：提高 tuned frozen AST 的 animal-level 家猫年龄分类表现，并检验 nested policy
> selection 是否比固定增强策略更可靠

## 1. 研究定位

IDEA-058 strict nested 复核后，tuned frozen AST 仍是当前最稳定的参考流程：animal Macro F1
为 `0.7570 ± 0.0086`。本路线保持 AST backbone、分类头、animal-ID splits、checkpoint
selection 和 cat-level aggregation 不变，只研究训练侧 input augmentation。

`Animal-grouped` 表示同一只猫的原始 calls、segments 和所有增强视图始终位于同一数据角色。
它是一条防止数据污染的实验边界，并不表示每只猫使用不同的增强算法。

本路线来自团队此前的人工 shortlist。团队曾尝试多种 augmentation，但旧实验记录不在当前
仓库，因此这些尝试只能记为 `team recollection`，不能作为方法有效或无效的证据。

## 2. 已知证据、假设和未知项

| 类型 | 内容 |
| --- | --- |
| Located evidence | IDEA-056/058 的 R0 在相同三个 split repeats 上达到 `0.7570 ± 0.0086` Macro F1 |
| Located evidence | 数据包含 111 只猫、792 条 calls 和 843 个 AST segments；主要评价单位是 cat |
| Located evidence | 严格 IDEA-058 的 12 个 fold 中有 8 个候选首位 Macro F1 平局，说明小 inner-validation set 难以稳定区分相近 recipe |
| Assumption | 温和增强可以保留年龄类别，同时降低模型对录音强度、局部遮挡或时间位置的敏感性 |
| Assumption | 多个 augmentation RNG streams 能降低策略排序被单次随机增强支配的风险 |
| Unknown | 最适合人类声音或通用音频的增强是否保留亚秒猫叫中的年龄线索 |
| Unknown | per-fold policy selection 是否优于一个预先固定的轻量策略 |
| Evidence boundary | 本计划没有新增系统文献检索；augmentation novelty 和外部先例仍待 paper-lookup 核验 |

## 3. 研究问题和可证伪假设

### 3.1 固定增强策略

**RQ039-A：** 在完全相同的 animal-ID 独立评价下，一个预先固定的轻量 augmentation policy
能否提高 tuned frozen AST 的 complete-OOF animal Macro F1？

**H039-A：** 训练侧温和增强减少小样本过拟合，使固定增强 pipeline 相对同期 no-augmentation
reference 获得跨 repeats 的 Macro F1 增益。

**反证结果：** 增益只出现在一个 repeat，平均 Macro F1 持平或下降，或者收益伴随 Senior
recall、QWK 或 animal CE 的明显退化。

### 3.2 Nested policy selection

**RQ039-B：** 每个 outer fold 内独立选择 augmentation policy，能否比预先固定策略取得更好的
outer-test 泛化？

**H039-B：** inner-validation 能辨别适合当前训练侧数据的增强类型，因此 nested-selected
policy 平均高于固定策略和 no-augmentation reference。

**反证结果：** 候选大量平局、不同 augmentation RNG streams 产生相反排序，或 selected policy
在 outer test 上没有超过固定策略。此时结论应指向 selector 不稳定，不能只替换候选继续搜索。

## 4. 竞争解释

| 编号 | Rival explanation | 可区分观察 |
| --- | --- | --- |
| R1 | 增益来自每条 call 被重复看见、optimizer steps 增加，而不是增强内容 | 等数量 identity/no-op views 取得相似收益 |
| R2 | 猫叫很短，time/frequency masking 删除了年龄线索 | mask 强度上升时 Macro F1、Senior recall 或 QWK 持续下降 |
| R3 | per-fold selector 在少量 validation cats 上选择噪声 | 候选频繁平局，跨 augmentation RNG 的排名一致性低，A2 不超过固定 A1 |
| R4 | online AST 与缓存 embedding 的数值或数据顺序差异产生表面变化 | online no-augmentation control 无法复现 frozen reference |
| R5 | 增强只改善某一类别或某一 call-count 区间 | 总体增益在类别或 cat call-count 分层后消失或反转 |

## 5. 候选与对照结构

### 5.1 Pipeline

| 编号 | Pipeline | 作用 |
| --- | --- | --- |
| R0 | Tuned frozen AST，no augmentation | 当前 matched reference |
| C0 | Online frozen AST 或增强缓存路径，使用 identity/no-op views | 检查执行路径和重复曝光效应 |
| A1 | 一个预先固定的轻量 policy | 检验 augmentation 本身能否稳定增益 |
| A2 | 每个 outer fold 内 nested-selected policy | 检验 policy selection 是否提供额外价值 |

C0 只在 augmentation 实现改变 encoder 执行路径、每 epoch view 数或 optimizer steps 时需要。
若 A1/A2 可以复用与 R0 完全相同的缓存和训练预算，protocol 可以省略 C0并记录等价性依据。

### 5.2 有限候选池

executable protocol 最多冻结四个候选，建议从以下家族选择：

1. identity/no augmentation；
2. 轻量 SpecAugment，只遮蔽较窄的 time/frequency 区域；
3. 温和 gain/noise，用于扰动录音强度和背景；
4. 轻量 time shift 或一个预先声明的组合策略。

Pitch shift 和幅度较大的 time stretch 可能直接改变音高、时长等年龄线索，当前不列为首轮候选。
具体强度需查看真实 call duration 和 fbank 尺度后冻结；outer-test 结果不能参与调节。

## 6. 数据边界和选择流程

1. 继续使用现有 3 repeats × 4 outer folds，independence unit 为 `cat_id`。
2. augmentation 仅作用于当前 outer fold 的训练侧。validation 和 outer test 始终使用原始输入。
3. 原始 call、切分 segments、增强视图及缓存条目继承同一 `cat_id` 和数据角色。
4. A2 在每个 `repeat × outer fold` 内独立选择 policy，不跨 folds 汇总候选得分。
5. 每个候选在 inner selection 中使用多个固定 augmentation RNG streams；记录各 stream 的
   Macro F1、animal CE 和候选排名。
6. 候选排序优先使用多个 RNG streams 的平均 inner animal Macro F1，平局时依次参考 animal
   CE、较弱增强和 lexical policy ID。最终规则在运行前写入 protocol。
7. per-fold policy、随机种子、增强参数和缓存 SHA-256 在读取 outer test 前进入 execution lock。
8. 选定 epoch 后，使用当前 outer train+validation cats 训练，再一次性评价当前 outer-test cats。

## 7. 评价和决策

Primary endpoint 为 complete-OOF animal Macro F1。Secondary endpoints 包括 Balanced
Accuracy、QWK、plain Accuracy、animal CE、逐类别 precision/recall/F1、预测改变数、运行时间
和峰值显存。

首轮使用 base seed 17 和三个 split repeats。建议将以下条件写入 executable protocol，作为
是否扩展 seeds 43/101 的 gate：

- A1 或 A2 相对同期 R0 的平均 Macro F1 增益至少 `+0.005`；
- 至少 `2/3` complete-OOF repeats 为正；
- QWK、Balanced Accuracy、animal CE 和任一类别 recall 的代价保持在预先声明范围；
- online/no-op C0 与 R0 达到预先声明的数值或预测一致性；
- A2 的候选排序没有被单一 augmentation RNG stream 支配。

H039-A 和 H039-B 分别判断。A1 提高而 A2 没有提高，支持固定增强并反对当前 selector；A2
提高而 A1 没有提高，才形成 policy selection 的初步支持。两者均未提高时，本候选池收尾，
保留负面结果，不继续扩大增强搜索空间。

## 8. 最小消融和报告要求

- R0 vs C0：执行路径或重复曝光对照；
- R0 vs A1：固定增强的净作用；
- A1 vs A2：nested policy selection 的增量；
- A2 per-fold locks：选中次数、首位与次位分差、平局数和跨 RNG 排名一致性；
- per-class 和 call-count 分层只作为解释性分析，不能替代 primary endpoint；
- 报告所有候选、失败运行、protocol deviation 和未达到 gate 的结果。

## 9. 预期交付物

- `configs/protocol/meowagenet_idea039_grouped_augmentation_v1.json`；
- 独立 runner 和对应 tests；
- augmentation identity、group boundary、determinism 和 no-test-access audit；
- per-fold policy locks 与 execution lock；
- complete-OOF 机器可读结果；
- 中文结果报告和下一阶段 decision record。

## 10. 术语说明

| 术语 | 含义 |
| --- | --- |
| augmentation | 在训练时对输入作受控变化，生成保留原标签的新视图。目标是减少过拟合或提高对非关键变化的鲁棒性。 |
| policy | 一组确定的增强类型、强度、概率和组合顺序。它比“用了增强”包含更完整的可复现定义。 |
| SpecAugment | 直接在时频表示上遮蔽部分时间或频率区域的音频增强方法。猫叫很短，因此 mask 宽度需要限制。 |
| identity/no-op view | 数据经过与增强 pipeline 相同的调用和重复过程，但输入内容保持不变，用于分离增强内容与重复训练效应。 |
| augmentation RNG stream | 控制每次随机增强位置、强度或噪声的随机数序列。使用多个固定 streams 可以检查结果是否由一次随机抽样主导。 |
| nested-selected policy | 在每个 outer fold 的训练侧单独选出的增强策略。该 fold 的测试猫不参与选择。 |
| label preservation | 增强后的音频仍应具有原来的年龄类别。对于可能改变音高和时长的操作，这项假设较弱。 |
| call-count stratum | 按每只猫拥有的 call 数量形成的分析分组，用来查看收益是否只来自录音较多的猫。 |

## 11. Decision log

| 日期 | 决定 | 类型 | 理由 |
| --- | --- | --- | --- |
| 2026-09-15 | IDEA-039 成为 IDEA-058 strict 之后的当前性能路线 | human-approved roadmap update | IDEA-058 未通过 strict gate，Stage B 暂停 |
| 2026-09-15 | 固定增强和 nested selection 分成 H039-A/H039-B | AI-assisted proposal | 避免把增强有效与 selector 有效合并成一个 claim |
| 2026-09-15 | 加入 online/no-op control | design control | 排除执行路径、重复曝光和训练步数差异 |
| 2026-09-15 | 候选池保持有限 | rigor/feasibility decision | 降低小 validation set 上的多重选择和搜索噪声 |
