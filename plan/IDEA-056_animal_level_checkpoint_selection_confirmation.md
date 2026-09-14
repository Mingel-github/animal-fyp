# IDEA-056｜按猫级验证损失选择训练轮次的确认计划

## 1. 术语说明

| 术语 | 含义 |
| --- | --- |
| epoch（训练轮次） | 模型完整遍历一次训练数据。不同训练轮次对应不同的模型状态。 |
| checkpoint（模型检查点） | 某个训练轮次结束时保存的模型参数。本文所说“选择 checkpoint”，就是决定最终使用第几轮的模型。 |
| call-level CE（叫声级交叉熵） | 对每条猫叫分别计算 cross-entropy，再汇总为验证损失。CE 越低，模型通常给真实类别分配的概率越高。 |
| animal-level CE（猫级交叉熵） | 先平均同一只猫所有叫声的预测概率，再以每只猫为一个单位计算 cross-entropy。它与本项目最终按猫评价的单位一致。 |
| inner validation（内部验证集） | 从当前 outer-training 数据中划出的验证部分，只用于选择训练轮次。 |
| outer test（外部测试折） | 当前交叉验证中留出的猫，只在训练轮次已经确定后用于评价。 |
| base seed（基础随机种子） | 控制模型初始化和训练随机过程的数值。使用不同随机种子可以检查结果是否依赖某次幸运训练。 |
| matched comparison（匹配比较） | 两组实验共享数据划分、初始化、batch 顺序、模型和训练配置，只改变需要检验的一个因素。 |
| complete OOF（完整折外预测） | 合并各测试折后得到覆盖全部 111 只猫的预测；每只猫均由训练时未包含它的模型预测。 |

## 2. 观察及证据边界

这是一个在查看 IDEA-051 结果后发现的探索性观察，因此不能写成预先提出的结论。

- 旧 C0 使用“最低叫声级验证交叉熵”选择训练轮次；在 base seed 17 的三个 repeat 中，
  animal-level Macro F1 均值为 0.7464。
- 新 A0 使用“最低猫级验证交叉熵”选择训练轮次；相同三个 repeat 的均值为 0.7570。
- 观察到的均值差为 +0.0106，三个 repeat 中两个方向为正。
- IDEA-051 至 IDEA-055 多次精确复现 0.7570，证明新 A0 的实现可重复；这些运行使用相同
  seed 和数据划分，因此不构成独立的统计确认。

本计划只确认训练轮次选择规则。它不引入新的 AST 模块，也不把流程优化描述为 AST
architecture innovation。

## 3. 研究问题

在 frozen AST、分类头、训练 loss、数据划分和训练轨迹保持一致时，以猫级验证交叉熵
选择训练轮次，能否比叫声级验证交叉熵带来更高、更稳定的 unseen-cat Macro F1？

## 4. 候选假设与竞争解释

`H056` 是本项目对 IDEA-056 研究假设的编号。

### H056-A：评价单位对齐假设

猫级验证交叉熵直接评价同一只猫多条叫声聚合后的预测，因此更接近最终评价过程。它选择的
训练轮次预计在新随机种子上继续提高 animal-level Macro F1。

### H056-R1：随机波动解释

seed 17 的 +0.0106 来自有限数据划分和训练随机性。使用 seed 43 和 101 后，差值可能收敛
到零或改变方向。

### H056-R2：概率目标与分类指标不一致解释

猫级交叉熵关注真实类别获得的概率，Macro F1 关注最终类别判断。猫级交叉熵可能改善概率
质量，同时不提高 Macro F1、Balanced Accuracy 或 QWK。

## 5. 预先声明的预测

### 主要预测

只使用尚未查看结果的 base seeds 43 和 101，共 6 组 complete-OOF 配对：

- 猫级选择规则相对叫声级选择规则的平均 Macro F1 增量至少为 +0.005；
- 6 组配对中至少 4 组方向为正。

### 支持预测

- Balanced Accuracy、QWK、普通 accuracy 或最差 repeat 的 Macro F1 至少一项同时改善；
- 猫级选择规则的 repeat 间波动不出现明显扩大；
- paired cat-cluster bootstrap 的差值分布中心位于正方向。

### 证伪或信息不足

- 新 6 组平均差小于或等于零，支持随机波动解释；
- 平均差为正但低于 +0.005，或少于 4/6 配对为正，记为弱而不稳定的信号；
- 猫级 CE 改善而 Macro F1 没有改善，支持“概率目标与离散分类指标不一致”；
- 某一规则只在单一 seed 或单一 repeat 上占优，结论限定为 seed/split dependent。

## 6. 匹配实验设计

### 6.1 两条规则

| 编号 | 训练轮次选择规则 | 角色 |
| --- | --- | --- |
| C0-call | 最低 unweighted inner-validation call-level CE | 旧规则参考 |
| C1-animal | 最低 unweighted inner-validation animal-level CE | 待确认规则 |

若多个训练轮次数值相同，固定选择最早的训练轮次。

### 6.2 共享训练轨迹

每个 fold 只建立一条内部训练轨迹，并在每个 epoch 同时记录叫声级 CE 和猫级 CE。训练持续
到 maximum 50 epochs，或在两种选择指标都连续 8 个 epoch 没有改善后停止。这样两条规则
从完全相同的候选 checkpoint 中选择，避免训练轨迹成为额外变量。

确定 `E_call` 和 `E_animal` 后，在合并后的 outer-training 数据上使用相同初始化和相同
batch 顺序训练至 `max(E_call, E_animal)`，分别保存两个目标 epoch 的模型状态。两组 outer
prediction 由同一条训练轨迹产生。

### 6.3 固定条件

- 数据：792 条 calls、111 只猫、kitten/adult/senior 三分类；
- split：现有 cat-ID-disjoint repeat 0/1/2 与四折角色；
- backbone：相同 frozen standard AST embedding；
- head：相同 `768 → 128 → 3`；
- dropout、optimizer、learning rate、global class-balanced call loss、batch accumulation 与
  当前 A0 保持一致；
- 猫级预测：平均同一只猫所有 call probabilities；
- 新确认数据：base seeds 43、101；
- seed 17：只作为历史探索性结果展示，不参与本计划的主要继续判断。

## 7. 评价与分析

主要结果为 6 组新 complete-OOF 的 paired animal-level Macro F1 差值。支持结果包括：

- Balanced Accuracy；
- QWK；
- 普通 accuracy；
- animal-level CE；
- kitten、adult、senior 的 precision 与 recall；
- confusion matrix；
- 每只猫预测改变、改正和改错的次数；
- selected epoch 的分布；
- 以 cat 为单位的 paired bootstrap 区间。

报告同时给出 seed 17 历史结果和 9 组联合描述，但确认判断只依赖新生成的 seed 43/101。

## 8. 阶段决策

- 达到主要预测：将猫级 CE 选择规则确定为后续 tuned frozen-AST reference 的标准组成。
- 平均差为正但未达到主要预测：保留为 mixed evidence，后续 reference 由团队决定。
- 平均差小于或等于零：继续使用证据更稳定的现有规则，并将 seed-17 改善记录为探索性结果。

本计划完成后，再启动新的 AST architecture idea。两个阶段分别记录，避免把 checkpoint
选择确认与新模型模块混在同一个贡献中。

## 9. 预期产物

- 可执行 protocol；
- 独立 runner 与单元测试；
- inner-only smoke 和 execution lock；
- 6 组新 complete-OOF 配对结果；
- 中文结果报告与机器可读 JSON；
- 对 IDEA-056 状态和最终 reference 定义的更新记录。

## 10. IDEA-056 之后的阶段顺序

以下顺序是团队确认的 `decision`。每一步形成记录后再进入下一步，避免诊断、模型选择和
组合实验同时展开。

1. 先完成 IDEA-056，确定后续实验统一使用的 AST 参考流程。
2. 对 AST 内部进行诊断，判断当前缺失或未被充分利用的信息主要位于局部 patch、某些
   中间层，还是预训练领域与猫叫领域之间的差异。
3. 根据诊断结果只选择一个 AST 内部改进方向，由团队确认后建立新的 Idea Card。
4. 首轮实验只测试这个模块，并设置参数量相近的对照和原始 AST 对照。
5. 单模块出现跨数据划分和随机种子的稳定正信号后，再考虑与其他模块组合。

第 2 步只负责定位问题。诊断相关性本身不证明某个模块有效；具体模块仍需通过第 4 步的
匹配实验检验。
