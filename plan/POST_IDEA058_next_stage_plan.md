# IDEA-058 之后的 AST 下一阶段计划

> 日期：2026-09-15
> 状态：团队已同意进入规划；尚未冻结 executable protocol，也未开始新实验
> 范围：MeowAgeNet 家猫年龄分类、公开数据、animal-ID 独立评价
> 当前参考：IDEA-056 tuned frozen AST，animal-level validation CE checkpoint selection

## 1. 计划要解决的问题

IDEA-058 的顶部单 block 适配取得 `0.7607 ± 0.0371` animal Macro F1，matched frozen AST
为 `0.7570 ± 0.0086`。平均增量为 `+0.0037`，三个 repeats 中两个为正；顶部更新又在三个
repeats 中全部高于同参数量的底部 block 更新。该结果支持“更新位置有影响”这一局部结论，
同时显示净增益较小、split 波动较大。

代码审查还发现，IDEA-058 先汇总全部 `3 repeats × 4 folds` 的 inner-validation 结果，再选出
一个全局 recipe 供所有 outer folds 使用。同一只猫会在一个 outer fold 中作为 test animal，
在其他 outer folds 中进入 train 或 validation。因此，对某个 outer fold 而言，全局 recipe
选择间接使用了该 fold 测试猫在其他 folds 中的标签信息。这属于跨 outer-fold 的选择信息
泄漏；`outer_test_accessed=false` 只说明单次 inner fit 没有读取本 fold 的 test role，不能消除
跨 folds 汇总造成的问题。

下一阶段先修正评价，再决定是否投入新的 top-block 方法。grouped augmentation 和 nuisance
诊断保留为独立路线。IDEA-021 猫域 self-supervised pre-adaptation 本阶段不考虑。

## 2. 证据、推断和待验证命题

| 类型 | 当前内容 | 可支持到什么程度 |
| --- | --- | --- |
| 已观察证据 | M1 相对 R0 的 Macro F1 为 `+0.0037`，2/3 repeats 为正，repeat SD 从 `0.0086` 增至 `0.0371` | 顶层适配具有弱正信号和更高波动 |
| 已观察证据 | M1 相对同参数量 C1 为 `+0.0222`，3/3 repeats 为正 | 顶部更新优于底部更新的证据比“优于冻结 AST”更一致 |
| 审计发现 | 四个候选在全部 12 个 inner splits 上汇总后只选择一个全局 recipe | IDEA-058 的增量需要 strict nested 复核 |
| 已观察证据 | Frozen AST embedding 对有效帧比例、时长和能量变量的 validation R² 为 `0.7295–0.8862` | AST 表示编码了这些变量 |
| 待验证推断 | top-block adaptation 的波动来自更新幅度、更新部件或小数据过拟合 | 需要训练动态和参数漂移诊断 |
| 待验证推断 | 时长、padding 或能量被分类器当作 shortcut | 现有 R² 不能证明；需要条件化误差分析 |

## 3. 四条路线的性质和优先级

| 顺序 | 路线 | 性质 | 当前决定 |
| ---: | --- | --- | --- |
| 1 | IDEA-058 strict nested 复核 | 评价修正，不是新方法 | 必做；先于任何 IDEA-058 扩展 |
| 2 | Top-block stabilization | 由 IDEA-058 结果直接产生的方法延展 | 复核保留正信号后，只选择一个实现 |
| 3 | IDEA-039 grouped augmentation policy | 已有 shortlist idea 的规范化实现 | 独立支持路线，不与路线 2 首轮组合 |
| 4 | Nuisance-variable robustness | 由内部诊断证据产生的新方向 | 先诊断；证实 shortcut 后才建立方法 Idea Card |

这里的顺序表示证据依赖关系，不是预期分数排名。路线 3 可以单独安排算力，但其结果不得用于
回头修改路线 1 的 recipe 或阈值。

## 4. 阶段 A：IDEA-058 strict nested 复核

### 4.1 目的

在每个 outer fold 内独立选择 top-block recipe，使该 fold 的测试猫在 recipe selection、epoch
selection 和模型选择过程中保持完全不可见。原 IDEA-058 结果继续作为探索性历史证据保存，
strict nested 结果另立报告，不能覆盖原报告。

### 4.2 最小流程

对每个 `repeat × outer fold` 分别执行：

1. 只使用该 outer fold 的 train/validation animals 比较已声明的 top-block 候选；
2. 按同一选择规则为该 fold 锁定一个 recipe；
3. 让 M1 使用该 fold 选出的 top-block 数量和 learning rate；
4. 让 C1 使用相同 block 数量、learning rate、final LayerNorm 和分类头，更新位置改为 bottom；
5. R0、M1、C1 使用相同 split、初始 head、batch order、checkpoint rule 和 cat-level aggregation；
6. 完成四个 outer folds 后拼接成一次 111-cat complete-OOF，再计算指标。

首轮沿用现有候选范围和 seed-17 的三个 split repeats，保证变化集中在 selection boundary。
如果工程实现需要改动候选范围，先写入新的 executable protocol，并把该运行标记为新实验，
不能称为对原结果的纯复核。

### 4.3 必须报告的结果

- Complete-OOF animal Macro F1、Balanced Accuracy、QWK、Accuracy 和 animal CE；
- 三个 repeat 的 M1−R0、M1−C1 配对差；
- Kitten、Adult、Senior 的 recall 与 F1；
- 每个 outer fold 选中的 recipe，而非只报告一个全局 recipe；
- 可训练参数量、训练时间、峰值显存、selected epoch 和失败运行；
- 与原 IDEA-058 结果的并列表，以及 selection-boundary 修正造成的变化。

### 4.4 决策边界

如果 strict nested 结果仍表现为 M1 平均高于 R0、至少 2/3 repeats 为正，并保持 M1 对 C1 的
位置优势，则 top-block stabilization 进入具体方法选择。建议沿用原 `+0.005` Macro F1 作为
seed expansion 的参考门槛，最终数值须在运行前由团队写入 executable protocol。

如果 M1 的正信号消失或位置对照也失去优势，IDEA-058 以“探索性弱正、严格复核未确认”收尾，
阶段 B 暂停。无论结果方向如何都完整保留并报告。

## 5. 阶段 B：Top-block stabilization

### 5.1 来源与目标

这是 IDEA-058 的直接后续。它要解决的是顶部更新相对 frozen AST 的高方差，而非重新证明 AST
优于 VGGish。开始前先查看每折的 train/validation 曲线、selected epoch、更新参数相对预训练
权重的漂移量，以及不同部件的 gradient/update norm。

### 5.2 候选机制

| 候选 | 大致作用 | 适用诊断 |
| --- | --- | --- |
| Gradual unfreezing | 先训练分类头，再在后期开放最后一个 block，减少训练早期对预训练表示的扰动 | 早期梯度或参数漂移过大 |
| L2-SP weight anchoring | 惩罚适配后权重偏离预训练权重，使 top block 在小数据上保持靠近初始化 | 训练集继续改善而 validation 波动扩大 |
| Top-block component adaptation | 只更新 attention、FFN 或 LayerNorm 中一个部件，缩小有效容量并定位收益来源 | 某一部件的 update norm 集中，或全 block 更新过强 |
| Top-block-only LoRA | 只在最后 block 的指定投影加入低秩更新 | 作为已有 PEFT 家族的补充配置；方法新意和优先级低于前三项 |

诊断后只选择一个候选，建立新的 Idea Card，再用 hypothesis-generation 写出主假设、竞争解释、
参数量匹配对照和反证条件。首轮比较保持单模块，不与 augmentation、fusion 或 local branch 组合。

## 6. 阶段 C：IDEA-039 grouped augmentation policy

IDEA-039 来自此前的人工 shortlist 和早期 augmentation 尝试。新工作重点是把经验性增强改写为
可审计的 animal-grouped policy，而非把“使用 augmentation”本身作为算法创新。

核心边界如下：

- augmentation 只在 outer-training role 内在线生成；
- 同一原始 call 及其所有增强版本始终属于同一 animal 和同一数据角色；
- policy 只在每个 outer fold 自己的 inner validation 上选择；
- 比较 no augmentation、单一增强和一个预先声明的有限组合；
- 强度范围围绕猫叫的短时特性设置，记录增强后有效时长、频谱变化和标签保持假设；
- 以 strict reference 作为同期对照，同时报告 Macro F1、类别 recall、CE 和跨 repeats 方差。

该路线主要服务性能提升、鲁棒性和毕业设计工作量。若要形成论文方法贡献，还需要证明选择策略
或 group-aware 约束带来可复用价值；单个数据集上的分数提升本身属于应用证据。

## 7. 阶段 D：Nuisance-variable robustness 诊断

Frozen AST embedding 能预测时长、有效帧比例和能量，说明这些信息存在于表示中。分类器是否
利用它们、它们属于有效年龄线索还是数据采集 shortcut，目前仍未知。

先完成低成本诊断：

1. 在 cat level 汇总 duration、valid-frame fraction 和 energy，查看其与年龄类别及数据来源的
   关联；
2. 使用已经锁定的 OOF prediction，分析控制真实年龄类别后，预测正确率、置信度和错误方向是否
   仍随这些变量系统变化；
3. 比较 R0 与 strict M1 的依赖强度，判断 top-block adaptation 是减弱还是放大这种关系；
4. 在预先定义的 duration/energy 匹配子集上复算指标，作为敏感性分析，不替代完整 OOF 主结果。

如果只有“信息可解码”，而没有“预测依赖或跨来源失稳”的证据，该路线停在诊断报告。若 shortcut
证据成立，再从 duration-balanced sampling、padding-invariant pooling、nuisance-adversarial
learning 或 nuisance-conditioned normalization 中选择一个方法，并另行分配 IDEA 编号。时长和
能量也可能是真实年龄线索，因此任何抑制方法都要同时检查总体性能和三个年龄类别的代价。

## 8. 阶段交付和停止规则

| 阶段 | 最小交付物 | 继续条件 |
| --- | --- | --- |
| A | strict protocol、runner、per-fold recipe locks、结果 JSON、中文报告、审计修正记录 | 正信号和 top-vs-bottom 位置优势在严格边界下保留 |
| B | 一个新 Idea Card、单模块 protocol、R0/M1/control 结果与参数漂移诊断 | 出现跨 repeats 的稳定正信号后才考虑模块组合 |
| C | IDEA-039 Idea Card、有限 policy 表、group-safe audit、paired OOF 结果 | 依据预先冻结 gate 决定 seed expansion |
| D | nuisance association、conditional error 和 matched-subset 报告 | 证实预测依赖后才进入方法实验 |

当前阶段禁止同时组合 top-block stabilization、augmentation 和 nuisance module。一次只改变一个
主要因素，保留 matched reference。最终路线取舍由团队成员或导师完成。

## 9. 术语说明

| 术语 | 含义 |
| --- | --- |
| strict nested evaluation | 每个 outer fold 都只利用本 fold 的训练侧数据选择 recipe。测试 animals 只在 recipe 和训练完全锁定后用于一次评价。 |
| cross-fold selection leakage | 在多个 outer folds 之间汇总 validation 结果选择同一个 recipe，使某 fold 的 test animals 可能通过其他 folds 的 train/validation 角色影响选择。它会让 outer 评价偏乐观。 |
| recipe | 一组模型和训练设置，例如更新哪些 blocks、learning rate、regularization 和 checkpoint rule。 |
| top-block stabilization | 降低 AST 顶层更新在不同数据划分上的波动，使增益更可重复的一类方法目标。这是项目工作名称，不是文献中的固定算法名。 |
| gradual unfreezing | 训练初期冻结 encoder，随后逐步开放靠近输出的层。它限制预训练权重在早期受到的大幅扰动。 |
| L2-SP weight anchoring | 用 L2 penalty 约束微调权重靠近 pretrained starting point。SP 指 starting point。 |
| component adaptation | 只更新 Transformer block 中的某类部件，例如 attention、FFN 或 LayerNorm，用于控制容量并定位作用来源。 |
| grouped augmentation policy | 以 animal group 为数据边界选择和应用增强，确保原音频及其增强副本不跨 train、validation、test。 |
| nuisance variable | 与目标标签无直接定义关系、却可能影响模型预测的变量，例如 padding 比例、录音能量或来源设备。它也可能携带真实信号，因此需要证据区分。 |
| shortcut | 模型利用数据中的容易相关线索取得分数，而没有学习期望的可泛化规律。是否构成 shortcut 需要跨来源或条件化分析支持。 |
| matched subset | 在时长、能量等变量分布接近的样本子集上比较模型，用来检查主结论是否依赖这些变量。 |

## 10. Decision log

| 日期 | 决定 | 类型 | 理由 |
| --- | --- | --- | --- |
| 2026-09-15 | IDEA-058 先做 strict nested 复核 | evidence/audit-driven | 原实现存在跨 outer-fold recipe selection 信息泄漏 |
| 2026-09-15 | Top-block stabilization 由 strict 结果决定是否启动 | conditional proposal | 原 M1 仅有弱平均增益且 repeat SD 较高 |
| 2026-09-15 | IDEA-039 作为独立性能与鲁棒性路线 | prior human shortlist | 已有 augmentation 经验需要 group-safe protocol 重新检验 |
| 2026-09-15 | Nuisance 路线先诊断后立项 | evidence-driven proposal | AST embedding 的高 R² 证明信息存在，尚未证明 classifier shortcut |
| 2026-09-15 | 本阶段不考虑 IDEA-021 | human decision | 团队当前选择 |
