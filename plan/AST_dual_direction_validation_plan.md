# AST 双方向独立验证计划

> 决策日期：2026-09-15
> 决策主体：项目团队
> 前置结果：`reports/30_AST_internal_diagnosis_results.md`
> 状态：两条路线均进入独立计划阶段

## 1. 团队决定

AST 内部诊断的三条完整支持规则均未通过。诊断同时形成两种性质不同的候选证据：局部
patch 提供较清楚的结构线索，Last-2 适配提供较直接但不稳定的分类指标提升。项目团队决定
保留两条路线，分别执行：

- **IDEA-057：结构化局部 patch 分支**，检验局部时间—频率结构能否在 final AST global
  embedding 之外提供稳定增量；
- **IDEA-058：受约束的 AST 顶层适配**，检验更新顶部 Transformer blocks 能否把诊断中的
  平均收益转化为稳定收益，并区分更新位置与单纯参数量效应。

两条路线是平行的独立实验，不在首轮相互组合。原诊断生成的
`decision_record_draft.json` 作为当时的冻结产物保留；本文件记录诊断完成后的团队决定。

## 2. 共同参考流程

两条路线使用同一个 R0 reference：

- standard pretrained AST；
- AST backbone 冻结，仅训练既有 `768 → 128 → 3` 分类头；
- 使用现有 animal-ID-disjoint nested roles；
- 以 animal-level validation cross-entropy 选择 checkpoint；
- 对同一只猫的 call probabilities 取算术平均；
- primary metric 为 animal-level Macro F1；
- 同时报告 Balanced Accuracy、QWK、普通 accuracy、animal-level CE 和各类别
  precision/recall/F1。

R0 采用 IDEA-056 确认后的流程。每条新路线都与本轮重新运行的 matched R0 比较，历史分数
只用于背景说明。

## 3. 两条路线的角色

| 路线 | 主要问题 | 首轮核心对照 | 论文中的潜在角色 |
| --- | --- | --- | --- |
| IDEA-057 | 保留二维位置的局部表示能否补充 global AST | R0、局部模块 M1、参数量匹配但移除局部结构的 C1 | AST 结构改造与机制消融 |
| IDEA-058 | 顶层受约束更新能否稳定适应猫叫年龄任务 | R0、顶部 blocks 更新 M1、同参数量的非顶部 blocks 更新 C1 | AST adaptation 方法与更新位置消融 |

IDEA-057 优先回答“利用什么信息”，IDEA-058 优先回答“更新 AST 的哪个位置”。两条路线的
正、负结果分别报告，不以其中一条结果修改另一条已经冻结的 recipe。

## 4. 执行顺序

1. 分别完成两张 Idea Card 和有界的 inner-only 候选定义；
2. 分别实现 runner、参数审计、初始化检查和 smoke test；
3. 只使用 inner-training / inner-validation 选择每条路线的一种 recipe；
4. 在读取 outer-test 结果前，分别保存 protocol、runner hash 和 selection lock；
5. 分别完成 R0/M1/C1 的配对 complete-OOF evaluation；
6. 根据各自预先声明的继续条件决定是否扩展 random seeds；
7. 两条路线都完成独立验证后，再决定是否设计组合实验。

两个实验可以在外部计算机上顺序或并行运行。并行只表示节省时间；数据角色、配置锁和结果
目录仍保持完全分离。

## 5. 共同证据规则

每条路线的第一判断都是 M1 相对同期 R0 的 paired Macro F1。C1 用于检查 M1 的收益能否
归因于目标机制。单次最高分、单个 fold 或训练集拟合改善不构成继续依据。

初始正信号需要同时满足：

- M1 相对 R0 的平均 Macro F1 增量至少 `+0.005`；
- 三个 complete-OOF repeats 中至少两个为正；
- M1 相对参数量匹配 C1 的方向与主要解释相容；
- Balanced Accuracy、QWK、关键类别 recall、CE 和波动没有出现无法解释的系统性代价。

达到初始条件后才扩展新的 random seeds。seed expansion 使用已经锁定的模块与训练 recipe，
不重新搜索结构或超参数。具体确认阈值在执行 protocol 中冻结，并与初始探索结果分开报告。

## 6. 组合条件

- 两条路线都形成稳定正信号：先各自完成确认，再建立 R0、IDEA-057、IDEA-058、二者组合
  的四组消融，检验组合是否具有额外增量；
- 只有一条路线形成稳定正信号：扩展该路线，另一条保留为有边界的负面或混合结果；
- 两条路线都未形成稳定正信号：本轮 AST 内部扩展收尾，保留诊断、参数匹配和失败结果；
- 任一路线出现 outer-test 泄漏、对照失配或执行锁错误：该路线结果失效，修复后重新运行。

组合实验属于新的阶段，不修改 IDEA-057 或 IDEA-058 的独立结果。

## 7. 重复使用数据的解释边界

现有 111 只猫和 split bank 已被多个探索使用。IDEA-057、IDEA-058 的 complete-OOF 结果属于
同一公开数据集上的 repeated grouped internal evidence。它适合支持毕业设计中的方法比较，
不能当作新数据上的 external replication。投稿时需要公开说明多轮探索过程，并将最终锁定
方法与探索结果区分。

## 8. 术语说明

| 术语 | 含义 |
| --- | --- |
| dual direction（双方向） | 项目决策用语，表示两个候选分别立项和验证。它不表示两个模块从第一轮开始组合。 |
| matched R0（同期匹配参考） | 与新方法使用相同 split、seed、batch 顺序、评价和 checkpoint 规则重新运行的原始 AST 流程。 |
| recipe（实验配方） | 模型结构、可训练参数、learning rate、regularization 和 checkpoint 选择规则的完整组合。它属于工程用语，不是新算法。 |
| selection lock（选择锁） | 在访问相应 outer-test 前保存的候选选择与文件哈希，用于证明方法选择没有利用该测试结果。 |
| seed expansion（随机种子扩展） | 在结构和超参数保持不变后增加训练随机种子，检查结果对初始化和训练随机性的敏感程度。 |
| external replication（外部重复验证） | 在独立数据或独立采样条件下重新验证结果。本项目复用同一 111 只猫，因此当前属于内部重复评价。 |

## 9. 文件分工

- IDEA-057：`plan/IDEA-057_structured_local_patch_branch.md`；
- IDEA-058：`plan/IDEA-058_constrained_top_block_adaptation.md`；
- 已完成诊断：`reports/30_AST_internal_diagnosis_results.md`；
- 冻结的原阶段计划：`plan/AST_internal_diagnosis_and_single_module_plan.md`。
