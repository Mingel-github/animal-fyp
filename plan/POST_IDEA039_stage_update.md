# IDEA-039 之后的阶段更新

> 日期：2026-09-16
> 状态：当前 roadmap
> 前置计划：`plan/IDEA-039_grouped_augmentation_policy.md`，已被 IDEA-039 protocol 和
> execution lock 固定哈希

## 1. IDEA-039 结果

| Pipeline | Animal Macro F1 | 相对 R0 | Repeat 方向 | Gate |
| --- | ---: | ---: | --- | --- |
| R0 cached frozen AST | 0.7570 ± 0.0086 | reference | — | — |
| C0 online identity | 0.7557 ± 0.0155 | −0.0013 | 1 正 / 2 负 | 路径门未通过 |
| A1 fixed SpecAugment | 0.7578 ± 0.0405 | +0.0008 | 2 正 / 1 负 | H039-A 未通过 |
| A2 nested policy | 0.7556 ± 0.0231 | −0.0014 | 2 正 / 1 负 | H039-B 未通过 |

A1 的 QWK 相对 R0 提高 `0.0109`，Balanced Accuracy 提高 `0.0027`，形成弱正副指标信号；
Macro F1 增益低于 `+0.005`，repeat 2 的下降使跨 split 波动扩大。A2 在 12 折中选中四种
policy，11/12 折达到流稳定标准；其外层 Macro F1 与 R0/C0 接近，animal CE 相对 R0 增加
`0.0642`。

C0 与 R0 的平均 Macro F1 差为 `−0.00134`，预测一致率为 `95.80%`，因此后续在线输入实验
继续保留 matched identity control，并同时报告缓存参考与同路径对照。

完整证据见 `reports/34_IDEA-039_grouped_augmentation_results.md`。

## 2. 路线状态

| 路线 | 状态 | 决定依据 |
| --- | --- | --- |
| Stage A：IDEA-058 strict nested | 完成 | strict 复核已形成正式结果 |
| Stage B：Top-block stabilization | 暂停 | IDEA-058 strict gate 未通过 |
| Stage C：IDEA-039 grouped augmentation | 完成 | A1/A2 与路径门均未达到继续条件；有限候选池收尾 |
| Stage D：Nuisance-variable robustness | 下一活动 | 使用现有 OOF 和元数据先完成低成本预测依赖诊断 |
| IDEA-021 self-supervised pre-adaptation | 本阶段不进入 | 团队既有决定 |

## 3. 下一阶段执行顺序

1. 冻结 Stage D 的只读诊断问题、变量定义、cat-level 汇总方法和 matched-subset 规则；
2. 在 cat level 汇总 duration、valid-frame fraction、energy 与数据来源，描述类别条件下的分布；
3. 使用已锁定 R0 和 strict M1 complete-OOF，分析控制真实年龄类别后，正确率、置信度和错误方向
   与 nuisance variables 的关系；
4. 在预先定义的 duration/energy 匹配子集上复算指标，形成敏感性分析；
5. 若出现预测依赖或跨来源失稳证据，再为一个具体鲁棒性机制建立新 Idea Card；若信号较弱，
   Stage D 以诊断报告收尾。

Stage D 首轮复用现有 OOF 预测，不训练新模型。它先回答分类器是否实际依赖可解码的时长、padding
或能量信息，再决定 duration-balanced sampling、padding-invariant pooling、adversarial learning
或 conditional normalization 中哪一种值得进入方法实验。

## 4. 审计边界

- IDEA-039 原计划、protocol、runner、selection locks、execution lock 和结果 JSON 保持原样；
- seeds 43/101 不进入 IDEA-039 扩展；
- 当前有限增强池不继续增加增强类型或强度组合；
- call-count 分层与 nuisance matched-subset 均承担解释性分析，complete-OOF animal Macro F1
  继续作为性能主指标；
- 新方法实验在 Stage D 诊断形成证据后另行编号和冻结。

## 5. Decision log

| 日期 | 决定 | 类型 | 理由 |
| --- | --- | --- | --- |
| 2026-09-16 | IDEA-039 有限候选池阶段性收尾 | evidence-driven decision | A1/A2 主指标增益均低于继续阈值 |
| 2026-09-16 | 不扩展 seeds 43/101 | frozen-gate decision | H039-A、H039-B 与路径门均为 false |
| 2026-09-16 | 后续在线增强保留 C0 | audit-driven decision | 在线/缓存预测一致率为 95.80% |
| 2026-09-16 | Stage D 成为下一活动 | roadmap transition | 既定顺序要求先做低成本 prediction-dependence 诊断 |
