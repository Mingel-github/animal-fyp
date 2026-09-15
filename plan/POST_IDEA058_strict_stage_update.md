# IDEA-058 strict 之后的阶段更新

> 日期：2026-09-15
> 状态：当前 roadmap
> 前置计划：`plan/POST_IDEA058_next_stage_plan.md`，已被 strict protocol 记录哈希并保持只读

## 1. Strict 结果

| Pipeline | Animal Macro F1 | 相对 R0 | Repeat 方向 |
| --- | ---: | ---: | --- |
| R0 tuned frozen AST | `0.7570 ± 0.0086` | 参考 | — |
| M1 strict top-block adaptation | `0.7493 ± 0.0211` | `−0.0077` | 1/3 为正 |
| C1 strict bottom-block control | `0.7337 ± 0.0139` | `−0.0233` | 0/3 为正 |

M1 相对 C1 平均提高 `0.0156` Macro F1，2/3 repeats 为正，bootstrap 区间跨过零。该结果
保留 top block 优于 bottom block 的有限线索，同时未确认 top-block adaptation 相对 frozen AST
的性能增益。M1 的 animal CE 比 R0 低 `0.0235`，属于 probability log-loss 的改善；当前实验
没有单独测量 calibration。

完整证据见 `reports/33_IDEA-058_strict_nested_confirmation_results.md`。

## 2. 路线状态

| 路线 | 状态 | 决定依据 |
| --- | --- | --- |
| Stage A：IDEA-058 strict nested | 完成 | per-fold selection 已执行，R0 锚点逐位复现 |
| Stage B：Top-block stabilization | 暂停 | M1 相对 R0 为负，strict gate 未通过 |
| Stage C：IDEA-039 grouped augmentation | 当前活动 | 独立性能路线，不依赖 IDEA-058 正信号 |
| Stage D：Nuisance-variable robustness | 排队中 | 先诊断 prediction dependence，再决定是否建立方法 IDEA |
| IDEA-021 self-supervised pre-adaptation | 本阶段不进入 | 团队既有决定 |

## 3. 当前执行顺序

1. 将 `plan/IDEA-039_grouped_augmentation_policy.md` 转化为 executable protocol；
2. 先验证 online/no-op pipeline 能复现 R0，随后运行固定轻量 augmentation；
3. 在每个 outer fold 内独立完成 augmentation policy selection；
4. 分别判断固定增强 H039-A 和 nested selector H039-B；
5. 依据预先冻结 gate 决定是否扩展 seeds 43/101；
6. IDEA-039 收尾后执行 nuisance-variable 低成本诊断。

IDEA-039 首轮与 top-block、fusion、local branch 和 nuisance module 保持分离。Top-block
stabilization 只在新的独立证据或团队明确重启后恢复。

## 4. 审计边界

- 原 `POST_IDEA058_next_stage_plan.md` 保持原 SHA-256，不反向写入结果；
- strict 结果报告和结果 JSON保持原样；
- IDEA-039 的候选、强度、随机种子、选择规则和继续条件在查看 outer-test 结果前冻结；
- 所有 augmentation 视图继承原 call 的 `cat_id` 和数据角色；
- 固定增强是否有效、nested selector 是否有效分别报告，避免合并 claim。

## 5. Decision log

| 日期 | 决定 | 类型 | 理由 |
| --- | --- | --- | --- |
| 2026-09-15 | IDEA-058 结束方法扩展 | evidence-driven decision | strict M1 未超过 R0，gate 未通过 |
| 2026-09-15 | Stage B 暂停 | conditional decision | 原计划的启动条件未满足 |
| 2026-09-15 | IDEA-039 进入当前活动队列 | human-approved roadmap update | 该路线与 IDEA-058 独立，适合继续检验性能增益 |
| 2026-09-15 | nuisance 诊断随后执行 | evidence-aware proposal | 现有高 R² 只证明信息可解码，尚未证明 classifier 依赖 |
