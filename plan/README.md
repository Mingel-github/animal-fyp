# Plan 目录说明

本目录保存需要交给协作者继续执行的内容：

- idea 交接和候选方法定义；
- 实验规划、对照组和消融矩阵；
- 可读版 protocol freeze 与 amendment；
- 查看实验结果之前确定的选择规则和停止规则。

`reports/` 保存已经完成的实验现象、实验结果、证据审查和阶段结论。可执行的 protocol JSON 继续放在 `configs/`，机器可读的结果记录继续放在 `metadata/`。

部分已经执行的旧 protocol 和 runner 将 `reports/...` 路径及文件 SHA-256 写入了审计记录。因此，`reports/09`、`reports/10`、`reports/13` 和 `reports/19` 保留逐字节一致的历史原件，`plan/` 中保留便于浏览的相同副本。这些历史文件保持只读；新的规划文件只在 `plan/` 创建。

一项 plan 执行后，原 plan 作为设计历史保留，并在 `reports/` 新建结果报告。后续结果不反向改写原计划。

## 当前活动计划

- [`POST_IDEA058_next_stage_plan.md`](POST_IDEA058_next_stage_plan.md)：当前总计划。先修正
  IDEA-058 的跨 outer-fold recipe selection 边界，再按证据决定 top-block stabilization，
  同时保留 IDEA-039 grouped augmentation 和 nuisance-variable 诊断两条独立路线。
- [`IDEA-058_constrained_top_block_adaptation.md`](IDEA-058_constrained_top_block_adaptation.md)：
  原始冻结计划，已经执行；结果见 `reports/32_IDEA-058_constrained_top_block_results.md`。
- [`IDEA-057_structured_local_patch_branch.md`](IDEA-057_structured_local_patch_branch.md)：
  原始冻结计划，已经执行；结果见 `reports/31_IDEA-057_structured_local_patch_results.md`。

IDEA-057 和 IDEA-058 的首轮独立验证均已完成。IDEA-057 的位置感知局部分支低于 matched R0
`0.0305` Macro F1，当前参数化关闭。IDEA-058 的 top-block M1 相对 R0 为 `+0.0037`，2/3
repeats 为正；相对同参数量 bottom-block C1 为 `+0.0222`，3/3 为正。M1 没有达到原
`+0.005` seed expansion gate，且 repeat SD 为 `0.0371`。

后续代码审查确认 IDEA-058 通过全部 12 个 inner splits 汇总选择一个全局 recipe。为确保每个
outer fold 的测试猫不影响该 fold 的模型选择，当前第一任务是 per-outer-fold strict nested
复核。原报告作为探索性历史证据保留，不反向修改。IDEA-021 本阶段不考虑。

`AST_internal_diagnosis_and_single_module_plan.md` 已完成诊断阶段并进入审计记录，作为原始
设计历史保留，不反向修改。

`AST_cat_balance_global_weighting_retest.md` 已执行完毕，对应结果保存在
`reports/23_AST_cat_balance_global_weighting_results.md`，现作为设计历史保留。
