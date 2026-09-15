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

- [`IDEA-039_grouped_augmentation_policy.md`](IDEA-039_grouped_augmentation_policy.md)：
  当前执行计划。分别检验固定轻量增强和 per-outer-fold nested policy selection，并使用
  online/no-op control 排除执行路径与重复曝光效应。
- [`POST_IDEA058_strict_stage_update.md`](POST_IDEA058_strict_stage_update.md)：当前阶段总览。
  IDEA-058 strict 已完成，top-block stabilization 暂停，当前进入 IDEA-039；nuisance-variable
  诊断排队。
- [`POST_IDEA058_next_stage_plan.md`](POST_IDEA058_next_stage_plan.md)：strict 运行前的冻结计划，
  已被 protocol 记录 SHA-256，作为设计历史保持原文。
- [`IDEA-058_constrained_top_block_adaptation.md`](IDEA-058_constrained_top_block_adaptation.md)：
  原始冻结计划，已经执行；结果见 `reports/32_IDEA-058_constrained_top_block_results.md`。
- [`IDEA-057_structured_local_patch_branch.md`](IDEA-057_structured_local_patch_branch.md)：
  原始冻结计划，已经执行；结果见 `reports/31_IDEA-057_structured_local_patch_results.md`。

IDEA-058 strict nested 复核已经完成，结果见
`reports/33_IDEA-058_strict_nested_confirmation_results.md`。M1 的 mean animal Macro F1 为
`0.7493`，相对 R0 的 `0.7570` 为 `−0.0077`，仅 1/3 repeats 为正。Strict gate 未通过，
IDEA-058 以“探索性弱正、严格复核未确认”收尾，top-block stabilization 暂停。

M1 相对 bottom-block C1 仍有 `+0.0156` Macro F1 的平均差，作为有限的位置线索保留。当前
性能路线切换到 IDEA-039；IDEA-021 本阶段不考虑。

`AST_internal_diagnosis_and_single_module_plan.md` 已完成诊断阶段并进入审计记录，作为原始
设计历史保留，不反向修改。

`AST_cat_balance_global_weighting_retest.md` 已执行完毕，对应结果保存在
`reports/23_AST_cat_balance_global_weighting_results.md`，现作为设计历史保留。
