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

- [`AST_dual_direction_validation_plan.md`](AST_dual_direction_validation_plan.md)：团队决定将诊断得到的两个候选方向分别开展，首轮保持独立验证。
- [`IDEA-057_structured_local_patch_branch.md`](IDEA-057_structured_local_patch_branch.md)：保留 final AST 主路径，检验位置感知的局部时间—频率分支。
- [`IDEA-058_constrained_top_block_adaptation.md`](IDEA-058_constrained_top_block_adaptation.md)：检验受约束的 AST 顶层更新，并以同参数量的非顶部 block 更新作为位置对照。

第一阶段 inner-only 诊断已经完成，结果见
`reports/30_AST_internal_diagnosis_results.md`。三条完整支持规则均未通过，局部 patch 形成
当前最强机制线索，Last-2 形成有实际幅度的 mixed signal。2026-09-15 团队决定两条路线
均继续，各自完成 R0/M1/C1 比较；首轮不组合模块。

`AST_internal_diagnosis_and_single_module_plan.md` 已完成诊断阶段并进入审计记录，作为原始
设计历史保留，不反向修改。

`AST_cat_balance_global_weighting_retest.md` 已执行完毕，对应结果保存在
`reports/23_AST_cat_balance_global_weighting_results.md`，现作为设计历史保留。
