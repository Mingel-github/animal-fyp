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

- [`AST_internal_diagnosis_and_single_module_plan.md`](AST_internal_diagnosis_and_single_module_plan.md)：在 IDEA-056 确定 AST 参考流程后，依次诊断局部 patch、中间层和预训练领域差异；根据证据只选择一个 AST 内部方向，并完成单模块、参数量匹配和原始 AST 对照。

第一阶段 inner-only 诊断已经完成，结果见
`reports/30_AST_internal_diagnosis_results.md`。三条完整支持规则均未通过，局部 patch 形成
当前最强机制线索；decision record 等待团队选择单模块方向和新的 IDEA 编号。

`AST_cat_balance_global_weighting_retest.md` 已执行完毕，对应结果保存在
`reports/23_AST_cat_balance_global_weighting_results.md`，现作为设计历史保留。
