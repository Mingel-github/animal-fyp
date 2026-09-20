# AST 与声学残差研究：GitHub 归档入口

归档日期：2026-09-20。本次收录当前研究工作区 IDEA-065 至 IDEA-088 的代码、测试、冻结协议、实验计划、中文报告和结果元数据。历史实验按各自协议保留，跨轮分数应结合对应评分单位、划分和训练方式理解。

## 建议阅读顺序

- [v3 综合研究报告（截至 IDEA-082）](58_Formal_Research_Summary_v3.md)：方法代号、残差原理、历史证据与研究边界。
- [IDEA-083：C1/U1 等权预测融合](61_IDEA-083_C1_U1_equal_weight_fusion_results.md)。
- [IDEA-084：分组双分支声学残差](64_IDEA-084_grouped_dual_branch_residual_results.md)。
- [IDEA-085：声学时间残差](70_IDEA-085_acoustic_temporal_residual_results.md)。
- [IDEA-086：声学集合残差](74_IDEA-086_acoustic_set_residual_results.md)。
- [IDEA-087：嵌套调参结果](78_IDEA-087_nested_tuning_results.md)。
- [IDEA-088：清洗数据上的原 baseline 风格对照](82_IDEA-088_original_style_clean_results.md)。
- [IDEA-088 独立结果审计](81_IDEA-088_original_style_clean_independent_audit.md)。

## 归档范围

代码、协议、测试与实验元数据分别位于 `scripts/`、`configs/protocol/`、`tests/`、`metadata/experiments/`。本轮 IDEA-088 另收录角色清单、训练前检查、聚合结果、独立验证、恢复执行状态，以及 60 个单次训练摘要，共 65 份轻量 JSON 审计记录。

原始音频、外部数据集、特征缓存、逐条预测、训练日志与模型权重继续保留在本地，未随本次归档上传。仓库中的摘要会引用这些本地产物及其 SHA-256；完整重跑或独立复算仍需要报告中指定的数据和缓存。此前已跟踪的历史文件保持原状。

本次归档使用独立研究分支，保留主分支及其他工作区的未提交改动。归档本身不新增训练、不重新选择参数，也不改写已锁定结果。

## 上传前检查

归档代码对应的 28 份新增测试文件合计 212 项测试全部通过（113.91 秒）；现有 NumPy、pandas、Keras 依赖产生弃用警告，无测试失败。归档包含 272 个新增文件，约 5 MB。原始文件内容及冻结哈希保持不变，保留已有 Markdown 换行空格和末尾空行。
