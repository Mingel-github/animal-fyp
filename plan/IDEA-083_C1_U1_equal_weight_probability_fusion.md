# IDEA-083：C1 / U1 固定等权概率融合分析计划

## 问题与范围

本轮只回答一个事后探索性问题：IDEA-076 中参数匹配的 C1 与 U1 是否具有足够互补的错误与概率质量，使一个**事前固定、不可调参**的等权概率平均同时接近 U1 的分类表现和 C1 的概率质量。

主分析唯一使用 IDEA-076 已保存的 inner-validation predictions：6 个 base seeds × 3 repeats × 4 folds，共 18 个 seed×repeat 单元、72 个 paired folds。A0、C1、U1 均只读，不训练、不调用 GPU、不生成或读取 outer-test 预测。IDEA-072 与 IDEA-073 只作两个分开标记的历史敏感性分析；不得根据其结果选择轮次、权重或主结论。

## 锁定融合

对每条 call 的三类概率固定计算：

```text
pE = 0.5 * pC1 + 0.5 * pU1
```

随后按 `cat_id` 对 call 概率作算术平均。权重固定为 `0.5 / 0.5`，不搜索、不按 seed、repeat、fold、类别或猫改变。必须同时从已保存的 C1/U1 animal probabilities 直接等权平均，并验证它与“call 融合后再猫均值”数值一致。

## 强制审计

1. 每个 pipeline 的 fit 必须是 complete，且 run / base seed / full seed / repeat / fold 身份一致、`outer_test_accessed=false`。
2. 逐 fit 验证 call 与 animal prediction 文件的 SHA-256；路径必须是 validation 文件且不得包含 outer-test。
3. A0/C1/U1 的 call 顺序、`call_index`、`call_id`、`cat_id`、`true_label`、概率列顺序必须一致；animal 的 `cat_id`、label、call_count 与角色必须一致。
4. 概率必须有限、非负、和为 1；保存的 animal probabilities 必须能由 call predictions 独立重建。
5. 行匹配主键固定为 `base_seed / repeat / fold / cat_id`（call 表再加 `call_index / call_id`）。同一只猫可能在不同 folds 再次作为 validation occurrence，不能假设四折互斥；必须同时报告 animal-occurrence 数与实际 unique-cat 数。相同 repeat/fold 在不同 base seeds 的 validation cats 必须一致。
6. 对每个 seed×repeat 检查概率平均的凸性 sanity：`CE(E) ≤ [CE(C1)+CE(U1)]/2`，且 `Brier(E) ≤ [Brier(C1)+Brier(U1)]/2`（只作数值/实现核验，不要求 E 必胜任一 parent）。

## 输出口径

- 错误互补：C1 错/U1 对、U1 错/C1 对、都对、都错、不同预测、错误交并与重叠；同时给出 animal-occurrence 总表与 18 个 seed×repeat 分表。重复 occurrence 不能解释为独立猫。
- 性能：E、A0、C1、U1 的 seed×repeat 等权 Macro-F1、balanced accuracy、animal CE、Brier、senior recall；E 相对三参照的正/平/负与 12 个共同 split-cell 的最差值。
- 实际作用：E 相对 C1/U1 各纠正多少、损坏多少及净变化。
- Oracle：只报告“至少一名 parent 正确”的潜力上限计数/比例，不作为模型成绩，不进入模型排名。
- 结论必须标记为 post-hoc exploratory；不更改 IDEA-076 的预注册 gate，也不修改 Formal / Research Summary v3。

## 计划产物

- `scripts/analyze_idea083_C1_U1_equal_weight_fusion.py`
- `tests/test_idea083_C1_U1_equal_weight_fusion.py`
- `runs/meowagenet_idea083_C1_U1_equal_weight_fusion_v1/`
- `reports/61_IDEA-083_C1_U1_equal_weight_fusion_results.md`
- `metadata/experiments/meowagenet_idea083_C1_U1_equal_weight_fusion_v1_results.json`
