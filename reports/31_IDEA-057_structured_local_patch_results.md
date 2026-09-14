# IDEA-057｜AST 结构化局部 patch 分支结果

> 执行日期：2026-09-15  
> 状态：首轮独立验证完成，seed expansion gate 关闭  
> 锁定代码提交：`e63d9d3ecf176ba7b71c4bd8e9c9b4be02b0bee5`

## 1. 本轮回答的问题

AST 内部诊断显示，中频区域保留了较强年龄分类信息；中段时间区域被替换后，真实类别概率在
12/12 个 inner splits 中下降。IDEA-057 将这条诊断线索实现为轻量、位置感知的局部分支，检验
它能否在 final AST global embedding 之外稳定提高 unseen-cat 分类。

本轮设置三组同期配对流程：

- `R0_tuned_frozen_ast`：锁定 AST 全局 embedding 与 `768 → 128 → 3` 分类头；
- `M1_position_aware_local_patch`：R0 加中频带的前、中、后三个相对时间位置；
- `C1_position_removed_control`：与 M1 的参数和网络完全相同，将九格均值重复到三个输入位置，
  用于分离位置结构与新增容量的作用。

三组均使用 animal-ID-disjoint splits、全局类别平衡 call 权重、animal-level validation CE
checkpoint selection，以及同猫 call probability 算术平均。

## 2. 特征与候选选择

冻结 AST 为 792 条 call、843 个 fbank segment 生成 `3 × 3 × 768` 相对时间—频率网格。
每一格先在 segment 内聚合 final-layer patch tokens，再在同一 call 的 segments 间取平均。
重新计算的 global embedding 与历史锁定缓存最大绝对差为 `1.93 × 10⁻⁵`。

候选选择只使用 12 组 inner-training / inner-validation roles：

| 候选 | 读取范围 | Inner animal Macro F1 | Inner animal CE | 参数量 |
| --- | --- | ---: | ---: | ---: |
| `middle_frequency_strip` | 中频带 × 前/中/后三段 | **0.7514** | **0.7138** | 136,227 |
| `full_3x3_grid` | 全部九格 | 0.7213 | 0.7387 | 130,067 |

因此 execution lock 选择 `middle_frequency_strip`。该选择与诊断中的中频线索一致，同时完整
九格带来的更高输入范围没有转化为更好的 inner 表现。

## 3. Smoke 与执行审计

- R0 可训练参数：`99,075`；
- M1 与 C1 可训练参数：均为 `136,227`；
- M1/C1 局部分支初始参数逐项一致；
- 零初始化 gate 使 R0/M1/C1 初始 logits 最大差为 `0`；
- 三组 checkpoint 重载后的概率最大差均为 `0`；
- smoke 与候选选择均未访问 outer-test；
- execution lock 在 36 个 outer fits 前写入。

## 4. Complete-OOF 结果

每组完成 3 repeats × 4 folds。每个 repeat 拼接四折后包含全部 111 只猫一次。

| Pipeline | Animal Macro F1 | Balanced accuracy | QWK | Accuracy | Animal CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| R0 tuned frozen AST | **0.7570 ± 0.0086** | **0.7645** | **0.6721** | **0.7568** | **0.7160** |
| M1 position-aware local patch | 0.7265 ± 0.0341 | 0.7418 | 0.6429 | 0.7237 | 0.7177 |
| C1 position-removed control | 0.7348 ± 0.0147 | 0.7475 | 0.6509 | 0.7357 | 0.7364 |

M1 相对 R0 的 repeat-level Macro F1 差值为：

- repeat 0：`0.0000`；
- repeat 1：`−0.0496`；
- repeat 2：`−0.0419`；
- 平均：`−0.0305`，正向 repeat 数为 `0/3`。

配对 cat-cluster bootstrap 的 M1−R0 Macro F1 区间为
`[−0.0579, −0.0058]`。在本轮 111-cat repeated grouped evaluation 中，R0 的优势覆盖了该
区间。

M1 相对 C1 的平均 Macro F1 差为 `−0.0083`，三个 repeat 中一个为正。C1 相对 R0 也下降
`−0.0222`。因此两层信息同时成立：新增局部分支容量整体降低了本轮性能；保留三个中频时间
位置也没有优于参数匹配的去位置输入。

## 5. 类别与稳定性信息

| Pipeline | Kitten recall | Adult recall | Senior recall |
| --- | ---: | ---: | ---: |
| R0 | 0.8000 | **0.7581** | **0.7353** |
| M1 | 0.8000 | 0.7097 | 0.7157 |
| C1 | 0.8000 | 0.7366 | 0.7059 |

M1 保持了 Kitten recall，Adult recall 下降约 `0.0484`，Senior recall 下降约 `0.0196`。
Macro F1 的 repeat SD 从 R0 的 `0.0086` 增加到 M1 的 `0.0341`，说明该局部分支主要增加了
split 敏感性。M1 animal CE 仅比 R0 高 `0.0017`，概率质量总体接近；argmax 决策及类别平衡
表现形成了更明显差距。

训练后局部 gate 在各 fold 中均形成活跃权重。它证明优化器使用了局部分支，同时也说明
“gate 学到非零值”与“该信息提高 unseen-cat 分类”是两个层面的结果。

## 6. 阶段结论

本轮支持以下三条论文级结论：

1. 中频相对时间结构在 inner-only 候选选择中优于完整九格读取，说明诊断定位对结构收缩具有
   实际价值；
2. 当前 `中频三段 → 有序拼接 → 零门控残差` 参数化降低 complete-OOF Macro F1，并扩大
   repeat 波动；
3. 参数量匹配的去位置 C1 平均高于 M1，当前结果更支持“局部信息与全局 embedding 冗余，
   新分支增加优化干扰”的解释。

IDEA-057 的所有预设 seed-expansion 条件均已完成核验，扩展条件关闭。该实现作为结构化局部
patch 与参数匹配机制消融保留；双方向计划下一步独立执行 IDEA-058 受约束顶层适配。

## 7. 汇总阶段记录

锁定 runner 已完成全部 36 个 fit、checkpoint 选择和原始 outer predictions。最终汇总阶段复用
IDEA-052 代码时需要 animal-level `mean_tokens_per_call` 字段；IDEA-057 的 call-level 文件已
保存固定值 `temporal_token_count=9`。独立后处理器从该字段重建派生列，并映射旧汇总器的 gate
键名。后处理过程核验 36 份 fit summary，保持模型训练、候选选择和原始预测逐字节不变。

## 8. 关键文件

- 计划：`plan/IDEA-057_structured_local_patch_branch.md`；
- Protocol：`configs/protocol/meowagenet_idea057_structured_local_patch_v1.json`；
- 锁定 runner：`scripts/run_meowagenet_idea057_structured_local_patch.py`；
- 只读后处理器：`scripts/summarize_meowagenet_idea057.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_idea057_structured_local_patch_v1_results.json`；
- 执行锁：`runs/meowagenet_idea057_structured_local_patch_v1/execution_lock.json`；
- 原始预测清单聚合 SHA-256：
  `f1986b1c971bf2ad6f67d81870035278b9c5c0248395e59d63f16d02937c490a`。
