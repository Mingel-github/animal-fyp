# AST Cat Balancing 多 Seed 扩展结果

## 结论摘要

本轮按锁定计划完成 A0 tuned frozen AST 与 A1 cat-balanced AST 的 base seed 43、101
扩展。新增实验包含 2 条 pipeline × 2 个 base seed × 3 个 repeat × 4 折，共 48 个
outer fit；每条 pipeline 新增 6 组 complete OOF。连同 seed 17 的三组结果，最终比较
覆盖 3 个 base seed、9 组配对 complete OOF，每组均覆盖 792 条 call 和 111 只猫。

预设阶段门槛已经达到：A1 − A0 的平均 macro F1 为 **+0.0031**，9 组中 **6 组为正**。
这个结果属于边界通过，呈现“整体接近、优势分布不同、A1 波动更小”的结构：

- A1 的 macro F1 为 **0.7416 ± 0.0343**，A0 为 0.7385 ± 0.0440；
- A1 的普通 accuracy 为 **0.7317**，A0 为 0.7267，相当于 999 次猫级评估中净多
  判对 5 次；
- A0 的 Balanced Accuracy 为 **0.7595**，A1 为 0.7587，差值 0.0009；
- A0 的 QWK 为 **0.6422**，A1 为 0.6397，差值 0.0025；
- A1 的 adult recall 提高 0.0179，A0 在 kitten 与 senior recall 上分别高 0.0074、
  0.0131；
- A1 的 macro-F1 sample SD 比 A0 小 0.0097，最低一组 macro F1 从 0.6716 提高到
  0.6889。

因此，A1 按预设规则保留为 confirmed post-formal performance candidate；A0 继续作为
matched tuned-AST reference。当前结论支持阶段性收尾，并为后续新 idea 与最终重复实验
保留模型选择空间。

## 1. 本轮具体比较什么

两条 pipeline 共用以下内容：

- 同一批 792 条 call、111 只猫和 kitten/adult/senior 三分类标签；
- 同一套 animal-independent nested splits；
- 同一组 base seed、repeat 与 fold；
- 同一 frozen AST final embedding；
- 同一个 `768 → 128 → 3` 分类头；
- dropout 0.4457、Adamax、学习率 0.006、batch size 8、最多 50 epochs、patience 8；
- 每只猫的多条 call 概率取算术平均，再产生 animal-level 预测。

两者的实验变量只有训练 loss 权重：

| Pipeline | 每条训练 call 的权重 | 作用 |
| --- | --- | --- |
| A0 | 按年龄类别的 call 数做 class balance | 三个年龄类获得相近总权重 |
| A1 | `N / (3 × 该类训练猫数 × 该猫训练 call 数)` | 三类等权，且同类中每只猫等权 |

例如 adult 000A 有 39 条训练 call、adult 004A 只有 1 条时，A1 会降低 000A 单条 call
的权重、提高 004A 单条 call 的权重，使两只 adult 猫对该类别 loss 的总贡献一致。全部
call 继续参与训练，变化发生在梯度贡献大小。

## 2. Smoke、锁定与正式运行

Smoke 使用 base seed 43、repeat 0、fold 0，只访问 inner train/validation：

| Pipeline | 最佳 epoch | Inner macro F1 | Inner validation loss |
| --- | ---: | ---: | ---: |
| A0 | 7 | 0.7474 | 0.7580 |
| A1 | 3 | 0.7474 | 0.6960 |

Smoke 通过后写入 hash-bound execution lock，再启动 outer evaluation。runner 与 protocol
的哈希值随后保持固定。正式运行完成 48/48 outer fits，形成 12 份新的 complete OOF。

## 3. 九组主结果

| Base seed | Repeat | A0 macro F1 | A1 macro F1 | A1 − A0 |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 0 | 0.7816 | **0.7897** | +0.0081 |
| 17 | 1 | 0.7373 | **0.7520** | +0.0147 |
| 17 | 2 | 0.7275 | **0.7878** | +0.0603 |
| 43 | 0 | **0.8020** | 0.7305 | -0.0715 |
| 43 | 1 | 0.7330 | **0.7539** | +0.0209 |
| 43 | 2 | 0.6790 | **0.7056** | +0.0266 |
| 101 | 0 | **0.7382** | 0.7198 | -0.0184 |
| 101 | 1 | **0.7761** | 0.7465 | -0.0296 |
| 101 | 2 | 0.6716 | **0.6889** | +0.0173 |
| **九组平均** |  | 0.7385 | **0.7416** | **+0.0031** |

A1 在 seed 17 的三组全部胜出；seed 43 中 2/3 胜出；seed 101 中 1/3 胜出，合计
6/9。seed 43 repeat 0 的 A0 达到全矩阵最高值 0.8020，并形成 -0.0715 的单组差值；
这个结果抵消了 A1 在多数组合上的较小正增益。

按 base seed 汇总：

| Base seed | A0 macro F1 | A1 macro F1 | A1 − A0 | A1 胜出次数 |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 0.7488 | **0.7765** | +0.0277 | 3/3 |
| 43 | **0.7380** | 0.7300 | -0.0080 | 2/3 |
| 101 | **0.7287** | 0.7184 | -0.0103 | 1/3 |

seed 17 提供较强的 A1 增益；新增 seed 的均值各自偏向 A0。9 组联合结果把 A1 的平均
优势收敛到 0.0031，清楚地说明 cat balancing 的收益具有 seed/split 依赖性，同时在
多数配对组合中维持正方向。

## 4. 各指标告诉我们什么

| Pipeline | Macro F1，mean ± SD | Balanced Accuracy | QWK | 普通 accuracy |
| --- | ---: | ---: | ---: | ---: |
| A0 class-balanced | 0.7385 ± 0.0440 | **0.7595** | **0.6422** | 0.7267 |
| A1 cat-balanced | **0.7416 ± 0.0343** | 0.7587 | 0.6397 | **0.7317** |
| A1 − A0 | +0.0031 | -0.0009 | -0.0025 | +0.0050 |

Macro F1 先分别计算 kitten、adult、senior 的 F1，再取三类平均，因此同时考虑每类的
precision 和 recall。A1 在这个 primary metric 上小幅领先，并且 sample SD 从 0.0440
降到 0.0343，表明九组结果的离散程度更小。

Balanced Accuracy 是三类 recall 的平均值。A0 与 A1 的差距只有 0.0009，表示两者在
“三个年龄类平均识别比例”上几乎相同。

QWK 会根据年龄顺序惩罚错误：kitten 预测成 senior 的代价高于 kitten 预测成 adult。
A0 高 0.0025，表示 A0 的预测在年龄顺序一致性上有极小优势；这个差距与两者的整体
接近程度一致。

普通 accuracy 计算 111 只猫中预测正确的比例。A1 高 0.0050；累计 9 组即 999 次猫级
评估后，A1 净多判对 5 次。

## 5. 三个年龄类如何变化

| Pipeline | Kitten recall | Adult recall | Senior recall |
| --- | ---: | ---: | ---: |
| A0 | **0.8593** | 0.6971 | **0.7222** |
| A1 | 0.8519 | **0.7151** | 0.7092 |
| A1 − A0 | -0.0074 | **+0.0179** | -0.0131 |

九组 confusion matrix 累计后的正确数为：

- kitten：A0 116，A1 115；
- adult：A0 389，A1 399；
- senior：A0 221，A1 217。

A1 的主要收益继续集中在 adult。adult → senior 从 141 次降到 134 次，adult → kitten
从 28 次降到 25 次，adult 正确数增加 10。Kitten 正确数减少 1，senior 正确数减少 4，
最终净增加 5 次正确预测。

这个类别结构解释了指标分工：adult 的 precision/recall 改善推动 macro F1 和普通
accuracy 小幅上升；kitten/senior recall 的小幅变化使 Balanced Accuracy 与 QWK 略偏向
A0。两条 pipeline 各有清晰优势，整体性能处于同一水平区间。

## 6. 稳定性信息

A0 九组 macro F1 范围为 0.6716–0.8020；A1 为 0.6889–0.7897。A1 的最低值提高
0.0173，最高值降低 0.0123，sample SD 同时下降 0.0097。这体现出 cat balancing 在本轮
更像一种收缩波动的训练策略：它压低一个特别高的 A0 结果，也抬高部分较低结果。

新增 seed 43、101 单独汇总时，A0 macro F1 为 0.7333，A1 为 0.7242；两者差值
-0.0091。将 seed 17 纳入预先声明的 9 组联合判断后，A1 以 +0.0031 和 6/9 正方向达到
边界门槛。报告同时保留这两个视角：新增数据说明收益的 seed 依赖性，联合数据决定预设
阶段 gate。

## 7. 阶段决策

预设规则为：9 组配对的 A1 − A0 平均 macro F1 大于 0，并且至少 6/9 为正。实际结果
为 +0.003142 与 6/9，恰好达到两项边界。

阶段性角色安排：

- A1 保留为 **confirmed post-formal performance candidate**，含义是它通过了本阶段
  预先锁定的筛选规则；
- A0 保留为 **matched tuned-AST reference**，它在新增 seeds、Balanced Accuracy 和
  QWK 上体现出优势；
- 后续论文表述采用“小幅、混合 seed、边界确认”，并明确 A1 的 adult/accuracy/稳定性
  优势与 A0 的类别平均召回/年龄顺序优势；
- 本轮完成阶段性收尾，未来的新 idea、最终重复实验和最终模型选择继续开放。

## 8. 文件与审计

- Protocol：`configs/protocol/meowagenet_ast_cat_balance_seed_expansion_v1.json`；
- 独立 runner：`scripts/run_meowagenet_ast_cat_balance_seed_expansion_v1.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_ast_cat_balance_seed_expansion_v1_results.json`；
- 完整运行目录：`runs/meowagenet_ast_cat_balance_seed_expansion_v1/`；
- Smoke 与 execution lock：`runs/meowagenet_ast_cat_balance_seed_expansion_v1/smoke/`、
  `runs/meowagenet_ast_cat_balance_seed_expansion_v1/execution_lock.json`；
- 汇总：`runs/meowagenet_ast_cat_balance_seed_expansion_v1/evaluation/summary.json`。

运行目录包含 117 个文件、1,321,624 bytes：56 个 compact JSON 日志与 61 个本地 CSV。
GitHub 版本记录 protocol、runner、报告、机器结果和全部 56 个 JSON 审计日志；48 个
call-level outer prediction、12 个 animal-level complete OOF 和 1 个逐猫变化 CSV 保留在
本地研究环境。
