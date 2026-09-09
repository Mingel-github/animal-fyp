# AST 准确率增强：Cat balancing 与多层融合结果

## 结论摘要

本轮完成诊断、独立 protocol/runner、inner-only smoke、execution lock，以及
4 条 pipeline × 3 个 repeat × 4 折的 48 个 outer fit。每条 pipeline 均形成
3 组 complete OOF；每组覆盖全部 792 条 call 和 111 只猫。

**A1：final AST embedding + cat-balanced loss** 是本轮领先候选：

- animal-level macro F1：0.7488 → **0.7765**，提升 **0.0277**；
- Balanced Accuracy：0.7644 → **0.7864**，提升 **0.0221**；
- QWK：0.6469 → **0.6724**，提升 **0.0255**；
- 普通 animal accuracy：0.7447 → **0.7748**，提升 **0.0300**；
- 三个 repeat 的 macro F1 差值均为正：+0.0081、+0.0147、+0.0603。

Cat balancing 在 final representation 和 multi-layer representation 上都带来正收益。
全 12 层 scalar fusion 在两种 loss 下均低于对应 final-layer pipeline；本轮多层实现
完成消融使命，后续正式扩展聚焦 A0 与 A1。

## 1. 实验问题

本轮把两个机制拆开检验：

| Pipeline | 训练权重 | AST 表示 | 检验问题 |
| --- | --- | --- | --- |
| A0 | 按年龄类别平衡 call | final embedding | tuned AST 同轮对照 |
| A1 | 类别等权，且类别内每只猫等权 | final embedding | animal-level 训练目标对齐是否有收益 |
| A2 | 按年龄类别平衡 call | 12 层 learned scalar fusion | 多层表示本身是否有收益 |
| A3 | 类别内每只猫等权 | 12 层 learned scalar fusion | 两个机制组合后的效果 |

四组共用同一个 `768 → 128 → 3` 分类头、dropout 0.4457、Adamax、学习率
0.006、batch size 8、最多 50 epochs 和 patience 8。多层模型只增加 12 个
layer logits，softmax 后形成 12 个层权重；融合结果仍为 768 维，因此与 A0/A1
使用同一分类头。

## 2. Cat-balanced loss 做了什么

当前 A0 对三个年龄类别做 class balancing。同一类别内部，call 较多的猫会参与更多次
loss 计算。A1/A3 采用：

`每条 call 权重 = 训练 call 总数 / (3 × 该类别训练猫数 × 该猫训练 call 数)`

它同时实现两层平衡：三个类别各自获得相同的总 loss 权重；同一类别内每只猫也获得
相同的总权重。所有训练 call 继续参与训练。

repeat 0 / fold 0 的 inner-train 包含 517 条 call，实际权重如下：

| 猫 | 年龄类 | 训练 call 数 | 每条 call 权重 | 该猫总权重 |
| --- | --- | ---: | ---: | ---: |
| 046A | kitten | 45 | 0.4255 | 19.1481 |
| 041A | kitten | 1 | 19.1481 | 19.1481 |
| 000A | adult | 39 | 0.1227 | 4.7870 |
| 004A | adult | 1 | 4.7870 | 4.7870 |
| 103A | senior | 33 | 0.2487 | 8.2063 |
| 024A | senior | 1 | 8.2063 | 8.2063 |

例如，kitten 046A 的 45 条 call 共同获得与 kitten 041A 单条 call 相同的总权重；
adult 000A 与 004A、senior 103A 与 024A 也分别遵循同样规则。不同年龄类别中的
单猫总权重不同，因为各类别的训练猫数量不同；每个类别的总权重保持相同。

## 3. 训练前诊断

### 3.1 call 数、时长与 padding

使用既有 tuned AST 的三组 seed-17 OOF 结果进行描述性诊断：

| 每猫 call 数 | 猫数 | 三次平均正确率 | 平均置信度 | 平均 padding 比例 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20 | 0.8000 | 0.8434 | 0.4614 |
| 2–5 | 42 | 0.6746 | 0.7421 | 0.4642 |
| 6–10 | 25 | 0.7867 | 0.7160 | 0.4637 |
| 11–20 | 19 | 0.7895 | 0.7000 | 0.4679 |
| 21+ | 5 | 0.7333 | 0.6467 | 0.4713 |

Spearman 相关结果：

- call 数与正确率：ρ = 0.087；
- call 数与置信度：ρ = -0.310；
- 平均时长与正确率：ρ = -0.117；
- 平均 padding 比例与正确率：ρ = 0.082。

正确率随 call 数、时长和 padding 均没有呈现稳定的单调变化。Cat balancing 的主要
依据因此是“训练贡献单位与 animal-level 评价单位对齐”。call 数较多的猫不再在同一
年龄类别中占据更高的总梯度份额。

### 3.2 模型错误互补

以 repeat 0 为例，tuned AST 与 VGGish 同时正确 72 只，AST 单独正确 14 只，
VGGish 单独正确 8 只，同时错误 17 只。tuned AST 与 PANNs CNN14 同时正确 61 只，
AST 单独正确 25 只，PANNs 单独正确 6 只，同时错误 19 只。其余两个 repeat 也保留
一部分独占正确样本。这为后续 cross-model probability fusion 留下依据；本轮核心矩阵
先完成 AST 内部的训练权重与表示消融。

### 3.3 Inner-only layer probes

既有 12 个 inner-only probe 的平均排名前四层为第 8、12、11、10 层。第 11 层进入
top-two 的次数最多，为 8/12；第 12 层为 6/12，第 8 层为 4/12。不同 fold 选择的层
存在变化，说明 frozen AST 的中后层确实携带不同的可用信号，也说明简单地平均全部
12 层会同时引入 probe 较弱的早期层。

## 4. Smoke 与执行锁

Smoke 使用 repeat 0 / fold 0 / seed 17，只读取 inner train 和 inner validation：

| Pipeline | 最佳 epoch | Inner macro F1 | Inner validation loss |
| --- | ---: | ---: | ---: |
| A0 | 7 | 0.7474 | 0.7025 |
| A1 | 5 | 0.7474 | 0.6989 |
| A2 | 5 | 0.7889 | 0.4638 |
| A3 | 7 | 0.7474 | 0.5488 |

A0 与既有 tuned AST 在该 fold 上完全复现：最佳 epoch、macro F1 和 validation loss
逐项一致。随后写入 hash-bound execution lock，再开始 outer evaluation。A2 在单个
inner fold 上的优势说明 smoke 的作用是验证数据流与训练信号；complete OOF 决定
最终阶段判断。

## 5. Complete OOF 主结果

| Pipeline | Macro F1，mean ± SD | Balanced Accuracy | QWK | 普通 accuracy |
| --- | ---: | ---: | ---: | ---: |
| A0 final + class balance | 0.7488 ± 0.0288 | 0.7644 | 0.6469 | 0.7447 |
| **A1 final + cat balance** | **0.7765 ± 0.0212** | **0.7864** | **0.6724** | **0.7748** |
| A2 scalar fusion + class balance | 0.7027 ± 0.0321 | 0.7272 | 0.6025 | 0.6997 |
| A3 scalar fusion + cat balance | 0.7230 ± 0.0226 | 0.7476 | 0.6171 | 0.7087 |

每个 repeat 的 primary macro F1：

| Repeat | A0 | A1 | A1 − A0 | A2 | A3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7816 | 0.7897 | +0.0081 | 0.7340 | 0.7446 |
| 1 | 0.7373 | 0.7520 | +0.0147 | 0.7043 | 0.7249 |
| 2 | 0.7275 | 0.7878 | +0.0603 | 0.6698 | 0.6995 |
| Mean | 0.7488 | **0.7765** | **+0.0277** | 0.7027 | 0.7230 |

结果形成两个清晰的主效应：

1. Cat balancing 相对 A0 在三个 repeat 全部提升；在 scalar fusion 表示上，A3 相对
   A2 也分别提升 +0.0106、+0.0206、+0.0297，均值 +0.0203。
2. 当前全 12 层 scalar fusion 相对 final representation 持续下降：A2 相对 A0 均值
   -0.0461，A3 相对 A1 均值 -0.0534。

因此，cat balancing 的正信号跨两种表示成立；全层 scalar fusion 的负信号也跨两种
loss 成立。

## 6. 各年龄类别发生了什么

| Pipeline | Kitten recall | Adult recall | Senior recall |
| --- | ---: | ---: | ---: |
| A0 | 0.8222 | 0.7258 | 0.7451 |
| A1 | **0.8444** | **0.7796** | 0.7353 |
| A1 − A0 | +0.0222 | **+0.0538** | -0.0098 |

A1 的最大收益来自 adult：三个 repeat 中，adult 被正确识别的平均比例提高 5.38 个
百分点。三组 confusion matrix 合计后，adult → senior 从 A0 的 43 次降至 A1 的
35 次，adult → kitten 从 8 次降至 6 次。Kitten recall 同时提高 2.22 个百分点；
senior recall 小幅变化 -0.98 个百分点。

Balanced Accuracy 同步提高 0.0221，说明三类平均识别能力整体上升。QWK 提高
0.0255，说明预测与 kitten–adult–senior 的年龄顺序在总体上更加一致。Macro F1 的
sample SD 从 0.0288 降至 0.0212，本轮三个 split repeat 的离散程度也更小。

## 7. 逐猫配对变化

A1 相对 A0 改变预测的猫数分别为 6、16、12 只：

| Repeat | 改正 | 改错 | 两者都错但类别改变 | 净增加正确猫数 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 2 | 1 | +1 |
| 1 | 8 | 6 | 2 | +2 |
| 2 | 9 | 2 | 1 | +7 |

对应的正确猫数为 86 → 87、81 → 83、81 → 88。平均每个 repeat 多正确 3.33 只猫。

实际例子包括：

- adult 027A 有 7 条 call，在 repeat 0 和 1 均由 A0 的 senior 修正为 adult；
- adult 073A 只有 1 条 call，在 repeat 1 和 2 均由 senior 修正为 adult；
- repeat 2 中，kitten 046A 有 45 条 call，由 adult 修正为 kitten；
- repeat 2 中，adult 026B 和 100A 各只有 1 条 call，分别由 senior/kitten 修正为 adult；
- senior 054A 有 2 条 call，在 repeat 1 和 2 由 A0 的 senior 变为 A1 的 adult。

这些变化同时覆盖 call 很多与 call 很少的猫。A1 的收益来自训练贡献重新分配后的整体
decision boundary 调整，其中 adult 区域获得了最明显改善。

## 8. 为什么本轮 scalar fusion 下降

12 个 softmax layer weights 从 1/12 = 0.0833 均匀初始化。完成 outer retrain 后，
A2/A3 的跨折平均权重仍集中在约 0.0821–0.0852，实际状态接近“十二层近似平均”。
这会把第 1–5 层等 probe 较弱表示与第 8、10、11、12 层一起混合。分类头只能在少量
epoch 内适应新的混合空间，最终 A2/A3 在 adult 和 senior recall 上损失较多。

这个结果同时提供了后续设计信息：多层信息的价值更适合通过 inner-only 选择层、
更强的 layer-attention 参数化或 final-layer residual 初始化来研究。当前全层均匀起始
scalar fusion 在本阶段收尾，论文中可以作为结构消融报告。

## 9. 阶段决策与下一步

本轮预设的扩展条件为：候选相对 A0 的 mean macro F1 至少提高 0.01，并在三个
repeat 中至少两个为正。A1 达到 +0.0277 且 3/3 repeats 为正，条件已经满足。

阶段决策如下：

- 将 **A1 final AST + cat-balanced loss** 保留为 provisional AST performance candidate；
- 将 A0 作为 matched tuned AST 对照；
- A2/A3 作为完成的 multi-layer ablation 保留；
- 下一阶段只扩展 A0 与 A1 到 base seed 43 和 101，形成更完整的配对重复实验；
- 论文当前可以写成“seed-17 三组 complete OOF 上获得一致的阶段性提升”，扩展 seeds
  将决定最终结果表中的稳定性表述。

## 10. 文件与审计

- 可执行 protocol：`configs/protocol/meowagenet_ast_accuracy_enhancement_v1.json`；
- 独立 runner：`scripts/run_meowagenet_ast_accuracy_enhancement_v1.py`；
- 机器可读结果：
  `metadata/experiments/meowagenet_ast_accuracy_enhancement_v1_results.json`；
- 完整运行目录：`runs/meowagenet_ast_accuracy_enhancement_v1/`；
- diagnostics：`runs/meowagenet_ast_accuracy_enhancement_v1/diagnostics/`；
- smoke 与 execution lock：`runs/meowagenet_ast_accuracy_enhancement_v1/smoke/`、
  `runs/meowagenet_ast_accuracy_enhancement_v1/execution_lock.json`；
- 48 个 outer fit summaries、12 个 complete-OOF animal tables、逐猫 paired changes 和
  scalar layer weights：`runs/meowagenet_ast_accuracy_enhancement_v1/evaluation/`。

运行目录共 123 个文件、1,536,512 bytes；其中包含 4 个 smoke fit summary、48 个
outer fit summary 和 12 个 complete-OOF animal prediction table。A0 的三组结果与既有
tuned AST 逐组完全一致。GitHub 版本记录 60 个 compact JSON 日志（503,878 bytes）；
逐 call 和逐猫 prediction CSV 按仓库数据策略保留在本机，报告和机器结果 JSON 已包含
完整汇总表、混淆矩阵、逐 repeat 指标与配对差值。
