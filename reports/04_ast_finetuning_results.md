# 标准 AST 微调实验报告

> 日期：2026-08-26
> 状态：完成 4 折 GPU pilot；用于路线决策，不覆盖既有锁定 baseline

## 1. 结论摘要

本轮在同一套 111-cat、792-call、cat-ID-disjoint outer folds 下，比较了三种
standard AST 训练状态：冻结 encoder、只解冻最后两个 Transformer blocks、完整
解冻 encoder。

结果如下：

| 状态 | Animal macro F1 | Balanced accuracy | QWK | 正确猫数 |
|---|---:|---:|---:|---:|
| Frozen AST + MLP | 0.7260 | **0.7696** | **0.6791** | 81 / 111 |
| Last-2 blocks fine-tuning | **0.7338** | 0.7563 | 0.6592 | 81 / 111 |
| Full fine-tuning | 0.7251 | 0.7306 | 0.6095 | 81 / 111 |

核心信息不是“某一个方案全面胜出”，而是三种方案判对的猫数完全相同，错误在
三个年龄类别之间的分配不同：

- `last2` 的 macro F1 最高，比本轮 Frozen 内部对照高 0.0078。它改善了 kitten
  的精确率和 adult 的召回，但牺牲了 senior 召回，因此 balanced accuracy 和
  QWK 没有同时提高。
- Frozen 的 balanced accuracy 与 QWK 最好，说明三个类别的召回更均衡，且在
  年龄有序关系上整体错误较轻。
- Full fine-tuning 的 adult 召回最高，但 senior 召回最低，QWK 也最低；训练损失
  很快下降而验证损失上升，呈现清楚的小样本过拟合。

因此，本轮对 fine-tuning 的直接判断是：**有限解冻有轻微、但不稳定且未达到
实践阈值的方向性信号；完整解冻没有总体收益。** 如果只保留一个微调候选，
`last2` 比 full 更合理，但它不取代 VGGish 三分类 baseline，也不需要包装成新的
独立 baseline。

## 2. 实验问题与边界

本轮回答的问题是：在标准 AST geometry 已确定后，更新 encoder 权重是否比只训练
MLP head 更有帮助。

三种状态为：

1. **Frozen**：直接读取冻结 AST 的 768 维 call embedding，只训练 MLP head。
2. **Last-2**：训练第 11、12 个 Transformer block、最终 LayerNorm 和 MLP head；
   其余 AST 参数冻结。
3. **Full**：训练完整 AST encoder 和 MLP head。

三者共享：

- MIT AudioSet AST checkpoint 与 standard `10 × 10` stride geometry；
- 每段 1.28 秒、0.64 秒 hop 的 fbank 输入；
- 先对同一 call 的 segment `pooler_output` 做算术平均，再进入 MLP；
- 固定的四个 cat-ID outer folds，以及每折固定 inner validation cats；
- 768 → 128、ReLU、BatchNorm、Dropout、3-class output 的分类头；
- 每折只用 outer-training cats 的冻结 AST embedding 拟合标准化器，微调过程中保持
  该标准化器固定；
- animal-level 概率聚合、macro F1 主指标及相同 seed 规则。

训练使用 RTX 4060 Ti、PyTorch 2.2.2+cu121、FP16，encoder learning rate 为
`1e-5`，head learning rate 为 `0.003109800273709165`，Adamax，micro-batch 8，
gradient accumulation 4，最多 50 epochs，`val_loss` patience 8。最佳 epoch 在
inner validation 上选择，再使用 outer train + validation cats 从头训练相同轮数，
最后仅在 outer-test cats 上评价。

## 3. 训练规模与运行情况

| 状态 | 可训练参数 | 占该模型总参数 | 四折训练与预测时间 | 峰值显存 |
|---|---:|---:|---:|---:|
| Frozen | 99,075 | 100%（仅 head） | 9.5 秒 | 0.02 GiB |
| Last-2 | 14,276,355 | 16.7% | 80.0 秒 | 0.69 GiB |
| Full | 85,466,115 | 100% | 164.4 秒 | 2.31 GiB |

各折由验证损失选出的最佳 epoch：

| 状态 | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|---:|
| Frozen | 3 | 11 | 1 | 2 |
| Last-2 | 3 | 7 | 1 | 2 |
| Full | 2 | 6 | 1 | 2 |

Full 在多个 fold 中训练损失持续接近零、验证损失却快速升高。例如 fold 2 的验证
损失从 epoch 1 的 1.183 上升到 epoch 9 的 2.918；这不是 GPU 故障，而是大参数量
模型在当前小数据条件下拟合训练集快于获得可迁移模式。

## 4. 四折表现

| 状态 | Fold 0 F1 | Fold 1 F1 | Fold 2 F1 | Fold 3 F1 | 合并 111 cats F1 |
|---|---:|---:|---:|---:|---:|
| Frozen | **0.8125** | **0.6537** | 0.8100 | 0.6251 | 0.7260 |
| Last-2 | 0.7737 | 0.5907 | **0.8472** | 0.7122 | **0.7338** |
| Full | 0.7737 | 0.6013 | 0.7856 | **0.7341** | 0.7251 |

`last2` 相对 Frozen 在 fold 2、3 上提高，在 fold 0、1 上下降；Full 只在 fold 3
提高。因而 last2 的总体领先来自类别错误结构和部分 folds，并非四折方向一致。

## 5. 类别层面的具体变化

### 5.1 每类召回

| 状态 | Kitten recall | Adult recall | Senior recall |
|---|---:|---:|---:|
| Frozen | **0.8667** | 0.6774 | **0.7647** |
| Last-2 | **0.8667** | 0.7258 | 0.6765 |
| Full | 0.8000 | **0.7742** | 0.6176 |

从 Frozen 到 last2，adult 多识别对 3 只，但 senior 少识别对 3 只；总正确数因此
不变。Last-2 同时减少了被错误预测成 kitten 的 adult，使 kitten precision 从
约 0.619 提高到约 0.722，进而推高 kitten F1 和总体 macro F1。

Full 继续把预测重心推向 adult：adult 识别对 48/62，是三者最高；但 senior 只
识别对 21/34，并出现 kitten 与 senior 之间的双向跨级错误。因此它的 adult F1
较好，balanced accuracy 与 QWK 却下降。

### 5.2 混淆矩阵

行是真实类别，列是预测类别，顺序均为 `kitten / adult / senior`。

Frozen：

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 13 | 2 | 0 |
| Adult | 7 | 42 | 13 |
| Senior | 1 | 7 | 26 |

Last-2：

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 13 | 2 | 0 |
| Adult | 4 | 45 | 13 |
| Senior | 1 | 10 | 23 |

Full：

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 12 | 2 | 1 |
| Adult | 3 | 48 | 11 |
| Senior | 1 | 12 | 21 |

这解释了三个指标为何给出不同排序：

- **Macro F1** 同时考虑每类 precision 与 recall。Last-2 减少了 kitten 的假阳性，
  并提高 adult 的正确识别，因此尽管 senior 变弱，平均 F1 仍略高。
- **Balanced accuracy** 是三类 recall 的平均。Frozen 对 senior 的保留更好，所以
  获得最高值。
- **QWK** 把年龄看作有顺序的标签，对 kitten ↔ senior 的跨两级错误惩罚比相邻
  年龄错误更重。Frozen 的年龄顺序一致性最好；Full 出现更多跨级和 senior→adult
  错误，因此 QWK 最低。

## 6. 配对差异与不确定性

在 111 只外测猫上进行 2,000 次、按真实类别分层的 paired animal bootstrap：

| 对比 | Macro F1 点差 | 95% bootstrap 区间 |
|---|---:|---:|
| Last-2 − Frozen | +0.0078 | [-0.0349, +0.0522] |
| Full − Frozen | -0.0009 | [-0.0707, +0.0622] |
| Full − Last-2 | -0.0087 | [-0.0644, +0.0396] |

预先锁定的实践差异阈值是 0.03。Last-2 的 +0.0078 没有越过该阈值，区间也跨过
零；因此它是可继续利用的轻微信号，不是已经确认的稳定提升。Full 与 Frozen 几乎
相同，且 secondary metrics 更弱，本轮没有理由继续优先扩大 full fine-tuning。

## 7. 与既有锁定 baseline 的关系

既有 TensorFlow 锁定实验仍是历史正式结果：

| 既有 pipeline | Animal macro F1 |
|---|---:|
| VGGish + MLP 三分类 baseline | 0.6846 |
| Frozen standard AST + MLP | 0.6525 |

本轮 PyTorch Frozen 内部对照达到 0.7260，明显高于旧 Frozen AST。这里改变的不只是
是否更新 encoder，还包括训练框架、micro-batch、梯度累积、早停预算与 BatchNorm
的实际 batch statistics。因此，这个差值首先说明**分类头训练 recipe 本身可能是
重要变量**，不能把 0.6525→0.7338 全部写成 last2 fine-tuning 的贡献。

不过，这并不使本轮结果失去价值：三种新状态在同一 PyTorch recipe 下直接比较，
已经回答了 encoder 更新的增量问题——last2 只比 Frozen 高 0.8 个百分点，full
没有提高。若之后要正式声称 AST 超过 0.6846 的 VGGish baseline，应让 VGGish 在
同一新 head recipe 下补跑一次公平对照；这只需要补一个对照，不需要建立两个
baseline。

## 8. 路线判断与下一步

按 Main Route v2，本轮可记为：standard AST 的 encoder fine-tuning 没有形成超过
`δ_task` 的稳定增益，完整解冻尤其不值得继续投入。路线继续按原先顺序转向
**IDEA-013：冻结 backbone，只学习 temporal pooling**。

IDEA-013 与本轮不同：本轮改变 encoder 是否更新，但仍把同一 call 的 segment
简单平均；IDEA-013 保持 encoder 不动，直接检验“平均聚合是否丢掉了少量关键时间
片段”。最小下一轮建议为：

1. 以 standard AST frozen segment embeddings 为主，先比较 `mean pooling` 与一个
   预先固定的 `single-head gated attention pooling`；
2. 保持相同 folds、animal-level 指标和 MLP 容量，加入 capacity-matched mean +
   MLP control；
3. 记录 attention 对有效 vocal segment、padding 和 call duration 的关系；
4. 若 AST 内部显示正向信号，再决定是否用 VGGish segment sequence 做 transport
   check；
5. IDEA-003 ordinal objective 与 IDEA-019 PEFT placement 继续后置，一次只加入
   一个模块。

论文层面的当前表述可以是：**AST 的标准冻结表示已经有可用年龄信息；有限解冻只
改变了类别权衡，完整解冻在小样本上过拟合。下一阶段优先定位 aggregation
bottleneck，而不是继续增加 encoder 可训练参数。**

## 9. 产物与复核

- 正式汇总：`runs/ast_finetuning_v1/summary.json`
- 三模式 call 与 animal predictions：`runs/ast_finetuning_v1/`
- 执行脚本：`scripts/run_ast_finetuning.py`
- 机器可读归档：`metadata/experiments/meowagenet_ast_finetuning_v1_results.json`
- smoke test：`runs/ast_finetuning_smoke_2026-08-26/`，仅用于验链，不进入表格

已核验每种模式均有 792 个唯一 call、111 个唯一 cat-ID；每只猫只出现在一个
outer-test fold；三类概率和为 1；协议、roles、fbank 与冻结 embedding 哈希均已
写入正式 summary。
