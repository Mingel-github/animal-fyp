# IDEA-051 Cat-level Set Aggregation 初始实验结果

## 结论摘要

本轮完成 3 条 pipeline × 3 个 repeat × 4 折，共 **36/36 个 outer fit**，形成 9 份
111-cat complete OOF。三条 pipeline 共享 frozen AST final embeddings、同一组 cat-ID
splits、同一组 model seeds、`768 → 128 → 3` 主体规模以及 animal-level checkpoint
selection；主要变量是训练单位和猫内 call 聚合方式。

| Pipeline | Animal macro F1，mean ± SD | Balanced accuracy | QWK | 普通 accuracy |
| --- | ---: | ---: | ---: | ---: |
| S0 call probability mean | **0.7570 ± 0.0086** | **0.7645** | **0.6721** | **0.7568** |
| S1 hidden mean set | 0.6777 ± 0.0226 | 0.7125 | 0.5678 | 0.6727 |
| S2 attention set | 0.6688 ± 0.0265 | 0.6915 | 0.5848 | 0.6727 |

S1 相对 S0 的三个 paired macro-F1 差值为 `−0.1102、−0.0799、−0.0479`，均值
`−0.0794`。S2 相对 S0 为 `−0.0665、−0.1014、−0.0969`，均值 `−0.0883`。
两个 set candidate 的三组 repeat 全部低于 matched S0，因此首轮 seed-expansion gate
关闭，base seeds 43/101 本轮无需执行。

这组结果支持三条具体结论：

1. 当前数据上，call-level 训练后平均同猫 call probabilities 的简单方案表现最好。
2. hidden mean 直接优化 animal-level loss 提高了 kitten recall，但 adult 与 senior 的综合
   识别下降，局部收益没有转化为总体指标收益。
3. attention scorer 确实学出了非均匀权重，并相对 S1 改善 QWK、减少跨两级年龄错误；
   它选中的 call 权重尚未形成更高的总体 macro F1。

## 1. 实验问题与三条 pipeline

本轮回答的问题是：每只猫拥有 1–45 条 call，最终又按整只猫评价时，模型是否应当从
训练阶段就把同猫 calls 当作一个 set。

### S0：call probability mean

每条 call 独立经过 frozen AST embedding 和 `768 → 128 → 3` head，训练 loss 按 call
计算；推理时先得到每条 call 的三个年龄概率，再对同一只猫的 probabilities 求均值。
S0 是本轮 matched reference。

### S1：hidden mean set

同一只猫的每条 768 维 embedding 先通过共享的 `768 → 128` encoder，再对所有 128 维
hidden representations 求均值，最后通过 `128 → 3` classifier。训练时每只猫对应一个
animal-level loss，所有猫在各自年龄类别内等权。

### S2：attention set

S2 与 S1 共享主体结构，并增加 129 个参数的 scalar attention scorer。scorer 为每条 call
生成一个分数，猫内 softmax 将这些分数转换为总和为 1 的权重，再进行加权 hidden
pooling。attention 参数从全零开始，所以训练起点与 S1 的 uniform mean 完全一致。

## 2. 数据依据与运行范围

数据包含 792 条 call、111 只猫：kitten 15 只、adult 62 只、senior 34 只。每只猫的
call 数最少 1、最大 45、中位数 5、平均 7.14；20 只猫只有 1 条 call，28 只猫至少有
10 条 call。

实验前诊断覆盖既有 C0 的 999 次猫级评估，其中 484 次（48.45%）包含猫内 call argmax
分歧，313 次（31.33%）至少四分之一 call 与最终猫级预测不同。该信号说明猫内声音存在
可利用的差异，也为 set aggregation 提供了直接的数据依据。

本轮使用 base seed 17、repeat 0/1/2、每个 repeat 四个 animal-ID-disjoint outer folds。
每条 pipeline 在每个 fold 内只用 inner train/validation 选择 epoch，outer predictions 在
epoch 锁定后生成。Smoke 覆盖 111 只猫、792 条 call、20 个 singleton sets 和最大 45-call
set，三条 checkpoint 重载后的最大概率差均为 0.0。

S1 与 S2 的 12 对 fold 共享完全相同的 cat batch order；所有共同训练 epoch 的顺序哈希
一致。正式运行在 RTX 4060 Ti 上约 1 分 40 秒完成，随后生成 complete-OOF、5,000 次
paired cat bootstrap、bag-size subgroup 和 attention 统计。

## 3. 三组 complete-OOF 主结果

| Repeat | S0 macro F1 | S1 macro F1 | S1 − S0 | S2 macro F1 | S2 − S0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **0.7659** | 0.6556 | −0.1102 | 0.6993 | −0.0665 |
| 1 | **0.7565** | 0.6766 | −0.0799 | 0.6551 | −0.1014 |
| 2 | **0.7487** | 0.7008 | −0.0479 | 0.6518 | −0.0969 |
| **平均** | **0.7570** | 0.6777 | **−0.0794** | 0.6688 | **−0.0883** |

S0 的三个 repeat 范围为 0.7487–0.7659，sample SD 为 0.0086。S1 的范围为
0.6556–0.7008，S2 为 0.6518–0.6993。S0 同时取得更高均值、更高最差 repeat 和更小的
repeat 间波动。

333 次逐猫评估中，S0 判断正确 252 次，S1 与 S2 各判断正确 224 次。配对变化进一步
展示差值来源：

- S1 相对 S0 新增 24 次正确，同时失去 52 次原本正确的判断，净变化 −28；
- S2 相对 S0 新增 19 次正确，同时失去 47 次原本正确的判断，净变化 −28；
- S2 与 S1 之间各自新增 23 次正确、失去 23 次正确，所以两者普通 accuracy 完全相同，
  具体判对的猫则有明显交换。

## 4. Macro F1、Balanced accuracy 与 QWK 各自说明什么

Macro F1 先分别计算 kitten、adult、senior 的 F1，再让三个类别等权平均。它同时受
precision 和 recall 影响，适合作为类别数量不均衡时的主指标。S0 比 S1 高 0.0794、比
S2 高 0.0883，说明 S0 对三个年龄类别的综合查准与查全最完整。

Balanced accuracy 是三个类别 recall 的平均。S0 为 0.7645，S1 为 0.7125，S2 为
0.6915。S1 的 kitten recall 略高，但 adult 与 senior recall 的下降使三类平均识别率仍由
S0 领先。

QWK 把年龄视为有顺序的类别，并让 kitten↔senior 这类跨两级错误承担更大代价。S2 的
QWK 0.5848 高于 S1 的 0.5678，尽管 S2 的 macro F1 略低。这说明 attention 相对 hidden
mean 调整了错误的距离结构：三组 repeat 累计跨两级错误从 S1 的 8 次降到 S2 的 5 次。
S0 只有 4 次跨两级错误，因此以 0.6721 保持最高 QWK。

普通 accuracy 统计预测正确的猫所占比例。S1 与 S2 都是 0.6727，说明二者累计正确数量
相同；macro F1 和 QWK 的差异进一步揭示了正确与错误在三个年龄类别中的分布差别。

## 5. 类别变化

| Pipeline | Kitten recall | Adult recall | Senior recall | Kitten F1 | Adult F1 | Senior F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 0.8000 | **0.7581** | **0.7353** | **0.7914** | **0.7853** | **0.6943** |
| S1 | **0.8222** | 0.6290 | 0.6863 | 0.7248 | 0.6981 | 0.6101 |
| S2 | 0.7333 | 0.6452 | 0.6961 | 0.6752 | 0.6970 | 0.6340 |

S1 多识别正确 1 次 kitten，使 kitten recall 相对 S0 提高 0.0222；同时 kitten precision
从 S0 的 0.7833 降至 0.6501，所以 kitten F1 仍下降。主要总体差距位于 adult：S0 三组
累计正确识别 141/186 次 adult，S1 为 117/186，S2 为 120/186。adult 是 111 只猫中
数量最多的类别，其 recall 与 precision 同时影响总体正确数和 macro F1。

S2 相对 S1 将 adult recall 提高 0.0161、senior recall 提高 0.0098，并减少跨两级错误；
与此同时 kitten recall 下降 0.0889。这组交换解释了 S2 的 QWK 较高、macro F1 较低。

## 6. Attention 学到了什么

S2 的平均 normalized attention entropy 为 0.9014。数值 1 代表完全均匀；0 代表权重
集中到单条 call。平均最大 call 权重为 0.4891，平均绝对偏离 uniform 的幅度为 0.0693；
三个 repeat 的平均 effective calls 分别为 4.73、5.64、6.23，而原始平均 call 数约为
7.14。

因此，attention scorer 已经离开零初始化的 uniform 状态，并把一部分权重集中到少数
calls。S2 与 S1 的差异确实包含 learned pooling 的作用。当前学到的权重带来了更好的
ordinal error structure，却没有带来更高 macro F1；这说明“存在猫内异质性”和“能从
111 个 animal bags 学出稳定可靠性规则”是两个不同层次的问题。

## 7. Bag size 与训练动态

| 每只猫的 call 数 | 猫数 | S0 macro F1 | S1 macro F1 | S2 macro F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20 | **0.7813** | 0.7346 | 0.6907 |
| 2–4 | 33 | **0.6018** | 0.5080 | 0.4544 |
| 5–9 | 30 | **0.8389** | 0.7244 | 0.7560 |
| 10+ | 28 | **0.7695** | 0.7192 | 0.7046 |

S0 在四个 bag-size groups 均领先。S2 相对 S1 的局部优势出现在 5–9 calls 组（+0.0316），
其余三组由 S1 较高。2–4 calls 组只包含 1 只 kitten，这一组的 macro F1 对单个样本变化
非常敏感，适合作为描述性定位信息。

inner-validation animal cross-entropy 选择出的 outer epoch 平均值为：S0 10.17、S1
5.75、S2 5.42。S1/S2 多个 fold 在第 1–5 个 epoch 取得最低 validation loss，随后更快
进入过拟合区间。set 训练虽然保留全部 792 条 call，但监督单位从 792 个 call observations
收缩到 111 个 animal bags；当前轻量 head 依然能快速拟合训练 bags，泛化所需的稳定
animal-level 模式较难从有限猫数量中建立。

## 8. 机制解读与阶段决策

S0 的优势可由当前数据和运算位置共同解释：

1. call-level loss 提供更多训练观测与更丰富的 batch 组合；同一只猫的多条声音都能形成
   独立梯度信号。
2. probability mean 先让每条 call 完成非线性分类，再汇总最终证据。hidden mean 在分类
   前压缩整只猫，方向不同的 call representations 可能互相抵消。
3. animal-level class balancing 基于 15/62/34 个猫级样本。kitten recall 的局部提升说明
   类别权重生效；adult 区域的压缩成为主要代价。
4. attention 已学会集中权重，说明容量足以改变 pooling；111 个 bags 提供的监督仍不足以
   稳定区分“可靠 call”和“偶然容易拟合的 call”。

预设继续条件要求 set candidate 相对 S0 至少 `+0.005`、至少 2/3 repeats 为正，并由
secondary metric 支持。S1 与 S2 分别为 `−0.0794、0/3` 和 `−0.0883、0/3`，因此两者
完成为有解释力的 prediction-unit / pooling ablation，seed 43/101 扩展关闭。S0 继续作为
本阶段的 matched frozen-AST reference。

本轮还形成一个独立的简单信号：S0 使用 animal-level validation cross-entropy 选择
checkpoint，seed-17 平均 macro F1 为 0.7570；此前 strict-global C0 使用原 checkpoint
规则时，同三组 seed-17 repeat 的均值为 0.7464，差值为 +0.0106，其中 2/3 repeats
提高。该变化来自 epoch selection，独立于 set aggregation，适合作为后续 AST workflow
中的轻量候选继续评估。

## 9. 文件、环境与哈希

- Idea Card：`plan/IDEA-051_cat_level_set_aggregation.md`；
- 数据诊断：`metadata/experiments/meowagenet_idea051_cat_set_diagnostics_v1.json`；
- Protocol：`configs/protocol/meowagenet_idea051_cat_set_v1.json`；
- Runner：`scripts/run_meowagenet_idea051_cat_set.py`；
- 机器可读结果：`metadata/experiments/meowagenet_idea051_cat_set_v1_results.json`；
- 完整运行目录：`runs/meowagenet_idea051_cat_set_v1/`；
- 代码 commit：`eb8cb828bba40455c26307784121d5f351760ae0`；
- protocol SHA-256：`a8e092418f11e83252fec92b59a8b3cc259a40eb2d68ce7a224cd547f7ac0f48`；
- runner SHA-256：`4e8acdefca96215ae3cec94d9158ac081da82eabce08b73c0fe612334ce5d3d5`；
- execution-lock SHA-256：`0b6f50a1b381ec94abc1704f620940a2b447121d0e2f0984619437366046ec27`；
- environment-lock SHA-256：`42d4e117914605e75051aa2550b6b09d88e0bb93c52990af9c3e273c1190ce2e`；
- evaluation-summary SHA-256：`b1a11a94ee936f92092fb2a2cb3ba0c57f4d923e0786d28c4a336b2da36ebd2e`；
- raw-prediction inventory SHA-256：`22ac6cefe4891894dd815abab59c27429406d2740287d070ad302539dc517271`；
- raw-prediction aggregate SHA-256：`1b59c30373c582238543d0f8ee53100a9a713b152cbd43fb8de3b29bdfd04481`。

运行目录共有 131 个文件、6,192,540 bytes：46 个 JSON 审计文件进入版本控制；82 个
CSV 原始预测与 3 个 smoke checkpoints 保留在本地研究环境。raw-prediction inventory
覆盖 82 个 CSV、637,661 bytes，并提供聚合哈希用于后续完整性核对。
