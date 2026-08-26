# IDEA-013 冻结 AST Temporal Pooling 实验报告

> 日期：2026-08-26
> 正式结果：`v2-headmatched`
> 状态：四折 GPU pilot 完成；作为 aggregation 诊断归档

## 1. 结论摘要

本轮把 standard AST encoder 完全冻结，在固定的 111-cat、792-call folds 下比较三种
patch-temporal aggregation：简单 mean、参数量匹配的 mean + residual adapter、
single-head gated attention。

| Pipeline | Animal macro F1 | Balanced accuracy | QWK | 正确猫数 |
|---|---:|---:|---:|---:|
| Temporal mean + MLP | 0.6222 | **0.6892** | 0.5811 | 70 / 111 |
| Mean + capacity-matched adapter | **0.6329** | 0.6813 | **0.5901** | **71 / 111** |
| Gated-attention pooling | 0.6032 | 0.6740 | 0.5448 | 68 / 111 |
| Length-only probe | 0.2541 | 0.2580 | -0.2126 | 32 / 111 |

结果支持三个直接判断：

1. **Gated attention 没有形成性能收益。** 它比 temporal mean 低 0.0191，比参数量
   匹配 control 低 0.0298，并且在四个 outer folds 中全部低于 capacity control。
2. **额外的轻量非线性容量带来约 1.1 个百分点的小幅改善。** Mean-capacity 是本轮
   最优，但增益位于实践阈值 0.03 以内。
3. **当前 AST patch-temporal 表示的瓶颈不以“少量关键时间片没有被选中”为主。**
   Attention 权重的归一化熵达到 0.999，模型基本保留均匀平均。

IDEA-013 在当前实现中完成了诊断作用：temporal aggregation 会影响结果，简单增加
attention 选择机制没有提升最终性能。当前最有信息量的方案仍是 AST 自带的
CLS/distillation `pooler_output`，而不是频率平均后的 patch 时间序列。

## 2. 为什么使用时间 token，而不是窗口 embedding

已有 fbank 数据包含 843 个 1.28 秒窗口，对应 792 条叫声：

- 750 条叫声只有 1 个窗口；
- 38 条有 2 个窗口；
- 其余 4 条有 3–6 个窗口。

如果 attention 只作用在窗口级，94.7% 的叫声只有一个可选对象。为使 IDEA-013
真正检验 temporal pooling，本轮读取 frozen standard AST 的最后一层 patch tokens：

1. 每个窗口有 `12 frequency × 12 time` 个 patch positions；
2. 对 12 个 frequency positions 做平均，得到 12 个时间表示；
3. 根据 fbank 的真实帧与常量 padding 建立 mask；
4. 保留 receptive field 至少 50% 落在真实音频上的时间 patch；
5. 极短叫声每个窗口至少保留一个 token。

最终得到 5,842 个有效 temporal tokens，每条叫声 1–72 个，中位数为 7 个。Encoder
参数全程冻结，可训练参数只位于 pooling 与分类头。

## 3. 三种 pooling 的公平控制

### Temporal mean

所有有效 token 等权平均，再进入统一 MLP：

`768 → Dense(128) → ReLU → BatchNorm → Dropout → 3 classes`

可训练参数为 99,075。

### Mean-capacity control

先做等权平均，再经过 `768 → 64 → 768` residual bottleneck，然后进入相同 MLP。
可训练参数为 198,211。

### Gated attention

使用 64 维 `tanh × sigmoid` gated scoring，为每个 call 内的 token 学习权重，再进入
相同 MLP。可训练参数为 197,572，与 mean-capacity 相差约 0.3%。

正式 v2 采用以下初始化控制：

- 三个模型的 MLP head 参数逐项完全相同；
- residual adapter 初始为恒等映射；
- gated attention 初始为均匀权重；
- 训练前两种扩展模型与 temporal mean 的最大 logit 差为 0。

因此，训练开始时三个模型代表同一个函数，后续差异来自 adapter 或 attention 学到的
更新。探索性 v1 使用了不同的 head 随机流，只保留为审计产物；本文表格全部来自
head-matched v2。

## 4. 训练协议

- Encoder：冻结 MIT AudioSet standard AST；
- 数据：792 calls、111 cats；
- 划分：固定 4 个 cat-ID-disjoint outer folds；
- batch size：128；
- optimizer：Adamax，learning rate `0.003109800273709165`；
- class handling：balanced class weights；
- 最大 500 epochs，inner `val_loss` patience 30；
- 每折在 inner validation 选择最佳 epoch；
- 使用 outer train + validation cats 从头训练相同 epoch 数，再评价 outer-test cats；
- 主指标：animal-level macro F1；
- Secondary：balanced accuracy、QWK、per-class recall、混淆矩阵。

各模型选择的 epoch：

| Pipeline | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|---:|
| Mean | 1 | 2 | 1 | 1 |
| Mean-capacity | 1 | 6 | 1 | 1 |
| Gated attention | 1 | 3 | 1 | 1 |

多数 fold 在第一个 epoch 获得最低验证损失，说明 patch-temporal head 很快拟合训练集，
可迁移性能主要来自早期表示。

## 5. 四折表现

| Pipeline | Fold 0 F1 | Fold 1 F1 | Fold 2 F1 | Fold 3 F1 | 合并 F1 |
|---|---:|---:|---:|---:|---:|
| Mean | 0.6671 | 0.5906 | 0.6652 | **0.5839** | 0.6222 |
| Mean-capacity | **0.7083** | **0.6101** | **0.7303** | 0.5572 | **0.6329** |
| Gated attention | 0.6603 | 0.5616 | 0.7020 | 0.5211 | 0.6032 |

Mean-capacity 在 fold 0、1、2 提高，在 fold 3 下降。Gated attention 相对
mean-capacity 的四折差分别约为 `-0.0480、-0.0485、-0.0283、-0.0361`，方向一致。

## 6. 类别变化与混淆矩阵

### 每类召回

| Pipeline | Kitten recall | Adult recall | Senior recall |
|---|---:|---:|---:|
| Mean | 0.8000 | 0.5323 | **0.7353** |
| Mean-capacity | 0.8000 | **0.5968** | 0.6471 |
| Gated attention | 0.8000 | 0.5161 | 0.7059 |

Mean-capacity 的主要变化是把 adult 正确数从 33 提高到 37，同时 senior 正确数从
25 变为 22。它改善了 adult 区分能力，并改变了 adult–senior 权衡。

Mean：

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 12 | 3 | 0 |
| Adult | 13 | 33 | 16 |
| Senior | 2 | 7 | 25 |

Mean-capacity：

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 12 | 3 | 0 |
| Adult | 11 | 37 | 14 |
| Senior | 1 | 11 | 22 |

Gated attention：

| 真实 \ 预测 | Kitten | Adult | Senior |
|---|---:|---:|---:|
| Kitten | 12 | 3 | 0 |
| Adult | 14 | 32 | 16 |
| Senior | 3 | 7 | 24 |

## 7. 配对差异

使用 111 只 outer-test cats、2,000 次按真实类别分层的 paired bootstrap：

| 对比 | Macro F1 点差 | 95% bootstrap 区间 |
|---|---:|---:|
| Mean-capacity − Mean | +0.0107 | [-0.0394, +0.0576] |
| Gated attention − Mean | -0.0191 | [-0.0650, +0.0252] |
| Gated attention − Mean-capacity | -0.0298 | [-0.0786, +0.0202] |

Capacity control 的 +0.0107 表明轻量 residual nonlinearity 有一点方向性价值；该幅度
低于 `δ_task = 0.03`。Gated 与 capacity control 的点差接近负 3 个百分点，四折
方向一致，当前排序清楚地支持 capacity control 位于 gated 之前。

## 8. Attention 与捷径诊断

### Attention 是否真正选择了关键时间片

- 归一化 attention entropy：均值 0.9990，中位数 0.9995；
- 平均最大 attention weight：0.2121；
- 771 条多 token 叫声中，权重仍接近均匀；
- 完全落在真实音频上的 patch 平均 lift 为 1.005；
- 部分接触 padding 的 patch 平均 lift 约为 0.94–0.97。

这些数字表明 attention 只进行了很轻的可靠性调整，主要保持 mean pooling。它稍微
偏向完整音频 patch，但没有形成集中、可迁移的时间选择。

### 时长是否足以预测年龄

Length-only probe 只使用：

- `log1p(duration)`；
- `log1p(segment_count)`；
- `log1p(valid temporal token count)`。

它的 animal macro F1 为 0.2541，QWK 为 -0.2126，正确 32/111 只猫。Attention
concentration 与 duration 的 Spearman 相关为 0.128，效应较小。结果支持主模型使用
声学表示，时长和 token 数本身提供的年龄线索很弱。

## 9. 与已有 pipeline 的关系

| 已有或本轮 pipeline | Animal macro F1 |
|---|---:|
| VGGish + MLP 锁定 baseline | 0.6846 |
| Frozen standard AST `pooler_output` + MLP | 0.6525 |
| IDEA-013 mean-capacity patch pooling | 0.6329 |
| IDEA-013 gated-attention patch pooling | 0.6032 |

本轮最优 mean-capacity 比 frozen standard AST `pooler_output` 低约 2.0 个百分点，
比 VGGish baseline 低约 5.2 个百分点。AST 的 CLS/distillation pooled representation
已经融合全局上下文；直接把 patch tokens 沿频率平均后，再学习轻量 temporal pooling，
保留的全局判别信息更少。

因此，IDEA-013 提供了一个明确诊断：**当前主要机会不在 gated temporal selection；
保留 pretrained AST 的全局 pooling 结构更有效。** Mean-capacity 可以作为轻量
head-capacity 观察归档，不进入最终 performance candidate。

## 10. 路线判断

IDEA-013 已完成预定作用。当前路线按既定顺序进入 **IDEA-003：ordinal learning**。

IDEA-003 的最小实验将锁定 VGGish embeddings、cat-ID folds 和 MLP 容量，只改变
objective：

1. nominal three-class cross entropy；
2. cumulative/CORAL ordinal loss；
3. reversed-order ordinal control；
4. cost-sensitive CE control。

Primary metric 转为 animal-level QWK，同时报告 ordinal MAE、kitten↔senior 极端
翻转率、macro F1、adult prediction rate 与每类 recall。这样能够直接检验真实年龄
顺序是否比普通代价加权或正则化提供更多信息。

## 11. 产物

- 正式结果：`runs/idea013_temporal_pooling_v2_headmatched/summary.json`
- Call/animal predictions：`runs/idea013_temporal_pooling_v2_headmatched/`
- Attention 权重：`gated_attention_token_weights.csv`
- Frozen temporal tokens：`runs/ast_temporal_tokens_v1/ast_standard_temporal_tokens.npz`
- 提取脚本：`scripts/extract_ast_temporal_tokens.py`
- 训练脚本：`scripts/run_idea013_temporal_pooling.py`
- 机器可读元数据：`metadata/experiments/meowagenet_idea013_temporal_pooling_v2_results.json`

完整性检查覆盖 792 个唯一 call、111 个唯一 cat-ID、5,842 个唯一 temporal token；
每只猫只进入一个 outer-test fold，每个 call 的 attention 权重和为 1，三类预测概率
和为 1。
