# IDEA-051 | Cat-level Set Aggregation

## Provenance

- Origin: mixed human direction and AI-assisted experimental decomposition
- Stage: post-evidence exploratory candidate
- Date: 2026-09-13

## Primary angle

- 主要角度：B. Prediction unit
- 支持角度：I. Optimization 与 inference

## Located evidence

### 本地结果

当前 tuned frozen AST 以 call 为训练单位，并在 inference 时对同一只猫的 call
probabilities 取算术平均。数据包含 792 条 call、111 只猫，每只猫拥有 1–45 条 call，
中位数为 5，均值为 7.14；20 只猫只有 1 条 call，28 只猫至少有 10 条 call。

`metadata/experiments/meowagenet_idea051_cat_set_diagnostics_v1.json` 对严格 global
class-balanced C0 的 9 份 complete-OOF 结果进行了 post-evidence 诊断：

- 999 次猫级评估中，484 次（48.45%）至少有一条 call 的 argmax 与最终猫级预测不同；
- 313 次（31.33%）至少四分之一 call 与最终猫级预测不同；
- 猫内 call argmax 平均分歧率为 15.20%；
- call 数与分歧率的 Spearman rho 为 0.5345；
- call 数与猫内 probability total variation 的 rho 为 0.6985；
- 错误猫级评估的平均分歧率为 17.90%，正确评估为 14.19%；
- 分歧率与正确性的 rho 为 -0.0470，当前关联较弱。

这些结果表明，多 call 猫内部存在可测量的预测异质性。异质性随 call 数增加而增大，
同时其强弱尚未直接决定最终正确性，因此需要区分“animal-level objective 的作用”与
“learned reliability pooling 的作用”。

### 文献依据

本轮首先建立本地最小可区分实验。最接近的 multi-instance learning、Deep Sets 与
attention-based MIL 工作将在候选通过首轮数据流与性能筛选后进行局部精读和引用冻结。

### 证据边界

诊断使用已经完成的 outer OOF 预测，作用是生成 post-evidence exploratory idea。它不
提供新模型的独立确认，也不参与新实验 inner checkpoint selection。

## Observed bottleneck

### 已观察事实

1. 训练 loss 作用于单条 call，primary endpoint 作用于整只猫。
2. 同一只猫的 call 数相差最多 45 倍。
3. 近一半重复猫级评估包含 call argmax 分歧。
4. 当前 arithmetic probability mean 固定给予同一只猫的每条 call 相同 inference 权重。

### 尚待验证的 assumption

1. 在隐藏表示层聚合并直接优化 animal-level loss，可以比 call-level loss 更贴近目标。
2. 一部分 call 对年龄判断的可靠性不同，learned attention 可以识别这种差异。
3. 111 个 animal bags 足以训练一个仅增加 129 个参数的 attention scorer。

## Idea claim

将同一只猫的 frozen AST call embeddings 组成可变长度 set，在共享 call encoder 后进行
hidden-level mean 或 learned reliability aggregation，并直接优化 animal-level age loss，
以缩小 call-level 训练与 cat-level evaluation 之间的单位差异。

## Proposed mechanism

每条 768 维 AST embedding 先经过共享的标准化与 `768 → 128` 非线性映射。S1 对同一只
猫的 128 维隐藏表示取均值，再通过 `128 → 3` classifier。S2 为每条隐藏表示学习一个
scalar reliability score，在猫内 softmax 后加权求和，再通过同一个 classifier。

S2 的 attention scorer 以全零权重初始化，因此训练开始时与 S1 的 uniform mean 完全
一致。它只增加 129 个参数，后续差异来自学习到的 call 权重。只有一条 call 的猫自动
得到权重 1；所有 call 均进入集合，不做截断或复制。

animal-level class weights 按 inner-training cats 计算，三个年龄类获得相等总权重，每只猫
在所属类别内等权。这样，000A 等多-call 猫与单-call 猫各自提供一个 animal-level loss，
call 数影响可用观测数量，不再直接决定 loss 项数量。

## Predictions

1. 主预测：S1 或 S2 相对 matched S0 提高 seed-17 三组 complete-OOF animal Macro F1；
   `+0.005` 以上视为值得扩展的首轮信号，`+0.010` 以上视为清楚的性能信号。
2. 支持预测：若单位对齐是主要机制，S1 将高于 S0；若 call reliability 学习提供额外
   信息，S2 将进一步高于 S1。
3. subgroup 预测：set pipeline 在多-call cats 上减少猫级错误或降低不同 bag-size group
   的性能差，同时保持 single-call cats 的直接预测路径。
4. 稳定性预测：有效机制应在至少 2/3 repeats 中产生正的 paired Macro-F1 delta，或改善
   最差 repeat，并在 Balanced Accuracy、QWK 或关键类别 recall 中保留可解释收益。

## Rival explanations

1. 收益只来自训练单位改变：由 S1 对 S0 的比较检验。
2. 收益只来自增加 attention 参数：S2 只增加 129 个零初始化参数，同时报告 S1 与 S2。
3. 收益来自 checkpoint 规则：三条 pipeline 均使用 minimum unweighted inner-validation
   animal-level cross-entropy 选择 epoch。
4. 收益来自随机性：共享 split 与 full seed，并报告三个 paired complete-OOF deltas。
5. 收益来自某一 bag-size 或年龄类：报告 bag-size subgroup、各类别 recall 和 confusion
   matrix。

## Disconfirming evidence

以下结果会关闭当前实现或推动机制修改：

- S1 与 S2 的平均 Macro F1 均低于 S0，且多数 repeat 为负；
- S2 接近或低于 S1，attention 权重仍接近 uniform，说明 reliability scorer 没有提供额外
  信息；
- S2 attention 过度集中于单条 call，并伴随 validation gap 或最低 repeat 下降；
- set pipeline 的收益只出现在类别组成偏斜的 bag-size subgroup，整体 paired 指标缺少
  对应变化。

## Minimum discriminating experiment

### Matched reference

`S0_call_probability_mean`：tuned frozen AST final embeddings，`768 → 128 → 3` head，
strict global class-balanced call loss，最终对同猫 call probabilities 取均值。为了与 set
pipeline 对齐，checkpoint selection 使用 unweighted inner-validation animal-level
cross-entropy。

### 唯一主要变量

训练和预测单位从独立 call 改为 cat bag，并将聚合位置从 probabilities 移到 128 维 hidden
representations。

### 必要 ablations

- `S1_hidden_mean_set`：animal-level loss + uniform hidden mean；
- `S2_attention_set`：animal-level loss + learned hidden attention；
- S2 额外报告 attention entropy、最大权重和 call-count 关系。

### Inner-only selection

- 使用既有 animal-ID-disjoint nested roles；
- 每个 outer fold 仅在 inner train/validation 中选择 epoch；
- selection metric：minimum unweighted animal-level cross-entropy；
- maximum epochs 50，patience 8；
- dropout 0.4457103536、Adamax、learning rate 0.006；
- S0 使用 8-call micro-batch、4-step accumulation 和固定 32-call denominator；
- S1/S2 使用 4-cat batch 和固定 4-cat denominator，每个 epoch 覆盖全部 train cats 一次。

### Outer evaluation

- 首轮 base seed：17；
- repeats：0、1、2；
- folds：0、1、2、3；
- pipelines：S0、S1、S2；
- 总计：36 outer fits，9 份 111-cat complete OOF；
- primary：animal Macro F1；
- secondary：Balanced Accuracy、QWK、accuracy、各类 precision/recall、confusion matrix、
  bag-size subgroup、paired prediction changes。

### 继续条件

任一 set pipeline 相对 S0 的 mean Macro-F1 delta 至少 `+0.005`、至少 2/3 paired repeats
为正，并在 Balanced Accuracy、QWK、最差 repeat 或类别 recall 中呈现一致信息时，进入
base seeds 43/101 扩展。`+0.010` 以上直接视为强扩展信号。首轮未达到该条件时，当前
set 实现完成为可解释 ablation，新的 set encoder 需要新的机制依据。

### 计算预算

使用已有 792 个 frozen AST embeddings。首轮 36 个轻量 fit 适合单张 RTX 4060 Ti，预计
GPU 运行时间为 10–25 分钟；主要工程时间来自可变长度集合数据流、审计和报告。

## Novelty location

- task：将猫年龄分类显式表述为 multi-call animal-level prediction；
- method：共享 call encoder 与零初始化 reliability attention；
- training：直接优化 class-balanced animal loss；
- evaluation：按 call-count group 和 attention concentration 解释 set 行为；
- 与本地实现差别：现有实现只在 inference 端平均 call probabilities；本方法在训练和
  representation 层面共同使用 cat set。

## Status

- Status: shortlisted / implementation authorized
- Human decision owner: project team
