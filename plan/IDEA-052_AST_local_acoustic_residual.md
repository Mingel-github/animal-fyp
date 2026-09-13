# IDEA-052 | AST Local Acoustic Residual

## Provenance

- Origin: repository priority-2 route, refined from local diagnostics and completed temporal-pooling evidence
- Stage: post-evidence exploratory candidate
- Date: 2026-09-13

## Primary angle

- 主要角度：E. Feature extraction
- 支持角度：D. Input representation

## Located evidence

### 当前强参考

IDEA-051 的 matched `S0_call_probability_mean` 使用 frozen AST final call embedding、
strict global class-balanced call loss 和 animal-level checkpoint selection，在三份 seed-17
complete OOF 中得到 animal macro F1 `0.7659、0.7565、0.7487`，均值 `0.7570`。

### 局部表示诊断

`metadata/experiments/meowagenet_idea052_local_residual_diagnostics_v1.json` 读取同一 frozen
standard AST 的 792 个 global call embeddings 和 5,842 个有效 temporal tokens：

- 每条 call 有 1–72 个有效 temporal tokens，中位数 7；
- token RMS dispersion 均值为 0.4407，peak-minus-mean residual RMS 均值为 0.6733；
- global embedding 与 temporal-token mean 的平均 cosine similarity 只有 0.2728，说明两种
  representation 在 frozen AST 空间中保留了不同信息；
- token dispersion 在 kitten、adult、senior 三类中的均值依次为
  `0.4066、0.4383、0.4625`；
- ordinal age label 与 token dispersion 的 Spearman rho 为 0.2470，与 peak residual 的 rho
  为 0.1735；
- token variation 与当前 S0 正误的直接相关较弱，说明固定阈值或手工规则难以直接使用，
  更适合通过受约束的 residual gate 在分类任务中选择有效维度。

### 与 IDEA-013 的区别

IDEA-013 用 temporal mean、matched-capacity mean 或 gated attention 直接替代 global AST
representation，macro F1 分别为 0.6222、0.6329、0.6032。IDEA-052 始终保留 global
embedding 主路径，并以零初始化 residual gate 添加局部信息。训练开始时三个 pipeline 的
global logits 完全一致，局部分支只能通过后续学习贡献增量。

## Observed bottleneck

### 已观察事实

1. AST global pooler 将一个 call 的 patch/token 信息压缩为一个 768 维向量。
2. Temporal tokens 在同一 call 内具有明显离散度，并呈现年龄类别梯度。
3. Temporal-only replacement 明显弱于 global AST，说明局部表示更适合作为补充信息。
4. 当前强 reference 的训练单位和 checkpoint selection 已稳定，可用于 matched comparison。

### 待检验 assumption

1. temporal-token mean 与 global pooler 的差异包含可用于年龄分类的补充方向；
2. temporal-token peak-minus-mean 表示亚秒级显著激活，可补充被 global pooling 平滑的线索；
3. 128 个零初始化 gate 参数足以选择有用 residual dimensions，同时保持小数据稳定性。

## Idea claim

保留 frozen AST global embedding 的主分类路径，并将同一 AST 最后一层 temporal tokens
编码为受控 residual，以极少新增参数向分类头补充局部声学信息，可以比 matched global
reference 获得更好的 animal-level 年龄分类结果。

## Proposed mechanism

三条 pipeline 共享 global standardization、`768 → 128` linear、ReLU、BatchNorm、
dropout 和 `128 → 3` classifier。局部 tokens 使用当前 outer/inner training calls 拟合的
独立 standardization，再经过同一个 `768 → 128` linear 和 ReLU。

- `R0_global_probability_mean`：只使用 global hidden representation；
- `R1_temporal_mean_residual`：计算 temporal hidden mean 与 global hidden 的差，经过 128
  维 `tanh(gate)` 逐维缩放后加回 global hidden；
- `R2_temporal_salience_residual`：计算 temporal hidden max 与 temporal hidden mean 的差，
  经过同样的 128 维 gate 加回 global hidden。

R1/R2 各只增加 128 个 trainable scalars。gate 全零初始化，因此初始 fused hidden 与 R0
完全相同。所有 5,842 个有效 tokens 均保留；单-token call 的 R2 salience residual 自动为
零。三个 pipeline 最终都对同猫 call probabilities 取算术平均。

## Predictions

1. 若 temporal mean-shift 提供 global pooler 之外的信息，R1 相对 R0 的 complete-OOF
   macro F1 将提高，且 gate 离开零值。
2. 若短时显著激活更重要，R2 将高于 R1 或 R0，并在 senior recall、QWK 或长 call subgroup
   中表现出支持信号。
3. 若 residual information 与任务关联不足，R1/R2 的 gate 即使发生变化，也不会形成稳定
   的 paired OOF 增益。
4. 若局部分支主要造成小样本过拟合，R1/R2 会更早选择 epoch、repeat 方差增加，或仅在
   单个 split 上提高。

## Rival explanations

1. 收益来自 head 参数增加：R1/R2 各只增加 128 个逐维 gates，且从与 R0 完全相同的
   logits 开始；需要报告 gate 使用程度。
2. 收益来自 checkpoint 规则：三条 pipeline 均使用 minimum unweighted inner-validation
   animal-level cross-entropy。
3. 收益来自 batch order：三条 pipeline 共享 call order 和 full seed，并核对共同 epoch 的
   order hash。
4. 收益来自某个时长、token-count 或年龄类别：报告对应 subgroup、class recall 和 paired
   prediction changes。

## Minimum discriminating experiment

### 数据与 split

- frozen global AST call embeddings：792 × 768；
- frozen final-layer temporal tokens：5,842 × 768；
- 使用既有 animal-ID-disjoint nested roles；
- 首轮 base seed 17，repeat 0/1/2，每个 repeat 四折。

### 固定训练条件

- strict global class-balanced call loss；
- 8-call micro-batch、4-step accumulation、固定 32-call denominator；
- dropout 0.4457103536、Adamax、learning rate 0.006；
- maximum epochs 50、patience 8；
- checkpoint selection：minimum unweighted inner-validation animal cross-entropy；
- output：同猫 call probabilities 算术平均。

### 运行矩阵

- pipelines：R0、R1、R2；
- 36 个 outer fits；
- 9 份 111-cat complete OOF；
- primary：animal macro F1；
- secondary：balanced accuracy、QWK、plain accuracy、各类 precision/recall/F1、confusion
  matrix、duration/token-count subgroup、residual gate magnitude 和 paired changes。

### 继续条件

任一 residual pipeline 相对 R0 的 mean macro-F1 delta 至少 `+0.005`、至少 2/3 repeats
为正，并在 balanced accuracy、QWK、最差 repeat 或关键 class recall 中保留同方向信息时，
扩展 base seeds 43/101。`+0.010` 以上视为清楚的首轮性能信号。首轮未达到条件时，当前
local residual parameterization 完成机制消融，workflow 转向 AST–VGGish fusion / distillation。

## Novelty location

- task：针对亚秒猫叫年龄分类中的 local/global 信息互补；
- method：共享 global projection 的 zero-initialized temporal residual gate；
- efficiency：每个 local pipeline 只增加 128 个参数；
- evaluation：同时检验 mean-shift 与 transient salience 两种局部信息；
- 与本地历史差别：保留强 global AST path，避免 temporal-only replacement。

## Status

- Status: shortlisted / implementation authorized
- Human decision owner: project team
