# IDEA-058｜受约束的 AST 顶层适配

> 阶段：诊断后候选，等待 executable protocol
> 来源：AST internal diagnosis 的 Last-2 mixed signal
> 主要 claim 类型：predictive / adaptation-location candidate

## 1. 已观察结果

inner-only 诊断比较 frozen AST 与 Last-2 adaptation：

- Macro F1：`0.7189 → 0.7439`，平均 `+0.0249`；
- Balanced Accuracy：`+0.0278`；QWK：`+0.0375`；普通 accuracy：`+0.0245`；
- 12 个 split 中 5 个提高、3 个相同、4 个下降；三个 repeat 的均值差为
  `+0.0506、+0.0348、-0.0106`；
- animal-level CE 变化为 `+0.0023`，概率质量没有同步改善；
- Last-2 有 14,276,355 个可训练参数，约为 frozen head 的 144 倍。

该结果是有实际幅度的 mixed signal。它表明顶层适配值得检验，尚不能证明收益来自
pretraining domain mismatch。

## 2. 研究问题与假设

**研究问题：** 在 animal-level checkpoint selection 下，对 AST 顶部 blocks 进行受约束
更新，能否比 frozen AST 和同参数量的非顶部 block 更新获得更稳定的猫级年龄分类增量？

**H058：** 最后若干 Transformer blocks 更接近预训练任务的高层判别空间。以较小 encoder
learning rate 更新这些层，可以让表示适应猫叫年龄分类，同时保留底层声学特征。

**H058-R1，容量解释：** 任意位置增加约 14M 个可训练参数都会产生相似变化，顶部位置没有
独特优势。

**H058-R2，随机波动解释：** `+0.0249` 由少数 inner splits 驱动，扩展到 complete OOF 或
其他 seeds 后收缩。

**H058-R3，checkpoint 与概率解释：** 顶层更新改变分类边界，但 animal-level CE selection
仍会选择概率校准较差的状态。

**H058-R4，过拟合解释：** 小数据不足以稳定估计 14M 个参数，收益伴随更大 split/seed
方差。

## 3. 适配范围

M1 以诊断中的 Last-2 为起点，并允许在 inner-only 阶段对以下内容作有限选择：

- 更新最后一个或最后两个 Transformer blocks；
- encoder learning rate 位于预先列出的窄范围；
- 是否更新 final LayerNorm；
- weight decay、gradient clipping 或同等强度的受约束 regularization。

分类头 recipe、音频输入、猫级聚合和 IDEA-056 checkpoint selection 保持一致。候选数量与
取值在 executable protocol 中冻结，outer-test 不参与选择。

## 4. 核心对照

| 编号 | Pipeline | 需要回答的问题 |
| --- | --- | --- |
| R0 | IDEA-056 tuned frozen AST reference | 更新 encoder 是否提供增量 |
| M1 | 受约束的 top-block adaptation | 顶层更新能否稳定改善分类 |
| C1 | 更新数量相同的 bottom blocks，并使用相同 optimizer、learning rate 和 head | 收益来自顶部位置还是普通可训练容量 |

AST 各 Transformer block 结构相同，top/bottom 对照预期具有相近参数量；执行时报告准确数值。
如果实现差异导致参数量无法合理匹配，protocol 应在运行前改用另一种位置对照并记录理由。

## 5. 可区分预测

- H058 获得支持性证据：M1 跨 repeats 高于 R0，同时高于参数量相近的 C1；
- 容量解释更相容：M1 与 C1 都提高且差异很小；
- 顶层位置解释更相容：M1 提高、C1 接近或低于 R0；
- 过拟合解释更相容：训练损失继续下降，但 inner/outer 指标波动扩大，更多 splits 反向；
- 概率解释更相容：Macro F1 提高，而 animal-level CE 持续变差；
- senior-specific effect：总体收益主要来自 Senior recall，需要同时检查 Adult、Kitten 和
  Balanced Accuracy 的代价。

## 6. 评价与继续条件

首轮使用现有 animal-ID-disjoint roles。具体 top-block recipe 只在 inner-only 范围选择，
随后锁定并执行 R0/M1/C1 complete-OOF comparison。

初始继续条件采用双方向总计划中的共同规则：M1 相对 R0 平均 Macro F1 至少 `+0.005`，三个
complete-OOF repeats 至少两个为正，并且 M1 相对 C1 的结果支持顶部更新位置。还需单独报告
CE、repeat/seed 方差、训练时间、显存和可训练参数。满足条件后锁定 recipe 并扩展 seeds；
未满足时 IDEA-058 作为 top-block adaptation 与更新位置消融收尾。

## 7. 最小交付物

- executable protocol、候选表和 runner；
- trainable parameter names/counts 与冻结层 audit；
- inner-only recipe selection 和 outer execution lock；
- R0/M1/C1 complete-OOF 指标、训练动态与逐类别分析；
- 中文报告、机器可读 JSON 和是否 seed expansion 的团队决定。

## 8. 术语说明

| 术语 | 含义 |
| --- | --- |
| top-block adaptation（顶层适配） | 只更新 AST 最后一个或几个 Transformer blocks，使靠近输出的表示适应当前任务。该短语描述训练范围，不是新算法名称。 |
| bottom-block control（底层 block 对照） | 更新相同数量的前部 Transformer blocks，保持参数量和训练方式接近，用来检验“更新位置”是否重要。 |
| constrained adaptation（受约束适配） | 通过限制更新层数、learning rate 和 regularization 控制模型改动，降低小数据上的过拟合风险。 |
| domain mismatch（领域差异） | 通用 AudioSet 预训练数据与家猫短叫年龄任务之间的数据和目标差异。Last-2 提升只能提示这种可能性，不能单独证明它。 |
