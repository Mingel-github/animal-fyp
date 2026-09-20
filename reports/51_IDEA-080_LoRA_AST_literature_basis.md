# IDEA-080：最后一个 AST block LoRA × C1 的文献依据

## 结论先行

IDEA-080 不进行新的 LoRA 结构搜索。依据 LoRA 原始论文、作者官方实现、本地 PETL-AST 原始论文及 AST 原始论文，在看任何 IDEA-080 结果前锁定：

- 仅适配 AST 第 12 个、也是最后一个 transformer block；block 1–11 与第 12 block 的其余参数全部冻结；
- 仅对 self-attention 的 query 与 value 投影加入 LoRA；key、attention output、FFN、LayerNorm 与 bias 不训练；
- rank `r=6`；`alpha=6`，因此 scaling `alpha/r=1.0`；
- LoRA dropout `0.0`；AST 原有且冻结的 attention dropout 仍按 eval 模式关闭；
- 低秩更新为 `ΔW=(alpha/r)BA`；`A` 用 Kaiming-uniform 随机初始化、`B` 为零，因此优化前 `ΔW=0`；
- 最后 block 的 Q/V 共增加 `2 × (768×6 + 6×768) = 18,432` 个 LoRA 参数；
- 与冻结的 C1 形成 `A0 / C1 / L1 / CL1` 四管线因子实验，不改变 C1 的 20 维年龄特征、width 60、cap 0.25 或训练规则。

该选择同时满足三点：原始 LoRA 的最常用 attention 起点、本地 AST 音频迁移论文的直接支持，以及只改变最后 block 的可解释参数预算。它不是从 MeowAgeNet 结果中挑出的最佳 rank、层位或投影组合。

## 阅读范围与原始来源

### LoRA 原始论文与官方实现

1. Edward J. Hu 等，*LoRA: Low-Rank Adaptation of Large Language Models*，arXiv:2106.09685，ICLR 2022。原始论文：<https://arxiv.org/abs/2106.09685>；HTML：<https://arxiv.org/html/2106.09685>。
2. Microsoft 官方 LoRA／loralib 实现：<https://github.com/microsoft/LoRA>，线性层实现：<https://github.com/microsoft/LoRA/blob/main/loralib/layers.py>。

原始论文把冻结权重 `W0` 的任务更新写为：

```text
W = W0 + ΔW
ΔW = (alpha / r) B A
A ∈ R^(r×k), B ∈ R^(d×r), r << min(d,k)
```

前向为 `W0 x + (alpha/r) B A x`。`W0` 不接收梯度，只训练 A/B。论文明确用随机 A、零 B 使初始 `ΔW=0`，并说明 alpha 可设为最先采用的 rank 而不再调节；作者官方 `loralib.Linear` 采用 Kaiming-uniform A、零 B、`scaling=alpha/r`，默认 `lora_dropout=0`。本文采用官方代码的具体初始化，同时保持论文要求的零初始更新。

原始论文指出 attention 中有 `Wq/Wk/Wv/Wo` 四个矩阵，主要实验为简洁与参数效率只适配 attention；多数实验使用 Q/V。官方 README 对融合 QKV 的实现也给出 `enable_lora=[True, False, True]`，即 Q 与 V 开启、K 关闭。论文的矩阵选择消融说明在相同参数预算下同时适配 Q 与 V 优于只适配单一投影，是当前 Q/V 锁定的直接依据。

### 本地 PETL-AST 原始论文与官方代码

本地原始 PDF：

```text
C:\Users\zhu\Desktop\essay_article\Audio\03_Training_And_Adaptation\Parameter_Efficient_Tuning\2024_Cappellazzo_PETL_AST.pdf
SHA-256 9ff59f6b4569b2202a320048a561884032fa86f4175458032b8935ec1d0b108d
```

对应 Cappellazzo 等，*Parameter-Efficient Transfer Learning of Audio Spectrogram Transformers*，IEEE MLSP 2024；arXiv:2312.03694：<https://arxiv.org/abs/2312.03694>；官方代码：<https://github.com/umbertocappellazzo/PETL_AST>。

论文第 2 页明确把 AST LoRA 放在 MHSA 的 Q/V 投影，并写出 `Q/V = Xin Wq/v + s Xin Aq/v Bq/v`。第 3 页报告 AST hidden size 768、12 层，并在参数近似匹配的设置中为 LoRA 使用 rank 6。第 4 页显示 LoRA 在四个音频／语音基准的标准 PETL 方法中具有强竞争力：平均结果高于 Pfeiffer bottleneck adapter，并在 4 个任务中的 3 个超过该对照；这为“LoRA 可以迁移到 AST 音频任务”提供直接证据，但不保证其在 MeowAgeNet 的小样本年龄分类中成功。

官方 PETL-AST 代码同样只改 Q/V，冻结 AST 基座，并用零输出侧矩阵实现零影响初始化。论文覆盖所有 12 个 blocks；IDEA-080 则有意只测试最后 block，以回答更窄的问题并把参数量从约 12 倍的全层方案降到 18,432。

另阅读了本地后续论文：

```text
C:\Users\zhu\Desktop\essay_article\Audio\03_Training_And_Adaptation\Parameter_Efficient_Tuning\2024_Cappellazzo_Soft_Mixture_of_Adapters_AST.pdf
SHA-256 345f0542e26f61da92dbb70a70632d8e6c9310a48834f70354a54366b4b9ba3b
```

该文进一步支持 AST 上参数高效模块的可行性，但核心是 soft mixture of adapters，不用于决定本轮 LoRA rank、投影或 scaling，避免把不同模块的结论混用。

### AST 原始论文

本地 PDF：

```text
C:\Users\zhu\Desktop\essay_article\Audio\02_Core_Audio_Models\2021_Gong_AST_Audio_Spectrogram_Transformer.pdf
SHA-256 c237e981755e9ba5047e7290ffcb7d38d8292e166730ae3cd180b7833f184303
```

Gong、Chung 与 Glass，*AST: Audio Spectrogram Transformer*，Interspeech 2021：<https://arxiv.org/abs/2104.01778>；官方代码：<https://github.com/YuanGongND/ast>。

AST 将 log-Mel spectrogram 切成带重叠的二维 patches，线性映射为 768 维 tokens，并使用 12 层、12 heads 的标准 Transformer encoder。其迁移路径来自 ImageNet/DeiT 到音频 spectrogram，再在 AudioSet 等音频任务上训练。当前 MeowAgeNet checkpoint 正是这类 AudioSet 预训练 AST；因此在最后 attention block 的 Q/V 上学习低秩任务更新，结构上与原始 AST 和 PETL-AST 的迁移假设一致。

## 设计选择与证据映射

| 锁定项 | IDEA-080 选择 | 文献／预算依据 |
|---|---|---|
| 目标 block | 仅 block 12 | 最靠近任务表征；用已有 block-11 cache 即可训练，保持 block 1–11 完全冻结；避免重复 IDEA-050 的 last-4/all-12 选择搜索 |
| 目标投影 | Q 与 V | LoRA 原论文多数实验、官方 `MergedLinear` 示例及 PETL-AST 均采用 Q/V |
| Rank | 6 | PETL-AST 在 768 维 AST 的标准 LoRA 设置中采用 rank 6；单 block 参数仅 18,432 |
| Alpha | 6 | 原始 LoRA 建议 alpha 设为最先尝试的 rank且不调节；得到 scaling 1.0 |
| LoRA dropout | 0.0 | 官方 loralib 默认值；PETL-AST 的 Q/V LoRA 路径没有单独输入 dropout；避免引入未经当前文献支持的新随机正则项 |
| 初始化 | A Kaiming-uniform，B zero | Microsoft 官方 loralib；初始 `BA=0`，四管线优化前 logits 可精确相等 |
| Bias | 全部冻结 | 原论文主要 LoRA 设置只训练低秩矩阵；bias 联合训练被列为可选扩展而非默认 |
| C1 | width 60、cap 0.25、20 特征不变 | 继承 IDEA-076 的冻结 C1，不用 IDEA-080 结果重开年龄分支设计 |

## 参数量与公平性

单个 768×768 投影的 LoRA 参数为：

```text
768×6 + 6×768 = 9,216
```

Q 与 V 合计：

```text
2 × 9,216 = 18,432
```

加上现有 99,075 参数分类头与 9,068 参数 C1 年龄分支：

| Pipeline | Head | C1 | LoRA | 总可训练参数 |
|---|---:|---:|---:|---:|
| A0 | 99,075 | 0 | 0 | 99,075 |
| C1 | 99,075 | 9,068 | 0 | 108,143 |
| L1 | 99,075 | 0 | 18,432 | 117,507 |
| CL1 | 99,075 | 9,068 | 18,432 | 126,575 |

这不是等总参数四管线；它是 2×2 因子设计。公平性来自两类组件的精确共享：A0/C1/L1/CL1 的 head 初态相同，L1/CL1 的 LoRA 初态相同，C1/CL1 的年龄分支初态相同，并且两个新增分支都从零影响开始。因此优化前四条管线的 logits 与 loss 必须完全一致。

## 与既往工作的边界

- IDEA-050 曾对 last-4/all-12、rank 4/8、多个 LoRA LR 做嵌套搜索，初始结果低于 matched head。它不等价于本轮“只改最后 block + 与冻结 C1 做因子比较”，但其负结果要求本轮不得再做候选搜索。
- IDEA-076 的 C1 最终确认 gate 已失败并冻结。IDEA-080 不改变这个结论；`CL1−C1` 只回答 LoRA 在固定 C1 上的增量。
- IDEA-078 的 special-token 年龄注入已失败并冻结。IDEA-080 复用其 label-free block-11 token cache，但不把年龄残差放入 token 序列。
- 犬数据与 IDEA-075 不参与 rank、投影、seed、gate 或任何结果解释。

## 结果前决定

文献与预算审计支持进入 CPU 结构预检，结论为 **GO_FOR_CPU_PREFLIGHT**。只有在 CPU 预检同时满足以下条件后，才可请求正式 GPU 跑数：

1. 共享 cache、roles、年龄特征及锁定 AST tail 的哈希一致；
2. 只存在 block 12 Q/V 的 18,432 个 LoRA 参数；
3. 四管线 head 初态相等、L1/CL1 LoRA state 相等、C1/CL1 age state 相等；
4. 四管线初始 logits 和 loss 精确相等；
5. LoRA B 与 C1 output 在零影响点可获得梯度，随后 A 与 C1 hidden 也可获得梯度；
6. 全部 AST base 权重冻结，outer test 未访问。

rank、target modules、alpha、dropout、初始化和四管线在此报告后冻结；无论 CPU 预检或未来正式结果如何，都不得基于结果修改。
