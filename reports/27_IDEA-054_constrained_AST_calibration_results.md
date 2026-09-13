# IDEA-054 LayerNorm、SSF 与 BitFit 受约束 AST 校准结果

## 结论摘要

本轮在相同的 111 只猫、3 个 repeat、4 个 animal-ID-disjoint folds 上比较 tuned frozen
AST 与三种轻量校准方式。四条 pipeline 统一使用 `768 → 128 → 3` 分类头、修正后的全局
class-balanced call loss、32-call 固定累积 denominator、animal-level cross-entropy 选 epoch
和 call-probability mean 猫级聚合，共完成 **48/48 个 pipeline-fold fit**。

| Pipeline | AST 适配参数 | Animal macro F1，mean ± SD | Balanced accuracy | QWK | 普通 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 tuned frozen AST | 0 | **0.7570 ± 0.0086** | **0.7645** | **0.6721** | **0.7568** |
| P1 LayerNorm tuning | 38,400 | 0.7429 ± 0.0196 | 0.7481 | 0.6507 | 0.7417 |
| P2 block-output SSF | 18,432 | 0.7356 ± 0.0036 | 0.7389 | 0.6424 | 0.7357 |
| P3 BitFit | 102,912 | 0.7372 ± 0.0150 | 0.7415 | 0.6436 | 0.7327 |

三条 candidate 相对 A0 的平均 macro-F1 差分别为 `−0.0141、−0.0214、−0.0198`。P1 在
repeat 1 提高 0.0066，另外两个 repeat 各下降约 0.024；P2 与 P3 的六个 repeat-level
差值全部位于 A0 以下。三条预设 seed-expansion gate 均关闭，base seeds 43/101 无需运行。

本轮形成三条可直接用于论文的结论：

1. 当前 tuned frozen AST 表示加非线性分类头保持最强且最稳定，说明 AudioSet 预训练表示
   已包含本数据可用的大部分年龄信息；在 111 只猫上继续移动 backbone 激活会削弱泛化。
2. 参数更少本身不等于效果更好。SSF 只增加 18,432 个适配参数，结果最稳定，同时稳定在
   较低的 0.7356；BitFit 允许更新 102,912 个 bias，平均 0.7372；更新位置和产生的表示偏移
   比参数数量排序更能解释结果。
3. LayerNorm 是三种候选中最接近 A0 的方案，并在一个 repeat 中取得局部提高；它的 sample
   SD 从 A0 的 0.0086 增至 0.0196，显示其收益依赖 split。该信号适合作为 PEFT 讨论，当前
   证据支持将资源转向 checkpoint averaging / output calibration。

## 1. 三种校准各自做了什么

P1 训练 AST 25 个 LayerNorm 的 weight 与 bias。LayerNorm 先规范每个 token 的特征分布，
可训练 weight/bias 再决定 768 个维度各自的尺度和中心。它能在不改 attention 权重的情况
下改变信息进入下一层的相对强度，共涉及 50 个 tensor、38,400 个 AST 参数。

P2 在 12 个 Transformer block 输出后各加入一组逐维 scale 与 shift：

`h_l' = gamma_l ⊙ h_l + beta_l`

scale 从 1 开始、shift 从 0 开始，所以训练前的网络与原 AST 完全一致。它为每层的 768 个
特征维度重新定标，共 24 个 tensor、18,432 个参数，是本轮容量最小的方案。

P3 训练 AST 现有的全部 bias，保留矩阵 weight 冻结。bias 控制 attention、feed-forward、
卷积嵌入和归一化等部件的响应中心，共 98 个 tensor、102,912 个参数。它覆盖的位置最广，
仍保持远少于完整 AST 的可训练容量。

三种方式都回答同一个问题：年龄判别是否只需要轻微校准预训练 AST 的中间表示。它们训练前
与 online frozen AST 的 logits 完全一致，因此最终差异来自训练期间的受约束更新。

## 2. 三组 complete-OOF 结果

| Repeat | A0 AST | P1 LayerNorm | P1 − A0 | P2 SSF | P2 − A0 | P3 BitFit | P3 − A0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **0.7659** | 0.7416 | −0.0243 | 0.7397 | −0.0262 | 0.7539 | −0.0120 |
| 1 | 0.7565 | **0.7631** | +0.0066 | 0.7330 | −0.0235 | 0.7330 | −0.0235 |
| 2 | **0.7487** | 0.7240 | −0.0247 | 0.7340 | −0.0147 | 0.7248 | −0.0239 |
| **平均** | **0.7570** | 0.7429 | **−0.0141** | 0.7356 | **−0.0214** | 0.7372 | **−0.0198** |

A0 再次精确复现 IDEA-051/052/053 的三个 repeat，说明本轮 matched reference 与前序工作
连贯。P1 唯一的正向结果出现在 repeat 1；它在该组把 kitten recall 从 0.8000 提高到
0.8667，同时 senior recall 从 0.7059 调整到 0.6765，因此 macro F1 小幅提高而 QWK 只提高
0.0026。其余两组里 P1 的下降主要落在 senior 识别。

P2 的 sample SD 只有 0.0036，低于 A0 的 0.0086。这表示三次结果非常接近，并且中心稳定
落在 0.7356。稳定性在这里帮助我们排除“个别 split 造成一次低分”的解释：SSF 对 12 层
输出的累计重标定产生了方向一致的总体回落。

P3 在 repeat 0 达到 0.7539，随后为 0.7330 和 0.7248。bias 遍布多个 AST 子模块，它比
SSF 具有更大的有效更新范围；这种更广的平移提升了不同 repeat 间的变化，并把部分 adult
预测推向 senior。

## 3. 类别层面的变化

三个 repeat 合计后，每种 pipeline 共有 kitten 45、adult 186、senior 102 次猫级评价：

| Pipeline | Kitten 正确 / recall | Adult 正确 / recall | Senior 正确 / recall |
| --- | ---: | ---: | ---: |
| A0 AST | **36 / 0.8000** | **141 / 0.7581** | **75 / 0.7353** |
| P1 LayerNorm | **36 / 0.8000** | **141 / 0.7581** | 70 / 0.6863 |
| P2 SSF | 35 / 0.7778 | 140 / 0.7527 | 70 / 0.6863 |
| P3 BitFit | 35 / 0.7778 | 136 / 0.7312 | 73 / 0.7157 |

P1 完整保留 A0 的 kitten 与 adult 正确数，senior 少 5 次正确。这把 P1 的平均差异清楚定位
到 adult–senior 边界。P2 在三个类别分别少 1、1、5 次正确；P3 的 senior 保留得更多，
同时 adult 少 5 次正确。三种更新方式因此呈现不同的边界移动模式，但都降低了三类等权的
macro F1。

逐猫配对变化也支持这一点：P1 改变 19/333 次 A0 最终预测，7 次改对、12 次改错；P2
同样改变 19 次，6 次改对、13 次改错；P3 改变 22 次，7 次改对、15 次改错。三种方式都
能纠正一小部分 A0 错误，新增错误数量更高，最终分别净少 5、7、8 次正确预测。

## 4. 配对不确定性与阶段判断

5,000 次 paired cat-cluster bootstrap 结果如下：

| Candidate − A0 | Bootstrap 平均差 | 95% 区间 | 重采样差值高于 0 的比例 |
| --- | ---: | ---: | ---: |
| P1 LayerNorm | −0.0141 | [−0.0409, +0.0128] | 14.68% |
| P2 SSF | −0.0217 | [−0.0467, +0.0025] | 3.44% |
| P3 BitFit | −0.0200 | [−0.0487, +0.0085] | 7.76% |

P1 的区间最靠近零，也与它的一次正向 repeat 相符；P2、P3 的分布中心进一步位于 A0 一侧。
预设 gate 要求平均至少提高 0.005、至少 2/3 repeats 为正，并有一项支持指标。三种候选的
平均差均为负，P1 达到 1/3 正向，P2/P3 达到 0/3，因此 gate 判断直接明确。

阶段决策：

1. A0 tuned frozen AST 继续作为当前参考；
2. LayerNorm、block-output SSF 与 BitFit 保存为完整的受约束 PEFT 消融；
3. seeds 43/101 扩展关闭；
4. 本轮统一 adaptation learning rate 为 0.0003，因此结论对应当前参数化和学习率；P1 的
   局部信号可在未来作为独立的 layer-scope / learning-rate refinement 候选；
5. 既定 workflow 进入优先级 5：checkpoint averaging / output calibration。该方向调整
   已训练模型在 epoch 或类别决策层面的稳定性，与本轮在线移动 AST backbone 的问题互补。

## 5. 运行与完整性审计

Inner-only smoke 覆盖 792 个 calls 和 111 个 cat IDs，其中 fold 0 使用 517 个 inner-train
calls 与 112 个 inner-validation calls，全程保持 `outer_test_accessed=false`。cached 与
online AST 的最大 embedding、logit、probability 差分别为 `0.002642、0.000239、0.000079`；
差异来自 GPU batch shape 的浮点路径，处于预设容差内。三种 candidate 相对同一 online
AST 的初始 logit 最大差为 0。

两个 smoke epoch 后，P1/P2/P3 分别更新 50/24/98 个适配 tensor，三者各更新 6 个 head
tensor；trainable state 重载后的概率最大差为 0。正式运行包含 48 份 fit summary 与 12 份
111-cat complete OOF，累计模型训练和预测时间约 1,861 秒，online AST 峰值显存约 1.11 GiB。
全套测试结果为 113 passed。

- Idea Card：`plan/IDEA-054_constrained_AST_calibration.md`；
- Protocol：`configs/protocol/meowagenet_idea054_constrained_ast_calibration_v1.json`；
- Runner：`scripts/run_meowagenet_idea054_constrained_ast_calibration.py`；
- 机器可读结果：`metadata/experiments/meowagenet_idea054_constrained_ast_calibration_v1_results.json`；
- 完整运行目录：`runs/meowagenet_idea054_constrained_ast_calibration_v1/`；
- 执行代码 commit：`1f3c1e7d209034ebabf81b9ee464fdc02afb209d`；
- idea-card SHA-256：`8f30a7c61115e1c15c3446c6fc5910a5f62321ce0108b3b10e60772ecb12fdee`；
- protocol SHA-256：`f82b3d4d759b415960721f4513501a2fb073a2dd11e657df0b2a91c784b60026`；
- executed-runner SHA-256：`379911136391b035dddea81f98368fecfe570328d592593a855ee13ac36924f4`；
- execution-lock SHA-256：`9c8e5190e55a4b48429f9947879ff0eec66bb4c3af86c46d1db4b8b5533487e8`；
- environment-lock SHA-256：`9dc291aae362d257c72859632fc00e8be7feabf5d5b1a7ca3371cb0df6947051`；
- evaluation-summary SHA-256：`375b2cf87e1709d62993987483394783e3fd3e71ef6a2db3189ffdfdc9e3f6d9`；
- raw-prediction inventory SHA-256：`8a20e1ce749862ca771c519f820118c48f9fe6e92e5ac3ec52843574d8b21924`；
- raw-prediction aggregate SHA-256：`aaba2d81424314f2fb9eab24501ec4f8cee950aa34d22d6fb35b939de00bbec4`。

运行目录共有 167 个文件、7,261,518 bytes：55 个 JSON 审计文件进入版本控制；109 个 CSV
原始预测与 3 个 smoke checkpoint 保留在本地研究环境。raw-prediction inventory 覆盖全部
109 个 CSV、1,229,639 bytes，并提供聚合哈希用于完整性核对。
