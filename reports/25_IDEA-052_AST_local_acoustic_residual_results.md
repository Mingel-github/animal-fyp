# IDEA-052 AST Local Acoustic Residual 初始实验结果

## 结论摘要

本轮完成 3 条 pipeline × 3 个 repeat × 4 折，共 **36/36 个 outer fit**，形成 9 份
111-cat complete OOF。实验保留 tuned frozen AST global embedding 主路径，并测试两种
只增加 128 个 gate 参数的局部 temporal residual。三条 pipeline 共享 splits、model seeds、
训练 batch 顺序、checkpoint selection 和 call-to-cat probability mean。

| Pipeline | Animal macro F1，mean ± SD | Balanced accuracy | QWK | 普通 accuracy |
| --- | ---: | ---: | ---: | ---: |
| R0 global reference | **0.7570 ± 0.0086** | **0.7645** | **0.6721** | **0.7568** |
| R1 temporal mean residual | 0.7562 ± 0.0295 | 0.7638 | 0.6676 | 0.7538 |
| R2 temporal salience residual | 0.7431 ± 0.0186 | 0.7529 | 0.6583 | 0.7447 |

R1 相对 R0 的三个 paired macro-F1 差值为 `+0.0107、+0.0131、−0.0264`，平均
`−0.0009`。它在前两个 repeat 提高约 1.1–1.3 个百分点，第三个 repeat 的下降抵消了
这些收益，sample SD 也由 R0 的 0.0086 增至 0.0295。R2 的三个差值为
`−0.0017、−0.0204、−0.0197`，平均 `−0.0139`。

这组结果形成三条直接结论：

1. temporal mean-shift 含有可学习的年龄信息，并能在部分 animal splits 上改善 AST；
   当前 128 维 gate 对 split 较敏感，三次平均表现与 R0 基本持平。
2. temporal peak-minus-mean salience 在当前共享投影与逐维 gate 实现中持续低于 R0，说明
   最大激活强调了局部强响应，也带入了更多与年龄判断关系较弱的变化。
3. 两条 residual gate 都在 12 个 fold 中广泛离开零值，训练过程确实使用了局部表示；
   性能差异来自学习到的融合方向，而非 residual 分支保持未启用状态。

预设 seed-expansion gate 要求平均 macro-F1 至少 `+0.005`、至少 2/3 repeats 提高，并有
支持指标。R1 满足 2/3 repeats 提高，同时平均值为 `−0.0009`；R2 的平均值为
`−0.0139`。两条 candidate 均完成首轮机制筛选，base seeds 43/101 本轮无需执行。
R0 继续作为 matched tuned frozen-AST reference，workflow 转向 AST–VGGish
complementarity、fusion / distillation。

## 1. 实验问题与三条 pipeline

AST 的 global pooler 把一条叫声压缩成一个 768 维向量。诊断显示同一条叫声内的 temporal
tokens 仍有明显变化，而且 token dispersion 从 kitten 到 adult、senior 逐步提高。本轮
检验这些局部变化作为补充信息时，能否在保留强 global representation 的条件下进一步提高
猫级年龄分类。

- **R0 global reference**：标准化后的 768 维 global embedding 经
  `768 → 128 → 3` 分类头，每只猫最终平均其全部 call probabilities。
- **R1 temporal mean residual**：temporal tokens 经同一个 `768 → 128` 投影后求均值，
  计算 `temporal hidden mean − global hidden`，再由 128 维 gate 逐维缩放并加回 global
  hidden。
- **R2 temporal salience residual**：计算 temporal hidden 的逐维
  `max − mean`，让 gate 选择局部峰值相对一般水平的增量。

R1/R2 各比 R0 增加 128 个 trainable scalars，总参数由 99,075 增至 99,203。gate 从全零
开始，因此三条 pipeline 的初始 fused hidden 和 logits 完全一致。R2 对只有一个 temporal
token 的 call 自动产生零 residual。

## 2. 数据、训练与运行审计

本轮同时读取 792 个 frozen AST global call embeddings 和 5,842 个 final-layer temporal
tokens，覆盖 111 只猫。每条 call 含 1–72 个有效 tokens，中位数为 7。global embedding
与 temporal mean 的平均 cosine similarity 为 0.2728，表明两种表示具有明显差异；年龄
标签与 token dispersion 的 Spearman rho 为 0.2470，为局部分支提供了可测量依据。

实验使用 base seed 17、repeat 0/1/2、每个 repeat 四个 animal-ID-disjoint outer folds。
训练固定为 class-balanced call loss、8-call micro-batch、4-step accumulation、32-call 固定
分母、Adamax、learning rate 0.006、dropout 0.4457、patience 8。inner-validation 的
animal-level cross-entropy 选择 outer epoch，最终评价单位为 111 只猫。

Smoke test 先完成以下核对：

- 792 个 calls 和 5,842 个 temporal tokens 全覆盖；
- R1、R2 相对 R0 的初始最大 logit 差均为 0.0；
- 三个 checkpoint 重载后的最大 probability 差均为 0.0；
- 单-token call 的 R2 residual 为 0；
- outer test 在 smoke 阶段保持未访问状态。

正式 36 个 fit 的共同 epoch batch-order hashes 全部一致。训练与预测累计约 306.6 秒，
运行设备为 RTX 4060 Ti；结果目录包含 46 个 JSON、91 个 CSV 和 3 个 smoke checkpoints。

## 3. 三组 complete-OOF 主结果

| Repeat | R0 macro F1 | R1 macro F1 | R1 − R0 | R2 macro F1 | R2 − R0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7659 | **0.7766** | +0.0107 | 0.7641 | −0.0017 |
| 1 | 0.7565 | **0.7696** | +0.0131 | 0.7361 | −0.0204 |
| 2 | **0.7487** | 0.7224 | −0.0264 | 0.7290 | −0.0197 |
| **平均** | **0.7570** | 0.7562 | **−0.0009** | 0.7431 | **−0.0139** |

R0 的范围为 0.7487–0.7659。R1 的范围扩展到 0.7224–0.7766：最高 repeat 比 R0 的最高
repeat 高 0.0107，同时最低 repeat 比 matched R0 低 0.0264。R1 因此呈现清楚的局部潜力
与更大的 split sensitivity。R2 的范围为 0.7290–0.7641，三个 matched repeat 均由 R0
保持领先。

333 次逐猫评估中，R1 相对 R0 新增 8 次正确，同时改变了 9 次原本正确的结果，净变化
为 −1；R2 新增 8 次正确，同时改变 12 次原本正确的结果，净变化为 −4。R1 的变化量很小，
却发生在不同猫和类别之间，因此 macro F1、balanced accuracy 与 accuracy 的均值只相差
约 0.001–0.003。

## 4. 指标共同说明的信息

R1 的 macro F1 0.7562 与 R0 的 0.7570 相差 0.0009，balanced accuracy 只相差 0.0006，
普通 accuracy 相差 0.0030。这三个指标共同说明 R1 与 R0 的总体分类能力接近。R1 的 QWK
低 0.0045，表示其错误在年龄顺序上的平均代价略高。

R2 相对 R0 的 macro F1、balanced accuracy、QWK 分别低 0.0139、0.0116、0.0138，三个
指标方向一致。它既减少总体正确数，也稍微削弱三类 recall 的平均值和年龄顺序一致性。

5,000 次 paired cat-cluster bootstrap 给出：

- R1 − R0 的平均差 `−0.00085`，区间 `[−0.0247, +0.0245]`，bootstrap samples 中
  46.34% 高于零；
- R2 − R0 的平均差 `−0.01413`，区间 `[−0.0413, +0.0135]`，15.70% 高于零。

R1 的区间围绕零近似对称，符合“总体接近、不同 animal compositions 下方向会交换”的
观察；R2 的分布中心更偏向 R0。

## 5. 类别变化与混淆结构

三个 repeat 合计后，每个类别共有 kitten 45、adult 186、senior 102 次猫级评估：

| Pipeline | Kitten 正确 / recall | Adult 正确 / recall | Senior 正确 / recall |
| --- | ---: | ---: | ---: |
| R0 | 36 / 0.8000 | 141 / 0.7581 | **75 / 0.7353** |
| R1 | **37 / 0.8222** | **142 / 0.7634** | 72 / 0.7059 |
| R2 | 36 / 0.8000 | 140 / 0.7527 | 72 / 0.7059 |

R1 相对 R0 多识别正确 1 次 kitten 和 1 次 adult，同时少识别正确 3 次 senior。它的局部
收益主要位于 kitten/adult，主要代价位于 senior。R2 保持 kitten，adult 少 1 次、senior
少 3 次。三条 pipeline 的 kitten↔senior 跨两级错误均为 4 次，QWK 差异主要来自相邻
类别之间的变化以及各类别边际分布。

按平均 call 时长分组，R1 在 0.5–1.0 秒的 79 只猫上 macro F1 为 0.7845，高于 R0 的
0.7782；在平均 token 数大于 9 的 25 只猫上为 0.7057，高于 R0 的 0.6953。与此同时，
大于 1 秒组只有 14 只猫，R1 为 0.6675、R0 为 0.7025。分组结果说明 R1 的作用与 call
长度和 token 数存在交互，也显示固定逐维 gate 尚未在所有局部结构上形成统一规则。

## 6. Gate 与训练动态

R1 的 12 个 outer models 平均绝对 gate 为 0.0240，最大绝对值 0.1492，平均 97.20% 的
维度超过 0.001。R2 对应为 0.0248、0.2170 和 97.46%。两条分支都广泛使用了 residual，
R2 甚至形成更大的单维调整，因此“R2 表现较低”对应的是学到的 salience 融合效果，而非
gate 容量完全闲置。

R0、R1、R2 的平均 selected epoch 分别为 10.17、9.17、12.75。R1 并未统一更早停止；
R2 平均训练更久且最高达到 29 epoch。结合 R2 较低的 OOF 指标，peak-minus-mean 信号给
优化提供了可拟合方向，其中一部分方向更接近局部声学幅度、时长或录音变化，跨猫泛化
价值低于 global AST representation。

## 7. 机制解读与阶段决策

R1 的结果很有信息量：它在两个 repeat 上带来约一个百分点提升，说明 global AST pooler
之外确实存在可利用的 temporal mean-shift。第三个 repeat 的反向幅度更大，说明共享投影
后的简单逐维 gate 会随训练猫组合学习不同的 residual 方向。111 只猫中 kitten 只有 15
只，senior 34 只，局部声学差异又同时受 call 时长和 token 数影响，因此一次 split 中有用
的方向在另一次 split 中可能改变 senior 决策边界。

R2 把每维最大 token 激活当作显著局部事件。最大值能突出短促强响应，也会强调瞬时能量、
背景或切分边界带来的激活。当前结果显示这些峰值变化的年龄特异性低于 temporal mean-shift；
R2 可作为“瞬时峰值 residual”机制消融保留。

阶段决策如下：

1. R0 保持当前 matched tuned frozen-AST reference；
2. R1 记录为具有 split-dependent positive signal 的 local-residual ablation；
3. R2 记录为低于 reference 的 salience ablation；
4. seed 43/101 扩展关闭，把计算资源转向优先级 3；
5. 下一阶段先量化 AST 与 VGGish 在同一 111-cat OOF 上的互补错误，再决定 probability
   fusion、feature fusion 或 distillation 的最小实验。

这不是对全部 local acoustic enhancement 的终止判断。它只完成当前 frozen AST temporal
tokens、共享 128 维投影、逐维零初始化 gate 这一种参数化的阶段筛选。R1 提供的两个正向
repeat 可以在论文中支持“局部均值信息有可取之处，稳定融合仍是关键问题”。

## 8. 文件、环境与哈希

- Idea Card：`plan/IDEA-052_AST_local_acoustic_residual.md`；
- 数据诊断：`metadata/experiments/meowagenet_idea052_local_residual_diagnostics_v1.json`；
- Protocol：`configs/protocol/meowagenet_idea052_ast_local_residual_v1.json`；
- Runner：`scripts/run_meowagenet_idea052_ast_local_residual.py`；
- 机器可读结果：`metadata/experiments/meowagenet_idea052_ast_local_residual_v1_results.json`；
- 完整运行目录：`runs/meowagenet_idea052_ast_local_residual_v1/`；
- 执行代码 commit：`df86cc862705fd39e99fa252679be1cf32a80077`；
- idea-card SHA-256：`bd34ab3898190ab1383720feae49d74d95bdd82b74fc0e891ad03f95d7908aaa`；
- diagnostic SHA-256：`00bb22a60af2f4e8f485eb64d09dbf8b0122b054d57a9548974f996eba3d701a`；
- protocol SHA-256：`a24da7047bd5d1d774964f92e33f7fb78d23c1ca1848b917d77a3d401f661212`；
- executed-runner SHA-256：`445c2688e081c30e6f619254484cf2e3367abedaf7183fab41707359ab1a2639`；
- execution-lock SHA-256：`ab6f39868036ec85f4c7c336b8777c83bf7269f2157d01899dd927a330596a9a`；
- environment-lock SHA-256：`d4dda11ea18641de55d5065dae13de0c7f47ae74caf002aee46ffe110938adc9`；
- evaluation-summary SHA-256：`247807f99c238e8de1fdfff26b39983d6122289e0fd67569e179192ce0419e97`；
- raw-prediction inventory SHA-256：`51e53a7cf228adc4dc1c8517cfbb42e09430df98a460fbecd4dae73941c3f696`；
- raw-prediction aggregate SHA-256：`57453d81a1b0be1fc10adab3841666df4a2ca110dc570c5861c21e6380fc60dc`。

运行目录共有 140 个文件、9,231,052 bytes：46 个 JSON 审计文件进入版本控制；91 个 CSV
原始预测与 3 个 smoke checkpoints 保留在本地研究环境。raw-prediction inventory 覆盖
91 个 CSV、1,831,229 bytes，并提供聚合哈希用于后续完整性核对。
