# IDEA-053 AST–VGGish 概率融合实验结果

## 结论摘要

本轮先在同一批 111 只猫的 complete OOF 预测上量化 AST 与 VGGish 的互补性，再以独立
runner 完成 3 个 repeat × 4 折的嵌套概率融合。每折分别训练一个 tuned frozen-AST 分类头
和一个 formal-v2.1 VGGish 分类器，共 **24/24 个模型 fit**；融合权重只由该折的 17 只
inner-validation 猫选择，随后一次性应用于 outer-test 猫。

| Pipeline | Animal macro F1，mean ± SD | Balanced accuracy | QWK | 普通 accuracy |
| --- | ---: | ---: | ---: | ---: |
| A0 tuned frozen AST | **0.7570 ± 0.0086** | **0.7645** | **0.6721** | **0.7568** |
| V0 VGGish + MLP | 0.6795 ± 0.0267 | 0.6795 | 0.5772 | 0.6907 |
| F1 inner-CE probability fusion | 0.7499 ± 0.0213 | 0.7576 | 0.6624 | 0.7477 |

F1 相对 A0 的三个 paired macro-F1 差值为 `+0.0082、−0.0149、−0.0147`，平均
`−0.0071`。融合在 repeat 0 提高约 0.82 个百分点，在 repeat 1 和 2 各下降约 1.5 个
百分点。预设扩展 gate 要求平均至少提高 0.005 且至少 2/3 repeats 提高；本轮达到 1/3
正向 repeat，gate 关闭，base seeds 43/101 无需扩展。

这组结果形成三条可直接使用的结论：

1. AST 与 VGGish 具有真实的预测互补性：333 次猫级配对评价中，VGGish 独自做对 21 次，
   AST 独自做对 43 次，两者预测不同 66 次。
2. 当前“每折一个全局标量权重”的融合方式利用互补性的能力有限。12 折权重在
   `0.2–1.0` 之间波动，融合最终改动 13 次 AST 预测，其中 5 次改对、8 次改错。
3. tuned frozen AST 继续保持当前参考。直接概率融合成为有完整对照的机制消融；后续把
   计算资源转向受约束的 AST 表示校准，条件式 feature fusion 可作为以后独立候选。

## 1. 为什么诊断看起来提高，完整实验却下降

前置诊断把已经生成的三组 complete-OOF 概率按固定权重混合。AST 权重 0.7 时，三个
repeat 的 macro F1 为 `0.7825、0.7565、0.7500`，平均 0.7630，相对 AST 平均提高
0.0060。这说明输出空间里存在一个有用的混合区域，也为运行嵌套融合提供了依据。

这个 0.7 来自查看全部 OOF 结果后的描述性扫描。正式候选需要回答更实际的问题：面对一只
尚未评价的猫，能否只根据训练侧数据选出合适权重？因此正式 runner 在每个 outer fold 内
只用 17 只 inner-validation 猫，以 animal-level cross-entropy 选择 AST 权重
`alpha`，再计算：

`P(fusion) = alpha × P(AST) + (1 − alpha) × P(VGGish)`

12 折选出的 alpha 为：

`0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.8, 0.4, 0.6, 0.5, 0.2, 0.4`

平均 0.575，中位数 0.5；10 折选择两者混合，2 折选择纯 AST。17 只猫构成的小验证集对
少数边界样本很敏感，而且选择目标是概率质量的 cross-entropy，最终主指标是三类等权的
macro F1。于是部分折选择了较大的 VGGish 占比，例如 alpha 0.2 表示 80% 概率来自
VGGish，而本轮 VGGish 平均 macro F1 只有 0.6795。这样的权重在 inner cats 上可降低
cross-entropy，在新的 outer cats 上却容易把 AST 原本正确的边界推向相邻类别。

因此两份结果回答不同问题：0.7 固定扫描证明“概率中有互补空间”；嵌套结果证明“当前
17-cat inner selection 加单一全局权重，尚未稳定找到这片空间”。

## 2. 配对互补性诊断

三个 repeat 共 333 次猫级评价，配对正确关系如下：

| 状态 | 次数 | 含义 |
| --- | ---: | --- |
| AST 与 VGGish 都正确 | 209 | 两种表示给出一致的有效年龄判断 |
| 只有 AST 正确 | 43 | 融入 VGGish 时需要保护 AST 的强判断 |
| 只有 VGGish 正确 | 21 | VGGish 确实提供 AST 偶尔缺失的信息 |
| 两者都错误 | 60 | 仅在两者间选择也无法解决这些样本 |

若有一个事后 oracle 每次都能从两者中挑出正确者，普通 accuracy 上限为 0.8198。这个数字
表示互补空间的理论描述，实际系统没有真实标签来执行 oracle 选择。VGGish-only 的 21 次
分布在 kitten 3 次、adult 12 次、senior 6 次，说明互补信息覆盖三个年龄组，主要数量位于
样本最多的 adult 类。

两组概率的 flattened Pearson correlation 在三个 repeat 中为 0.789、0.815 和 0.760。
它们总体方向相近，同时保留足以造成 66 次类别分歧的差异。由于 AST-only 正确次数是
VGGish-only 的两倍以上，融合器需要在保护 AST 的同时，精准识别 VGGish 的少量优势场景；
一个对所有猫通用的 alpha 缺少这种条件判断能力。

## 3. 三组 complete-OOF 结果

| Repeat | A0 AST | V0 VGGish | F1 fusion | F1 − A0 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7659 | 0.7098 | **0.7741** | +0.0082 |
| 1 | **0.7565** | 0.6694 | 0.7416 | −0.0149 |
| 2 | **0.7487** | 0.6593 | 0.7340 | −0.0147 |
| **平均** | **0.7570** | 0.6795 | 0.7499 | **−0.0071** |

A0 精确重现 IDEA-052 的三个 repeat 结果，V0 也重现所引用的 formal-v2.1 seed-17 VGGish
路径，说明本轮差异来自融合决策。F1 的 sample SD 为 0.0213，高于 A0 的 0.0086，表示
fold-level 权重选择增加了不同 split 之间的波动。

5,000 次 paired cat-cluster bootstrap 中，F1 − A0 的平均差为 `−0.00724`，95% 区间
为 `[−0.0341, +0.0198]`，28.62% 的重采样差值高于零。区间覆盖正负两侧，与 repeat 0
的局部提高以及 repeat 1/2 的回落相一致；分布中心位于 AST 一侧。

## 4. 类别与具体预测变化

三个 repeat 合计后，每个类别共有 kitten 45、adult 186、senior 102 次评价：

| Pipeline | Kitten 正确 / recall | Adult 正确 / recall | Senior 正确 / recall |
| --- | ---: | ---: | ---: |
| A0 AST | **36 / 0.8000** | **141 / 0.7581** | **75 / 0.7353** |
| V0 VGGish | 30 / 0.6667 | 133 / 0.7151 | 67 / 0.6569 |
| F1 fusion | **36 / 0.8000** | 139 / 0.7473 | 74 / 0.7255 |

融合完整保留 kitten 的 36 次正确结果；adult 少 2 次正确，senior 少 1 次正确。因此 macro
F1、balanced accuracy、QWK 和普通 accuracy 都小幅低于 AST。AST 与融合的 kitten↔senior
跨两级错误均为 4 次，QWK 的下降主要来自 adult–senior 等相邻年龄边界的额外偏移。

逐猫配对中，F1 只改变 13/333 次 AST 最终类别：repeat 0 改动 1 次并改对；repeat 1 改动
4 次，1 次改对、3 次改错；repeat 2 改动 8 次，3 次改对、5 次改错。净变化为少 3 次正确，
对应普通 accuracy 从 0.7568 降至 0.7477。这个数量级也说明融合概率大多没有跨过最终类别
边界，真正决定结果的是少量低间隔动物。

## 5. 机制解释与下一步

VGGish 的价值集中在 21 次 AST 错误的判断上，同时它在另外 43 次判断中落后于 AST。
全局 alpha 只能按整体比例折中，无法利用 call 数量、两模型置信差、类别边界或声学条件来
决定每只猫该依赖哪一路。inner-validation 每折只有约 17 只猫，其中 kitten 通常只有 2–3
只，少数预测就能明显改变 cross-entropy 最优权重。这使 alpha 体现出较强的 split
sensitivity。

阶段决策如下：

1. A0 tuned frozen AST 保持当前性能参考；
2. F1 保存为 AST–VGGish global probability fusion 的完整消融；
3. seeds 43/101 扩展关闭；
4. 互补性诊断继续支持“有些猫从 VGGish 获益”这一观察；
5. workflow 按原优先级进入 LayerNorm / SSF / BitFit 等受约束 AST 校准，比较更新位置与
   参数容量；
6. feature-level 或按猫条件选择的融合保留为以后独立 idea，届时需要专门设计可靠性信号，
   避免把当前 scalar alpha 直接复杂化。

直接把 VGGish 当教师进行 distillation 的预期较弱：VGGish 当前平均比 AST 低 0.0775，
教师提供的主要是少量互补样本，而非更强的整体决策。若以后重启该方向，更合适的目标是只在
两路分歧且可靠性信号充分时传递补充信息。

## 6. 运行审计与文件

Smoke test 确认 792 个 AST calls、936 个 VGGish embedding rows、111 个 cat IDs 完整
对齐；outer test 在 smoke 阶段保持未访问，融合概率和误差最多为 `2.22e−16`。正式运行
包含 12 个 fold summaries、24 个模型 fit 和 9 份 111-cat complete OOF；训练与预测记录
累计约 107.4 秒，AST 峰值显存约 20.8 MB。

- Idea Card：`plan/IDEA-053_AST_VGGish_probability_fusion.md`；
- 互补性诊断：`metadata/experiments/meowagenet_idea053_ast_vggish_complementarity_v1.json`；
- Protocol：`configs/protocol/meowagenet_idea053_ast_vggish_probability_fusion_v1.json`；
- Runner：`scripts/run_meowagenet_idea053_ast_vggish_fusion.py`；
- 机器可读结果：`metadata/experiments/meowagenet_idea053_ast_vggish_probability_fusion_v1_results.json`；
- 完整运行目录：`runs/meowagenet_idea053_ast_vggish_fusion_v1/`；
- 执行代码 commit：`3a6837a4dbec29c43cf04621773fc3e4c3a872b7`；
- idea-card SHA-256：`ac5318df7c17e5429c956dd98b58eb5d4d7caefb1b6a9bedbc99d2bd5c972aa6`；
- diagnostic SHA-256：`68c04f5853ea33e5c7e70e005e39152b1b28aaef7bcd3da3a8a9fbd729b11a3d`；
- protocol SHA-256：`7cc1b8fbdcf08fe954ae3eb9241b40336a915c25720bc9a55fa1e455a4b5639c`；
- executed-runner SHA-256：`773741c74b4769c70e65d3ee7ad5ff5ebd58a2e2d9f47ed091e66e696df4d7fb`；
- execution-lock SHA-256：`cea09f1365ccb8c303b59b90dab4848e0513f530abd9d82857e3767160c5f7af`；
- environment-lock SHA-256：`226db7739fb12917bd8482289991553aa42ed0989a888ce4626370777f41b0e4`；
- evaluation-summary SHA-256：`751cb44e56873dde4b73c224c334fcdac4d34af2137df7641105dfe87341173e`；
- raw-prediction inventory SHA-256：`60e797dd2a1f068997a17978732b37489c0e87ff1dea53802419716bef058a65`；
- raw-prediction aggregate SHA-256：`58051654b6f45957b54b1d55fd99eabca6fb5a5286e7c72db68c5294be3d291f`。

运行目录共有 90 个文件、3,032,021 bytes：20 个 JSON 审计文件进入版本控制；70 个 CSV
原始预测保留在本地研究环境。raw-prediction inventory 覆盖全部 70 个 CSV、600,399
bytes，并提供聚合哈希用于完整性核对。
