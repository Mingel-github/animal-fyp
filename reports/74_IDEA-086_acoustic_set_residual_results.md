# IDEA-086：排列不变声学集合残差结果

日期：2026-09-19  
正式结论：**本轮不升级主模型。SET1 相对同期 C1 只有很小的平均 Macro-F1 正差，且两条主分类 gate（SET1−A0、SET1−C1）均失败；SET1 的 Macro-F1、Accuracy 和 BA 也未优于 T1/J1。A0/C1 继续作为主候选，暂归档当前固定 SET1 设计，不继续本轮事后调参。**

## 1. 执行与完整性

- 仅新增 `SET1_acoustic_set_residual` 的 `3 base seeds × 3 repeats × 4 folds = 36/36 fits`；A0/C1/T1/J1 的 144 个 IDEA-085 fits 与 288 份预测文件只读复用，没有重训或替换。
- 总监先独立检查源代码并重跑专项测试，授权首个 SET1 fit；首 fit 的 reload、coverage、预测与已训练 checkpoint 排列不变性审计通过后，再授权其余 35 fits。并行的独立复核不是 GPU 授权的前置条件。
- 全量完成后再次执行 canonical `--resume`，7.4 秒结束且没有训练 epoch 输出；36 份新 fit summary 与 72 份新预测文件在复跑前后的组合哈希完全相同。
- 独立结果审计没有导入 IDEA-086 runner 或其 aggregate：核对新 36 + 旧 144 = 180 份 fit summary、360 份预测文件，并从 call 概率重建 180/180 份 cat 概率；与正式汇总逐字段差异为 0（容差 `1e-12`）。runner 与独立 verifier 专项测试 `15/15 passed`，含 IDEA-085 依赖的联合运行 `36/36 passed`。
- IDEA-085 与既有 v3 受保护基线按原始 450 文件清单复核后保持同一组合 SHA-256；未生成、读取或评分 outer-test prediction/metric，`outer_test_accessed=false`。

## 2. 锁定模型与解释范围

SET1 复用 IDEA-085 的六条逐帧原始声学轨迹及六条 finite indicators。每一帧独立通过共享 `Linear(12,48)+GELU → Linear(48,48)+GELU`，随后对真实帧做 masked mean，再经零初始化 `Linear(48,128)` 注入共同分类头。它不含时间卷积、位置编码、attention、variance pooling，也不叠加 C1。残差上限仍为 `0.25 × stopgrad(RMS(h)) × tanh(residual)`。

集合分支有 9,248 个参数，总可训练参数 108,323，比 T1/J1 少 32。因此 SET1−T1/J1 是近似容量匹配的方法比较，而不是只删除“时间顺序”的严格单变量因果消融。SET1 的排列不变性也只针对**同一个冻结 AST call embedding 条件下**送入辅助分支的联合声学帧集合；把原始音频乱序后，AST 主路径仍可能改变。

独立文献、历史差异与方法边界见 [报告 71](./71_IDEA-086_literature_history_and_method_review.md)，非空洞排列、padding、梯度、残差上限和旧结果只读核验见 [报告 72](./72_IDEA-086_model_and_CPU_preflight.md)，独立 CPU 与最终结果复算见 [报告 73](./73_IDEA-086_independent_CPU_preflight_audit.md)。共享逐帧映射加均值池化只可称为受集合模型启发的简化实现，不能声称完整声学分布建模、普适集合近似或新的 Deep Sets 方法。

## 3. 五条管线的 seed×repeat 等权均值

| Pipeline | Accuracy | Macro-F1 | BA | kitten recall | adult recall | senior recall | CE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 AST only | **0.72222** | **0.73622** | 0.75741 | 0.88889 | **0.71667** | 0.66667 | **0.68992** | **0.41245** |
| C1 summary residual | 0.71242 | 0.72547 | 0.75093 | 0.88889 | 0.70278 | 0.66111 | 0.70649 | 0.41817 |
| T1 real-order temporal | 0.72059 | 0.73328 | **0.77037** | **0.93056** | 0.69722 | **0.68333** | 0.71653 | 0.42573 |
| J1 joint-frame shuffled | 0.71895 | 0.73371 | 0.76852 | **0.93056** | 0.69722 | 0.67778 | 0.71238 | 0.42179 |
| SET1 acoustic frame set | 0.71242 | 0.72743 | 0.76204 | 0.91667 | 0.68611 | **0.68333** | 0.70782 | 0.42449 |

这些是 9 个 seed×repeat 估计的等权均值，也是本报告主表。Accuracy、BA、三类 recall、CE 与 Brier 是辅助剖面，不进入 Macro-F1 分类 gate。

## 4. 两条独立的主分类 gate

两条比较分别要求：平均 `ΔMacro-F1 ≥ 0.005`；至少 `2/3` 个 base-seed 均值为正；至少 `6/9` 个 seed×repeat 为正；至少 `8/12` 个 split-cell 非负；最差 split `≥ −0.03`。不存在单一全局 gate。

| 比较 | mean ΔF1 | seed×repeat 正/tie/负 | 正 base seed | 非负 split | 最差 split | 结果 |
|---|---:|---:|---:|---:|---:|---|
| SET1−A0 | −0.008794 | 4/0/5 | 0/3 | 6/12 | −0.111432 | **FAIL** |
| SET1−C1 | +0.001954 | 5/1/3 | 2/3 | 7/12 | −0.081481 | **FAIL** |

### SET1−A0：没有超过 AST-only

平均 Macro-F1 低 `0.008794`；三个 base-seed 均值分别为 `−0.024669/−0.000558/−0.001156`，全部为负。Accuracy `−0.009804`，BA `+0.004630`；kitten/adult/senior recall 分别为 `+0.027778/−0.030556/+0.016667`。CE gain `−0.017893`、Brier gain `−0.012042`，负值表示 SET1 更差。612 次成对重复动物出现中，SET1 纠正 11 次、引入 17 次，净 `−6`；9 个 seed×repeat 的净纠错正/tie/负为 `2/2/5`。五个 gate 条件全部失败。

### SET1−C1：保留微小正向信号，但远未过门槛

平均 Macro-F1 高 `0.001954`，三个 base-seed 均值为 `+0.000252/+0.014890/−0.009280`，达到 `2/3` 个为正；但均值不足 `0.005`，只有 `5/9` 个 seed×repeat 为正、`7/12` 个 split 非负，最差 split 为 `−0.081481`，所以 gate 明确失败。

Accuracy 相同，BA `+0.011111`；kitten/adult/senior recall 分别为 `+0.027778/−0.016667/+0.022222`。CE gain `−0.001324`、Brier gain `−0.006323`，概率质量未改善。纠正 19 次、引入 19 次，净 0；净纠错正/tie/负为 `3/2/4`。因此这里应保留“小幅 F1、BA、kitten/senior recall 正差”的事实，但不能据此升级模型。

## 5. 两条预设辅助比较（无 gate）

| 比较 | mean ΔF1 | seed×repeat 正/tie/负 | 正 base seed | 非负 split | 最差 split |
|---|---:|---:|---:|---:|---:|
| SET1−T1 | −0.005856 | 2/2/5 | 1/3 | 6/12 | −0.051090 |
| SET1−J1 | −0.006284 | 3/2/4 | 0/3 | 6/12 | −0.051090 |

| 比较 | ΔAccuracy | ΔBA | Δkitten | Δadult | Δsenior | CE gain | Brier gain | 纠正/引入/净 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SET1−T1 | −0.008170 | −0.008333 | −0.013889 | −0.011111 | 0.000000 | **+0.008712** | **+0.001246** | 9/14/−5 |
| SET1−J1 | −0.006536 | −0.006481 | −0.013889 | −0.011111 | +0.005556 | **+0.004562** | −0.002702 | 8/12/−4 |

正的 CE/Brier gain 表示 SET1 的损失更低。SET1 相对 T1 在 CE/Brier 上有小幅改善，相对 J1 仅 CE 改善；但两条辅助比较的 Macro-F1、Accuracy 与 BA 都更低。它们没有分类 gate，也不能改写两条主 gate 的失败结论。

## 6. 解释边界与决策

- 这是在 IDEA-085 结果已知后提出、并复用其验证预测的内部探索，不是独立确认或外部验证。
- T1≈J1 且 SET1 参数化不同；SET1 未胜出不能推出“时间顺序必然重要”，也不能推出“无序分布无用”。本轮只否定将这个固定最小集合分支升级为主模型的依据。
- masked mean 汇总的是逐帧非线性表示的经验均值，不保存所有分布性质，也不能重建完整轮廓。
- 每条管线 pooled validation 的 612 行是跨 seed、repeat、fold 的重复动物出现，实际为 97 只 unique cats；每只猫出现 3–15 次，中位数 6 次，不能把 612 当作独立样本量。
- **决策：不升级主模型；A0/C1 仍为主候选。暂归档当前固定 SET1 设计，不对其宽度、pooling 或其他选项做本轮事后搜索；不改变 IDEA-085 或 v3 结论，也不从本轮启动新的 IDEA。**

## 7. 固定结果哈希

- plan：`421e41781fea33adc77814b5a5d0c22f11f219b537c02c3bee4f5eecf347f759`
- protocol：`df576b4c0eb28d7dc524202aa15b40a563d1909a57e9460dbcbf21cf7ee4a527`
- runner：`33e02d175f42c3d799bc4578f6c0dbd59b5eb747f3398821b8d91137dfdbc472`
- tests：`a9533145f2d6abb63afb25914b34f47577e0aa045ba0613bc0aac9eed59d118b`
- literature/history/method review：`db834cb34f6af373fe189deadedb569429f53407909a56ee32bcd30ba2cf3e85`
- reuse manifest：`afec92014e172615433e76a5128e060f1dae6da7ecbb21f4905c985020891898`
- CPU preflight：`fb1f49ee1f279e9d79631d068ffec862854a179225d0f0a7abdd60d5c69f968c`
- run manifest：`6ad421f337f937995b44c60acb9dccc2880bc120cceb82bfb64eab178bc169bd`
- initial evaluation summary：`487e0ad0f86570cbc9ac44d52a11bbf4f17baa7033d6a9df437a1139aeb0f253`
- compact run summary：`0ee04d2b498a79fe483a99e48b0b71f881eeee93f49da9450ab9bec66039a192`
- first-fit audit：`4086002631bb7cd474590a5d7f74b5cef93f8cdd9f6424d0979427d49760ee2e`
- independent results audit：`7f05d5c66da9e62b4af983de63be893f848972add2b51996e38ab0795c027252`
- independent verifier：`aa4d10c3e55994dd237509b5c9d9f587318e31a5281a491487b051ef931ac888`
- independent verifier tests：`645b9b6a66585ae06d64fbf572767e7f4f98f3d5368ab0ca5fd17aa1f04fbfc4`
- independent audit report：`d9ef8915f504d47562d2fe23b4353cdbff1da75fc1144034582f655d007a358e`
- second-resume combined new-fit/prediction snapshot：`955f769260b68f7b6df6782cfe1aeea4f044b3dc27d85a813089b2bd417c156d`
- protected IDEA-085/v3 450-file baseline：`161e2357e544eaffed62b458f8d25eedc4b360a60000f02ca36c09fa7db89af3`
