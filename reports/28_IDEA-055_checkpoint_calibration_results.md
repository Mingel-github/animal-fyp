# IDEA-055 AST checkpoint ensemble 与猫级类别偏置校准结果

## 结论摘要

本轮完成当前五项优先路线的最后一项：在 tuned frozen AST 上比较单一 selected checkpoint、
tail-3 checkpoint 概率平均、猫级类别偏置，以及两者组合。四条 pipeline 共享相同训练轨迹，
共运行 **12/12 个 head trajectories**，产生 48 个 pipeline-fold predictions 和 12 份覆盖
111 只猫的 complete OOF。

| Pipeline | Animal macro F1，mean ± SD | Balanced accuracy | QWK | 普通 accuracy | Animal CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 single checkpoint | **0.7570 ± 0.0086** | 0.7645 | **0.6721** | **0.7568** | 0.7160 |
| P1 tail-3 ensemble | 0.7552 ± 0.0105 | **0.7665** | 0.6689 | 0.7508 | 0.7059 |
| P2 class-bias | 0.7513 ± 0.0056 | 0.7538 | 0.6717 | 0.7508 | 0.7171 |
| P3 ensemble + bias | 0.7457 ± 0.0039 | 0.7514 | 0.6668 | 0.7417 | **0.7056** |

P1 与 A0 最接近，平均 macro-F1 差为 `−0.0018`，可以理解为主指标基本持平、中心略低。
它把 balanced accuracy 提高 0.0020，并把越低越好的 animal cross-entropy 从 0.7160 降到
0.7059，说明相邻 checkpoint 平均让概率更平滑、三类召回更均衡；普通 accuracy 与 QWK
分别下降 0.0060 和 0.0032。

P2 相对 A0 平均下降 0.0058，P3 平均下降 0.0113。三条 candidate 都达到 1/3、1/3、0/3
正向 repeats，预设 seed-expansion gate 全部关闭。A0 继续作为主指标参考，P1 保留为具有
概率质量与 balanced-accuracy 小优势的支持性优化结果。

本轮形成三条可直接用于论文的结论：

1. selected epoch 前两个 checkpoint 与最终 checkpoint 的信息高度接近。tail-3 平均只改变
   8/333 次猫级预测，macro F1 基本持平，同时改善 balanced accuracy 和 animal CE；训练
   轨迹平滑主要影响少数决策边界附近的猫。
2. 17-cat inner-validation 上选择统一类别 bias 容易形成 split-specific 边界。kitten bias
   在 12 折中有 9 折选择网格下界 −0.4，senior bias 则覆盖正负两端；完整 OOF 中新增纠正
   少于新增错误，主指标中心下降。
3. checkpoint ensemble 与类别 bias 的组合具有最低的 repeat SD 和最低 animal CE，同时
   macro F1、balanced accuracy 与普通 accuracy 均低于 A0。这说明“概率更平滑、重复结果
   更集中”与“离散分类更准确”是两类不同收益。

## 1. 四条 pipeline 的含义

A0 使用每折 inner-validation animal CE 选出的 epoch `E`，随后在该折全部 outer-training
calls 上重训至 E，并用 E 的概率预测。这就是已经四次复现的 tuned frozen-AST reference。

P1 使用同一训练轨迹中 `E−2、E−1、E` 最多三个 checkpoint 的预测概率平均。12 折选出的
E 为：

`6, 19, 12, 11, 7, 16, 15, 4, 9, 11, 11, 1`

前 11 折都平均三个 checkpoint；最后一折 E=1，只存在 epoch 1，因此 P1 与 A0 在该折完全
相同。这里平均的是概率，每个 checkpoint 都保留自己的 BatchNorm running statistics。

P2 从 A0 的猫级概率出发，在 log probability 上加入
`[b_kitten, 0, b_senior]`。两个 bias 都从 `−0.4、−0.2、0、0.2、0.4` 选择，adult 固定为
0。每折只用约 17 只 inner-validation 猫，按 macro F1、balanced accuracy、animal CE 和
bias 幅度依次打破并列。

P3 对 P1 的 checkpoint-ensemble 猫级概率执行同样的 bias 选择。P1、P2、P3 因此分别
测量轨迹平滑、边界校准，以及两种机制的交互。

## 2. 三组 complete-OOF 结果

| Repeat | A0 | P1 ensemble | P1 − A0 | P2 bias | P2 − A0 | P3 combined | P3 − A0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **0.7659** | 0.7593 | −0.0066 | 0.7448 | −0.0211 | 0.7502 | −0.0157 |
| 1 | 0.7565 | **0.7631** | +0.0066 | 0.7551 | −0.0014 | 0.7435 | −0.0129 |
| 2 | 0.7487 | 0.7433 | −0.0054 | **0.7539** | +0.0052 | 0.7433 | −0.0054 |
| **平均** | **0.7570** | 0.7552 | **−0.0018** | 0.7513 | **−0.0058** | 0.7457 | **−0.0113** |

A0 精确复现 IDEA-051/052/053/054 的三个 repeat。P1 的三个差值围绕零分布：repeat 1
提高 0.0066，repeat 0/2 分别下降 0.0066 和 0.0054。它的 5,000 次 paired cat-cluster
bootstrap 平均差为 −0.00175，95% 区间 `[−0.0174, +0.0149]`，41.08% 的重采样差值
高于零。这是三条 candidate 中最接近 A0 的结果。

P2 在 repeat 2 提高 0.0052，repeat 1 接近持平，repeat 0 下降 0.0211。bootstrap 平均差
为 −0.00578，区间 `[−0.0254, +0.0118]`，27.18% 高于零。其 sample SD 为 0.0056，
低于 A0 的 0.0086，表示结果更集中，同时中心下降约 0.58 个百分点。

P3 的三次结果为 0.7502、0.7435、0.7433，sample SD 只有 0.0039；三次都低于 A0，
bootstrap 平均差为 −0.01162，区间 `[−0.0313, +0.0060]`。组合把波动压到最小，也把
macro F1 的中心压低约 1.13 个百分点。

## 3. 为什么 P1 的 balanced accuracy 提高、普通 accuracy 下降

三个 repeat 合计后，每条 pipeline 有 kitten 45、adult 186、senior 102 次评价：

| Pipeline | Kitten 正确 / recall | Adult 正确 / recall | Senior 正确 / recall |
| --- | ---: | ---: | ---: |
| A0 | 36 / 0.8000 | **141 / 0.7581** | 75 / 0.7353 |
| P1 ensemble | **37 / 0.8222** | 138 / 0.7419 | 75 / 0.7353 |
| P2 bias | 35 / 0.7778 | **141 / 0.7581** | 74 / 0.7255 |
| P3 combined | 35 / 0.7778 | 136 / 0.7312 | **76 / 0.7451** |

P1 比 A0 多识别 1 次 kitten，少识别 3 次 adult，senior 保持一致。普通 accuracy 按全部
333 次评价计数，因此净少 2 次正确；balanced accuracy 先分别计算三类 recall 再等权平均，
kitten 从 0.8000 提到 0.8222 的增量能够抵消 adult recall 的一部分下降，所以 balanced
accuracy 反而提高 0.0020。Macro F1 同时考虑 precision 与 recall，最终小幅下降 0.0018。

P1 只改变 A0 的 8/333 次最终分类：3 次改对、5 次改错。这说明 checkpoint 平均的作用
集中在很少的边界样本上。Animal CE 改善 0.0100，则说明即使最终 argmax 类别不变，平均
后的概率也常常给真实类别分配了更合理的概率。

P3 比 A0 多识别 1 次 senior，同时 kitten 少 1 次、adult 少 5 次。组合校准更偏向 senior
边界，形成较好的 senior recall 和较低的整体概率损失，离散预测的净正确数下降 5 次。

## 4. 类别偏置选择揭示了什么

P2 的 kitten bias 分布为：9 折选 −0.4、1 折选 −0.2、2 折选 0。P3 的 kitten bias 也有
9 折选 −0.4。训练侧经常倾向降低 kitten 相对 adult 的概率，因为每折约 17 只验证猫中
kitten 通常只有 2–3 只，一两个边界预测就足以改变 macro F1 最优点。

senior bias 的方向变化更明显。P2 在 7 折选择负值、5 折选择正值；P3 在 5 折选择 −0.4，
另外 6 折选择 +0.2 或 +0.4。这表示不同 split 对 adult–senior 边界给出不同校准方向，统一
类别偏置缺少稳定的跨折方向。

Bias 值经常落在网格边界，同时完整 OOF 的 P2 只在一个 repeat 提高。这组结果支持将 bias
曲线解释为 inner split sensitivity 的测量，而不是进一步扩大网格。更宽的 bias 会放大当前
边界移动，当前数据已经提供了足够的停止信息。

## 5. 阶段判断与后续位置

预设扩展 gate 要求平均 macro F1 至少提高 0.005、至少 2/3 repeats 为正，并获得一项支持
指标。P1 的 balanced accuracy 与 animal CE 提供支持信息，主指标平均差为 −0.0018、正向
repeat 为 1/3；P2 为 −0.0058 和 1/3；P3 为 −0.0113 和 0/3。三条 gate 均关闭。

阶段决策如下：

1. A0 single selected checkpoint 继续作为 primary macro-F1 reference；
2. P1 保存为近似持平的 checkpoint-smoothing ablation，并记录其 balanced accuracy 与
   animal CE 小优势；
3. P2/P3 保存为 class-bias 与交互消融，统一 bias 对 17-cat inner split 较敏感；
4. seeds 43/101 扩展关闭；
5. 五项优先探索路线完成阶段性收尾。当前结果不会锁死未来方法，后续新增 idea 可以继续以
   A0 为 matched reference；P1 也可在概率校准或置信度研究中作为支持候选。

## 6. 运行与完整性审计

Smoke 使用 repeat 0/fold 0 的 517 个 inner-train calls 与 112 个 inner-validation calls，
保持 `outer_test_accessed=false`。两套 25 点 bias grid 完整执行，zero-bias 的概率最大差为
0，全部输出的概率和误差低于 `7.71e−8`；6 个 head tensor 全部更新。

正式运行包含 12 份 fit summary、48 个 pipeline-fold predictions 和 12 份 111-cat complete
OOF。模型训练与预测累计约 51.65 秒，峰值显存约 19.3 MiB。全套测试为 121 passed。

- Idea Card：`plan/IDEA-055_checkpoint_ensemble_and_class_bias_calibration.md`；
- Protocol：`configs/protocol/meowagenet_idea055_checkpoint_calibration_v1.json`；
- Runner：`scripts/run_meowagenet_idea055_checkpoint_calibration.py`；
- 机器可读结果：`metadata/experiments/meowagenet_idea055_checkpoint_calibration_v1_results.json`；
- 完整运行目录：`runs/meowagenet_idea055_checkpoint_calibration_v1/`；
- 执行代码 commit：`de14d9939caa234086ec19bfb588e9fc4905c8bd`；
- idea-card SHA-256：`4199deea7a50b22726f25ab18272c85bc25c8a4fd7aee130c0dccc00afb9bf0a`；
- protocol SHA-256：`6df469103f75871225f3bf2f6a6ba989f63622ad3770fb14e43d1d61daa11134`；
- executed-runner SHA-256：`dc384a1309b8f585d96488b977a07a9b7d4c9d89d27bd9a1d5f9a6cfb496a1b8`；
- execution-lock SHA-256：`d71a7430c634fd3e7cc23b7eec11b1a1e893fd06342f930b6af2e7f775e2e02b`；
- environment-lock SHA-256：`3bdbb48a97bf180927941a9ec0f600d94bf4c0376722a0d68cfff0519d832a2e`；
- evaluation-summary SHA-256：`26266f2e54e19b085651eae6ecd7f0634a9cc6b026ee79b4f6c59a9ed5aa7989`；
- raw-prediction inventory SHA-256：`d0f31d51477c7e0c504dfa4fd6d60c96e69ae111c079f73d6104039ecb21093a`；
- raw-prediction aggregate SHA-256：`ca8d3dfff8a4d7791d39fdf0e08df46a4e51f3330b8ef6fce27b8d13d8455e7e`。

运行目录共有 138 个文件、3,852,679 bytes：19 个 JSON 审计文件进入版本控制；119 个 CSV
原始预测保留在本地研究环境。raw-prediction inventory 覆盖全部 119 个 CSV、1,568,431
bytes，并提供聚合哈希用于完整性核对。
