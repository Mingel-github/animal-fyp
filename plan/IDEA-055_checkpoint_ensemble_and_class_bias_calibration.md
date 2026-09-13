# IDEA-055｜AST checkpoint ensemble 与猫级类别偏置校准

## 1. Idea claim

当前 tuned frozen-AST head 在三个 seed-17 repeat 上达到 0.7570 animal macro F1，并且
12 个 outer folds 的 inner-selected epoch 分布在 1–19。单一 epoch 可能携带训练轨迹中的
局部波动；另一方面，kitten、adult、senior 的猫级概率边界可能存在可由训练侧数据校正的
类别偏移。

本轮问题是：

> 在 AST backbone、分类头、损失、splits 和 epoch selection 全部保持一致时，对 selected
> epoch 及其前两个 checkpoint 的预测取平均，并在猫级概率上加入受限类别偏置，能否提高
> 111-cat complete-OOF animal macro F1 或降低 repeat 间波动？

## 2. 方法边界

本轮沿用 IDEA-051/052/053/054 已复现的 tuned frozen-AST reference，只重新训练
`768 → 128 → 3` head。AST 768 维 call embeddings 保持冻结，因此 12 个 folds 只需 12 条
head 训练轨迹。

这里的 checkpoint ensemble 指 **预测平均**：若 inner-selected epoch 为 `E`，使用
`max(1, E−2)` 到 `E` 的最多三个连续 checkpoint，分别产生概率后取算术平均。它与直接
平均网络参数目的相同，都是平滑相邻训练状态；预测平均同时保留每个 checkpoint 自己的
BatchNorm running statistics，定义更直接。

类别偏置在每只猫已经聚合的三类概率上执行：

`p'(y|cat) = softmax(log(p(y|cat)) + [b_kitten, 0, b_senior])`

adult bias 固定为 0，消除三类 bias 同时加常数的冗余。`b_kitten` 与 `b_senior` 只从当前
outer fold 的 inner-validation 猫选择，outer-test 标签不参与选择。

## 3. 2×2 pipeline 矩阵

| Pipeline | Checkpoint 概率平均 | 猫级类别偏置 | 作用 |
| --- | --- | --- | --- |
| A0 single selected checkpoint | 关闭 | `[0, 0, 0]` | matched tuned frozen-AST reference |
| P1 tail-3 checkpoint ensemble | 开启 | `[0, 0, 0]` | 单独检验训练轨迹平滑 |
| P2 class-bias calibration | 关闭 | inner-selected | 单独检验类别边界校准 |
| P3 ensemble + class bias | 开启 | inner-selected | 检验两种机制的组合与交互 |

P1/P3 的 checkpoint 集固定为 selected epoch 结尾的最多三个连续 epoch。P2 与 P3 分别在
自己的 inner animal probabilities 上选择偏置，因此能区分单 checkpoint 与 ensemble 的
校准需求。

## 4. 固定训练条件

- 数据：792 个 calls、111 只猫、kitten/adult/senior 三分类；
- split bank：`meowagenet_formal_v2_nested_roles.csv`，cat-ID disjoint；
- frozen embedding：锁定的 768 维 standard AST call embedding；
- head：`768 → 128 → 3`，ReLU、BatchNorm、dropout；
- dropout：0.44571035356880917；
- optimizer：Adamax，learning rate 0.006，epsilon `1e-7`；
- loss：修正后的 global class-balanced call CE；
- 8-call micro-batch，4-step accumulation，固定 32-call denominator；
- maximum 50 epochs，patience 8；
- selected epoch：最小 inner-validation unweighted animal-level CE；
- 猫级聚合：同一只猫的 call probabilities 算术平均。

## 5. 偏置选择

两个自由 bias 都从固定网格选择：

`[-0.4, -0.2, 0.0, 0.2, 0.4]`

共 25 个组合。`±0.4` 对应把相对类别 odds 乘以约 `0.67–1.49`，属于受限校准。选择顺序
提前固定为：

1. inner animal macro F1 最大；
2. balanced accuracy 最大；
3. animal cross-entropy 最小；
4. `|b_kitten| + |b_senior|` 最小；
5. 固定数值顺序保证完全可重复。

Macro F1 是本项目主指标，因此类别偏置直接按主目标选择；balanced accuracy 与 CE 负责
打破有限 inner cats 上的并列。

## 6. Smoke 与实现审计

正式评价前在 `repeat 0 / fold 0 / seed 17` 运行两 epoch inner-only smoke：

1. 核对 792 calls、111 cats 与 split roles；
2. 确认训练只更新 99,075 个 head 参数；
3. 确认 tail checkpoint 集、概率平均和概率和为 1；
4. 确认两个 25 点 bias grid 都只读取 inner-validation cats；
5. 确认 zero-bias 精确复现对应未校准概率；
6. smoke 全程保持 `outer_test_accessed=false`；
7. 提交代码后生成 execution lock，再开放正式 outer evaluation。

## 7. 初始评价

- base seed：17；
- repeats：0、1、2；
- 每个 repeat：4 个 outer folds；
- 训练规模：12 个共享 head trajectories；
- 评价规模：4 pipelines × 12 folds = 48 个 pipeline-fold predictions；
- complete OOF：每条 pipeline 3 份，共 12 份，每份覆盖 111 cats；
- primary：animal macro F1；
- supporting：balanced accuracy、QWK、plain accuracy、分类别指标、混淆矩阵、paired changes、
  inner-selected bias、tail 长度、时间与显存。

主要配对比较为 P1/P2/P3 各自减 A0；P3−P1 用于识别 ensemble 后的 bias 贡献，P3−P2
用于识别经过 bias 后的 checkpoint ensemble 贡献。

## 8. 预期、竞争解释与扩展条件

### 主要假设

相邻 checkpoint 含有方向一致但局部波动的决策，概率平均可减少边界猫的偶然翻转；受限
类别偏置可校正训练侧反复出现的类别倾向。至少一条 candidate 将获得平均 macro-F1 正增量，
或在保持均值的同时降低 sample SD 和 worst-repeat 损失。

### 竞争解释

1. 当前 selected checkpoint 已由 animal CE 选择，前两个 epoch 可能把较弱状态重新混入；
2. 每折 inner-validation 约 17 只猫，其中 kitten 通常 2–3 只，偏置选择可能随少量边界猫
   改变；
3. 单一类别偏置对所有猫使用同一边界移动，只能处理总体类别倾向；
4. 校准可能改善 macro F1，同时改变概率 CE 或 QWK，四项指标会揭示这种取舍。

任一 candidate 满足以下条件时，将 matched A0 与该 candidate 扩展至 base seeds 43/101：

- 平均 macro-F1 增量至少 `+0.005`；
- 至少 2/3 repeats 为正；
- balanced accuracy、QWK、普通 accuracy、worst-repeat macro F1 或 sample SD 至少一项支持。

达到平均 `+0.01` 且三次方向一致时记为强信号。首轮 gate 关闭时，IDEA-055 仍形成完整的
训练轨迹平滑与决策边界校准消融，并完成当前五项优先路线的阶段性收尾。

## 9. 预期产物

- `configs/protocol/meowagenet_idea055_checkpoint_calibration_v1.json`；
- `scripts/run_meowagenet_idea055_checkpoint_calibration.py`；
- `tests/test_idea055_checkpoint_calibration_runner.py`；
- `runs/meowagenet_idea055_checkpoint_calibration_v1/`；
- `metadata/experiments/meowagenet_idea055_checkpoint_calibration_v1_results.json`；
- `reports/28_IDEA-055_checkpoint_calibration_results.md`。
