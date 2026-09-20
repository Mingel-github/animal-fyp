# IDEA-089 统一核心对照：技术预检

日期：2026-09-20  
状态：本地 CPU preflight **GO**；真实数据训练尚未启动

## 1. 冻结范围

IDEA-089 在 clean111 的 792 calls、111 cats 上，沿用 formal-v2.1 已有 repeats 0/1/2 × 4 outer folds 与 model seeds 17/43/101，不生成新划分。六条 pipeline 为：

- VGG128：128 维 VGGish embedding；
- VGG129：相同 embedding 加作者 `mean_freq` 频率标量；
- A0：冻结 AST 特征加分类头；
- D0：标准化 AST768 与同一 20 维声学特征直接拼接；
- U1：20→60→128 无界加法声学残差；
- C1：相同分支加 `0.25×stopgrad(RMS(h))×tanh(raw_shift)` 整组约束。

每条 pipeline 有 36 个 repeat×fold×seed cells。每个 cell 先完成一次 train/validation epoch selection，再从头在 train+validation 上固定轮数重训，所以总预算为 216 selection + 216 outer = **432 physical fits**。每模型最终形成 9 个完整 111-cat OOF 预测集。

## 2. 固定训练与门禁

所有模型以未加权 validation 猫级 CE 选取 1-based best epoch，严格改善阈值 `1e-6`。AST 四臂沿用 IDEA-076/087 original 配方：Adamax lr `0.006`、dropout `0.44571035356880917`、cat batch 4、max 50、patience 8、CUDA AMP 与 gradient clip 1.0。VGG 两臂沿用 formal-v2.1 Keras 结构及 Adamax lr `0.003109800273709165`、row batch 128、max 500、patience 30；明确变化是改用猫级 CE monitor 与统一 `1e-6` 改善阈值。本机 TensorFlow 2.15 实测没有可见 GPU，因此 VGG 实际运行在 CPU；AST 按命令请求的 PyTorch 设备运行。

Selection 保存 best 权重、训练侧预处理、逐 epoch 历史及 validation unit/cat 概率。全部 216 selection fits 完成后才能生成 epoch-selection lock。Outer 从头初始化并固定 epoch 重训，先保存并复载权重、预处理与训练历史，再写 checkpoint lock，之后只执行一次 test 概率推理。

初批授权与全量授权使用不同的显式 `authorization-scope`。初批 `initial-six-selection` 强制只能运行 repeat 0 / fold 0 / seed 17 的六条 pipeline；即使遗漏 filters 也不会扩大为 216 fits。Outer 若出现 checkpoint 或 test CSV 已写但 summary 未完成的残留目录，resume 会 fail-closed，不会静默重训和再次读取 test。

## 3. 数据与角色审计

- 角色表严格包含 3 repeats × 4 folds × 111 cats；每个 cell 的 train、validation、test 猫互斥且三类齐全。
- fold 0/1/2 的角色规模均为 66 train、17 validation、28 test；fold 3 为 67/17/27。
- 每个 repeat 的四个 test folds 恰好覆盖 111 cats 各一次，因此每个 repeat×seed 可组成完整猫级 OOF。
- AST 缓存为 792 calls/111 cats；声学缓存为 792×20，call ID 与 AST 顺序一致。
- VGGish clean111 投影为 936 rows/111 cats；VGG row 没有可靠 row→call 映射，因此保持 row→cat 概率平均。
- AST、VGG 与角色表的年龄标签逐猫一致；`049A` 不在 analysis cats 中。
- 标准化、声学缺失填补与类别权重只由当前 fit 训练侧估计；outer refit 在 train+validation 上重新估计。

## 4. 参数量口径

| Pipeline | 可训练参数 | 框架原生总状态/count_params |
|---|---:|---:|
| VGG128 | 17,155 | 17,411 |
| VGG129 | 17,283 | 17,539 |
| A0 | 99,075 | 99,075 |
| D0 | 101,635 | 101,635 |
| U1 | 108,143 | 108,143 |
| C1 | 108,143 | 108,143 |

主表只使用统一的“可训练参数”列。Keras `count_params()` 的 VGG 数字另外包含 256 个 BatchNorm moving mean/variance；它不与 PyTorch `parameters()` 作为跨框架可比的总状态指标混用。

## 5. 初始化公平性与可学习性

- A0、U1、C1 的公共 AST 头 state 逐张量精确相同。
- U1 与 C1 的完整公共头加声学分支初始 state 精确相同；不是只依据零输出 logits 推断。
- U1/C1 的 60→128 零初始化投影均获得非零梯度，最大绝对梯度分别为 `0.108461` 与 `0.010558`。
- D0 的前 768 输入列、bias、BN 与输出层精确复制 A0，额外 20 列严格为零；因 788 与 768 输入 GEMM 的浮点求和路径不同，D0−A0 初始 logits 最大绝对差为 `1.94e-7`，小于冻结容差 `1e-6`。
- D0 额外声学列获得非零梯度，最大绝对值 `0.210649`。
- VGG129 的前 128 列和后层与 VGG128 配对，`mean_freq` 列严格为零；在实际训练侧标准化输入上，初始 logits 最大差为 `0`。
- VGG129 的 `mean_freq` 列获得非零梯度，最大绝对值 `0.348830`。

## 6. 指标与比较

每模型先分别计算 9 个完整 111-cat OOF 的 Accuracy、Macro-F1、BA、三类 recall、CE 与 Brier，再报告均值和样本 SD。不以逐折 F1 均值替代 OOF F1，也不跨 seed 平均概率后把集成成绩当作单模型主结果。

预定比较共 9 组：C1−A0、C1−U1、C1−D0、U1−A0、D0−A0、U1−D0、VGG129−VGG128、A0−VGG129、C1−VGG129。U1−D0 比较两套整体融合方案；C1−U1 比较 RMS 相对尺度与 `tanh` 限幅整组约束，不能拆成单组件因果效应。

描述性区间使用固定 seed `20260920` 的 10,000 次年龄组内猫级分层配对 bootstrap；每次同一猫抽样应用于全部模型及全部 9 个 OOF 集。每模型的 999 次预测出现仍来自同一 111 cats，不是 999 个独立动物。

## 7. 测试、哈希与当前门禁

本地 runner 专项测试 12 项通过；连同牛马1的 5 项独立设计测试，共 **17 passed**。CPU preflight 结果为 `GO`，并明确记录 `training_started=false`、`outer_test_accessed=false`。

| 产物 | SHA-256 |
|---|---|
| protocol | `651a568ae48b89c9966345c0293572bbfed02e8a236a472a9fe12df45dff3195` |
| runner | `44f28e1bc809035c4f6793bc250c3d6566f2944c6c3dc6c05378df4d69328e70` |
| tests | `589df957618d3bab3e9bd922c140b3af4a6f4776e14c94a6ccdadf2f3946b605` |
| plan | `29402baed28b28491e913414096590ec7e765d55a509df73a5d4731ea426692b` |
| independent design report | `cb9809b50054f8d81672e2e2ae4df1a0e384590156aea2bfc360f85dfac2cf08` |
| CPU preflight JSON | `f667ad7233c98974bc5fde0a5e03e776735f2f17ba7eee9b182f80d7ccc80c90` |

本地技术预检结论为 **GO**，但它本身不是训练授权。下一步须由牛马1对上述冻结 runner、protocol 与 preflight 做独立只读复核；独立 GO 后，总监只可先放行 `initial-six-selection` 六个预定 selection fits。首批依据工程完整性审计决定是否放行剩余 selection，不依据 validation 成绩高低。
