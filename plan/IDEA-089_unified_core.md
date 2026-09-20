# IDEA-089：统一五模型核心对照与 F0 配对控制

日期：2026-09-20  
状态：技术设计已冻结；CPU 预检与独立审计前不得启动真实数据训练

## 1. 研究问题与范围

本轮在同一套 clean111 MeowAgeNet 内部评估协议下重新训练五个核心模型，并加入一个 VGGish F0 配对控制，回答：

1. 显式 20 维声学特征能否补充冻结 AST 表示；
2. C1 的相对幅度约束相对简单直接拼接与无界残差贡献多少；
3. VGGish 的作者 `mean_freq` 特征在本轮严格配对条件下贡献多少。

数据固定为 792 个去重 calls、111 个 analysis cats，排除重复别名 `049A`。使用 `meowagenet_formal_v2_nested_roles.csv` 中 repeats 0/1/2、每个 repeat 的 4 个 outer folds，以及 model seeds 17/43/101。不生成新划分，不扩展数据集，不做 HPO，也不按结果追加模型、seed 或阈值。

同一 111 猫已用于历史开发，因此本轮是统一内部验证，不是新动物或外部确认。

## 2. 六条预定 pipeline

- `VGG128_no_f0`：128 维 VGGish embedding，128-ReLU-BN-Dropout-3 分类头；
- `VGG129_with_f0`：相同结构，输入额外包含作者 `mean_freq`；
- `A0_ast_only`：冻结 AST 768 维 call embedding，768→128→3 分类头；
- `D0_direct_concat`：训练侧分别标准化 AST 768 维与同一 20 维声学特征，拼接后 788→128→3；
- `U1_wide_unbounded_additive`：20→60→128 自由加法声学残差；
- `C1_bounded_wide_additive`：相同分支，以 `0.25 × stopgrad(RMS(h)) × tanh(raw_shift)` 限幅后加到 AST 主路。

D0 的 128 维隐藏层、BN、dropout 和输出层与 A0 相同。D0 输入层前 768 列复制 A0 初态，新增 20 列置零，因此优化前 logits 在审计浮点容差内与 A0 相同，同时声学列从第一步即可学习。A0/D0/U1/C1 共享公共头初态、猫与 call 顺序及 post-build dropout/AMP 随机流；U1/C1 的公共头和完整声学分支初态均须逐状态核验一致，不能只因 60→128 投影置零而根据 logits 推断公平性。

VGG129 的前 128 列、bias、BN 与输出层复制 VGG128 初态，新增 `mean_freq` 列置零。两臂共享 row 顺序和 dropout 随机流，优化前 logits 必须在审计浮点容差内相同，且 `mean_freq` 列必须有非零可学习梯度。因此 VGG129−VGG128 只在本轮锁定条件内解释为额外作者频率标量输入的配对对照。

参数量分开报告。可训练参数为 VGG128 `17,155`、VGG129 `17,283`、A0 `99,075`、D0 `101,635`、U1/C1 各 `108,143`。Keras `model.count_params()` 还包含 256 个 BatchNorm moving mean/variance，因此 VGG128/VGG129 的总状态参数分别为 `17,411`/`17,539`；该口径不与 PyTorch 可训练参数混列。

## 3. 固定训练配方

所有模型以未加权 validation 猫级 cross-entropy 选择 1-based best epoch。严格改善定义为新值小于旧值减 `1e-6`；改善时更新 best 并清零 stale，否则 stale 加一。

AST 四臂沿用 IDEA-076/087 original 配方：Adamax，lr `0.006`，epsilon `1e-7`，dropout `0.44571035356880917`，cat batch 4，最多 50 epochs，patience 8，CUDA 上使用 AMP，gradient clip 1.0，无 weight decay。

VGG 两臂沿用 formal-v2.1 Keras 结构与主要配方：Adamax，lr `0.003109800273709165`，epsilon `1e-7`，dropout `0.44571035356880917`，row batch 128，最多 500 epochs，patience 30。相对 formal-v2.1 的明确变化是把 row-level validation loss 改为逐猫概率平均后的未加权猫级 CE，并统一采用 `1e-6` 严格改善阈值。

每个 pipeline×repeat×fold×seed 先只在既有 train/validation 角色上训练并选 epoch。随后从头重新初始化，在 train+validation 上按锁定 epoch 固定轮数重训；所有标准化、缺失填补与类别权重均重新只拟合该阶段训练侧。outer checkpoint 的权重、预处理与训练历史写盘并锁定后，才允许一次 test 概率推理。

## 4. 预算与门禁

- Selection：6×3×4×3 = 216 fits；不产生 outer-test 预测。
- Outer refit：6×3×4×3 = 216 fits；每个 fit 只在 checkpoint lock 后进行一次 test 推理。
- 总预算：432 physical fits。
- 每个模型产生 3 repeats×3 seeds = 9 个完整 111-cat OOF 预测集。

执行顺序固定为：CPU preflight → selection 授权 → 216 fits 完成 → epoch-selection lock → 独立 lock 审计 → outer 授权 → 216 refits → aggregate → 独立全量复算。任何 resume 都必须核验协议、runner、tests、输入、角色、fit 身份和所有引用产物哈希。

## 5. 主表、比较与不确定性

每个模型先分别计算 9 个完整 111-cat OOF 的 Accuracy、Macro-F1、Balanced Accuracy、kitten/adult/senior recall、CE 与 Brier，再报告 9 个值的均值与样本 SD。不得用逐折 F1 均值代替完整 OOF F1，也不得先跨 seed 平均概率后把集成结果作为单模型主成绩。

预定 9 组比较：

1. C1−A0；
2. C1−U1；
3. C1−D0；
4. U1−A0；
5. D0−A0；
6. U1−D0；
7. VGG129−VGG128；
8. A0−VGG129；
9. C1−VGG129。

其中 U1−D0 比较的是无界残差与直接拼接这两套整体融合方案；C1−U1 比较的是 RMS 相对尺度与 `tanh` 限幅这一整组约束，不拆分成单个组件的因果效应。

描述性区间使用固定 seed `20260920` 的 10,000 次年龄组内猫级分层配对 bootstrap。每次同一猫抽样同时作用于全部模型和全部 9 个 OOF 预测集；每个预测集重算指标后再取模型差。报告 2.5%/97.5% 分位数，不做独立样本 t 检验，不以多重比较挑选“显著”结果。

999 次预测出现来自同一 111 只猫，不是 999 个独立动物。训练 seed 与划分波动通过 9 个完整 OOF 值及配对差另行报告。

## 6. 复用与解释边界

只复用 formal-v2.1 角色、冻结 AST call embedding、20 维声学缓存以及已审计模型结构/训练语义。历史权重、checkpoint、预测和成绩不进入 IDEA-089 新主表。由于 VGG 猫级 CE monitor、VGG129、AST 训练配方及配对初始化均构成本轮新条件，历史结果只能作为带日期和协议标签的背景。

VGGish CSV 的 936 个 embedding rows 没有可靠 row→call 映射，因此保持 row→cat 概率平均；不得捏造 call 对应关系。AST 以 792 calls 为原生单位，再做 call→cat 概率平均。

参数量、selection 训练时间、outer 重训时间和一次预测时间分别报告。本轮计时不包含最初音频清洗、声学特征提取或 AST 预计算，不能称为完整端到端系统成本。
