# IDEA-086 声学集合残差：独立文献、历史差异与方法审查

日期：2026-09-19  
审查角色：牛马1（独立只读方法审查）  
结论：**方法边界清楚，可进入锁定实现与 CPU 预检；正式训练仍未获本报告授权。**

## 1. 一手文献核验

### Deep Sets

Zaheer et al., *Deep Sets*, NeurIPS 2017 的官方摘要明确研究定义在集合上的机器学习问题、排列不变目标，以及具有特定分解结构的一类集合函数；论文展示的任务包括总体统计估计、点云分类、集合扩展和异常检测。[官方论文页](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)

IDEA-086 的 `逐帧共享 MLP → masked mean → projection` 与简单排列不变集合处理具有邻近关系，但本轮必须保持以下边界：

- 只能称为受集合模型启发的、共享逐元素映射加均值池化的排列不变声学残差。
- 本实现使用 **mean**，没有显式保留集合基数；也没有为完整集合函数普适逼近建立本任务条件。因此不能声称“完整声学分布”、普适逼近或新发明 Deep Sets。
- 本轮没有比较 sum、mean、max、variance、attention 或更宽的 set encoder，不能把结果外推到全部集合架构。

### Attentive Statistics Pooling

Okabe, Koshinaka and Shinoda, *Attentive Statistics Pooling for Deep Speaker Embedding*, Interspeech 2018 的官方页面说明：其说话人验证方法对帧赋予注意力权重，并同时生成加权均值与加权标准差；论文报告的是 NIST SRE 2012 与 VoxCeleb 上的 EER。[ISCA 官方论文页](https://www.isca-archive.org/interspeech_2018/okabe18_interspeech.html)

该工作只提供“从帧级表示得到 utterance-level 表示是已有声学建模路线”的邻近动机。IDEA-086 使用均匀 masked mean，不使用 attention、std/variance pooling，也不复制说话人验证任务或其性能结论。

## 2. 锁定候选及参数核算

候选为 `SET1_acoustic_set_residual`：

1. 输入沿用 IDEA-085 的六个原始轨迹通道 `log_f0, voiced_probability, periodicity, log_rms, spectral_tilt, spectral_flatness`，再拼接六个 finite indicator，共 12 通道。
2. 每一帧独立通过共享 `Linear(12,48)+GELU → Linear(48,48)+GELU`。
3. 仅对真实帧做 masked mean；padding 不进入池化。
4. `Linear(48,128)` 为零初始化，将集合 context 映射成残差。
5. 与 C1/T1 一致，融合为 `h' = h + 0.25 × stopgrad(RMS(h)) × tanh(r_set)`；AST 冻结，不再叠加 C1 summary 分支。

参数独立核算：

- `12→48`：`12×48+48 = 624`；
- `48→48`：`48×48+48 = 2,352`；
- `48→128`：`48×128+128 = 6,272`；
- 集合分支合计 `9,248`；总可训练参数 `99,075+9,248 = 108,323`；比 IDEA-085 T1/J1 的 `108,355` 少 `32`。

该差异很小但不是严格等参。SET1−T1/J1 因而只能作为预设辅助比较，不能包装成纯粹的“是否使用顺序”单变量因果检验。

## 3. 与既有本地实验的真实差异

### IDEA-051：同猫多 call 集合

IDEA-051 把一只猫的多个 `768` 维 frozen-AST **call embeddings** 组成集合，以 animal 为训练单位，比较 hidden mean 与 learned call attention。它改变训练/预测单位，并在猫级集合上优化 animal loss。IDEA-086 则在**单条 call 内**汇聚原始声学帧，仍保留 call-level 训练与最终 call-probability-to-cat mean。因此二者的集合元素、监督单位、特征空间和 pooling 层级均不同。

IDEA-051 的 set 候选明显低于 call-probability mean reference；该结果不能直接预测 IDEA-086，因为 IDEA-086 没有把 111 cats 变成仅有的训练 bags，也不在分类前合并同猫 calls。

### IDEA-052：AST temporal token 残差

IDEA-052 在单条 call 内使用 `5,842` 个 frozen AST final-layer temporal tokens，每个 token 为 `768` 维；R1 使用 temporal hidden mean 相对 global hidden 的 mean-shift，R2 使用 max-minus-mean salience，再以逐维零初始化 gate 加回 AST global path。IDEA-086 使用的则是六个解释性声学轨迹及其缺失指示，不读取 AST temporal tokens，不计算 global-hidden 差，也不使用 max-minus-mean。

两者都包含 call 内聚合，但输入来源、维度、残差定义与参数化不同。IDEA-052 说明 AST token aggregation 已经做过；它不等于当前的原始声学帧集合。

### IDEA-079：AST patch-token 内部适配

IDEA-079 在 AST block 11 与 12 之间处理 `12×12` patch-token 网格，比较逐点 `1×1` 控制与 `3×3` depthwise 空间卷积；它修改的是 AST 内部 patch 表示路径。IDEA-086 保持 AST embedding 路径冻结，独立读取 call 对齐的原始声学轨迹，以外部有界残差注入 128 维 head hidden。IDEA-079 的 patch 空间邻接与 IDEA-086 的声学帧集合不是同一表示层或同一顺序概念。

### IDEA-085：直接父实验

IDEA-086 与 IDEA-085 共用轨迹 cache、六个通道、finite indicators、train-only frame statistics、base seeds `[2713,5395,5226]`、3 repeats、4 folds、训练/检查点规则与 `0.25×RMS` 残差上限。主要变化是：

- T1/J1：`Conv1d(12,32,k5) → Conv1d(32,32,k3) → masked mean`，对真实局部顺序敏感；J1 使用一个固定联合帧置乱。
- SET1：共享逐帧 MLP 后 masked mean，不含位置、卷积、attention 或 variance pooling，理论上对同一帧 tuple 的任意排列不变。
- IDEA-086 只新增 SET1 的 36 fits；A0/C1/T1/J1 的 144 fits 与预测均从 IDEA-085 只读复用，不重训、不选择性替换。

## 4. 必须满足的 CPU 预检项目

### 数据与预处理

- 轨迹 cache 必须继续对齐 792 calls、57,848 frames，六通道顺序不变；20 个无有效 F0 的调用及所有缺失 F0 帧仍保留。
- finite indicator 必须与对应六通道值作为整帧同行处理；不能删除帧、拼接 valid-F0 子序列或逐 call 居中。
- 原始通道的 finite median、imputation、mean/std 只能由当前 cell 的 training calls 全部真实帧拟合；validation/test 不参与，indicator 不标准化。
- 数字 call lookup 和可选时间轴只能用于 ragged retrieval/审计，不能进入逐帧 encoder。

### 非空洞排列不变性

零初始化 projection 会令最终残差恒为零，因此只比较初始 logits 是无效测试。CPU 必须至少完成：

1. 给逐帧 encoder 非零随机参数，比较同一真实序列的原序、逆序和 IDEA-085 J1 固定联合排列所得 **pooled context**；
2. 再使用非零 projection 探针，比较 residual 与最终 eval logits；
3. 在一个混合短/长序列的 padded batch 中重复检查，确认 padding 不参与任何 context；
4. 报告最大绝对差并使用明确 float 容差，不要求 bitwise 相等；
5. 首个已训练 SET1 fit 完成后，再对真实 checkpoint 做一次实际排列不变性复核。

### 初始化、梯度与上限

- zero-init projection 下 SET1 logits 必须与 A0 相同；公共 AST head 初始状态必须与同 cell 的 IDEA-085 路径一致。
- zero-init 时 projection 应可获梯度；在固定非零 projection 探针下，两层逐帧 Linear 都必须梯度可达。
- `0.25` 共享残差相对 RMS 上限须在真实 mixed-length batch 与饱和探针中通过。
- padding 必须在 masked mean 中完全排除；修改 padding 值或 batch companion 不应改变有效序列结果。

### 复用公平性与不可变依赖

- SET1 必须使用 IDEA-085 相同 full seed、roles、cat/call coverage、训练 RNG、batch 顺序、loss 与 animal-CE checkpoint selection；除新增集合分支外不改变 head。
- 正式前逐项核验 IDEA-085 protocol/runner/tests、run manifest、144 fit summaries 及 288 prediction CSV 的 SHA，并从 call CSV 重建 animal CSV、复核 role alignment、概率与 outer-test 标志。
- 总监在 2026-09-19 12:45:29 建立的 450 文件只读基线组合 SHA-256 为 `161e2357e544eaffed62b458f8d25eedc4b360a60000f02ca36c09fa7db89af3`；收尾必须以同一清单/算法复核不变。

## 5. 预注册比较与解释

- 主 classification gates：`SET1−A0` 与 `SET1−C1` 各自沿用 IDEA-085 的五条件：mean seed×repeat Macro-F1 delta `≥0.005`、至少 `2/3` base-seed 均值为正、至少 `6/9` seed×repeat 为正、至少 `8/12` split 非负、worst split `≥−0.03`。
- 预设辅助比较：`SET1−T1` 与 `SET1−J1`，完整报告但不设 gate。
- 四条比较均报告 Accuracy、Macro-F1、BA、kitten/adult/senior recall、CE、Brier、纠错转移、3/9/12 稳定性，以及 `612 repeated occurrences` 与 `97 unique cats` 的区别。
- 不存在单一全局 pass；CE/Brier 与其他辅助轴不进入主 classification gate。

允许的最强解释仅限于这个固定的局部声学集合残差：

- SET1−A0/C1 若通过，只说明该锁定的共享逐帧非线性加均值残差在当前 111-cat 内部探索中达到预设门槛。
- SET1−T1/J1 不能单独证明“顺序无用”或“分布优于时序”。SET1 与卷积分支参数化不同，J1 也只是一个固定联合置乱控制。
- SET1 的排列不变性只针对在**同一个冻结 AST call embedding 条件下**输入辅助分支的声学帧集合；它不表示把原始音频打乱后整个 AST+SET1 系统仍不变。冻结 AST 主路径自身仍可携带时间结构。
- 无论结果如何，都不能声称完整声学分布建模、普适集合近似、纯 F0 机制、外部确认或 Deep Sets 新方法。

## 6. 审查决定

固定候选在概念上与文献及本地历史可区分，参数核算自洽，比较与 gate 边界充分。**可进入实现锁定和 CPU 独立预检；在非空洞排列不变性、旧预测完整性、公平初始化/RNG、padding/梯度/cap 等证据全部通过前，不应启动 GPU。**
