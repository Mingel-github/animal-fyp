# IDEA-075：犬年龄任务上的 C1 年龄敏感 AST 探索性迁移

## 研究定位

本轮检验的不是“猫模型权重能否直接用于狗”，而是一个更有限的方法迁移问题：在完全不同物种、并按犬只身份隔离的五分类年龄任务上，冻结 AST 表示与20维年龄敏感声学特征之间的 C1 有界残差融合，是否仍然优于参数匹配的 AST-only 和无界残差对照。

IDEA-074 后，C1 **仅保留为探索性候选**，不是已经确认的主模型。本轮也不是独立确认实验：犬数据的 AST-only 基线已经在 IDEA-067 中观察过，而且当前只使用固定的2,290-unit资源受限子集。因此，即使通过 gate，也只能称“C1 结构获得跨物种、跨犬只的探索性支持”，不能称为犬年龄 benchmark 已被解决，更不能把结果外推到完整79,142-unit数据。

本计划只预注册数据、模型、预算与判断规则，不启动特征提取或正式训练。

## 已知数据事实与标签语义

本轮沿用 IDEA-067 已锁定的 Canine Age Transition 固定子集：

- 原始元数据共有79,142个 bark units、125只狗；
- 子集在每个 `dog_id × age_group` 单元内按 `barkunit_audio` 排序，确定性地选取至多10条，得到2,290个 bark units、125只狗、266个 `dog_id × age_group` 单元；这应称为“路径排序后均匀取点”，不能未经数据说明写成真实时间轴上的均匀采样；
- 目标直接使用数据提供方的 `age_group`，固定顺序为 `puppy / juvenile / adolescent / adult / senior`；不根据 `age_months` 重新分箱；
- 固定子集的类别分布分别为388、444、713、553、192个 bark units，对应45、54、82、64、21只狗；
- 同一只狗可能跨越多个年龄阶段：42只狗出现1个阶段、41只出现2个、27只出现3个、14只出现4个、1只出现5个阶段；
- 元数据中的类别月份范围互有重叠，因此本轮只把 `age_group` 当作数据提供方给出的五分类标签，不自行声称其采用统一月份阈值或某种犬种校正规则；正式论文解释标签构造前，必须补读并引用数据集官方说明。

### 样本、标签与独立单位

- **观察/预测单位**：单个 `bark_unit_id`；主要 Macro-F1 在 bark-unit OOF 预测上计算。
- **标签单位**：该 bark unit 所属采集时点的 `age_group`；同一狗在不同采集时点可有不同标签。
- **统计独立与切分单位**：`dog_id`。同一只狗的任何年龄阶段、视频、bark sequence 或 bark unit 都不得横跨 fit、validation、test。
- **辅助聚合单位**：`dog_id × age_group`。可报告先对同一单元概率取算术平均后的五分类指标，但266个单元不能被宣称为266个相互独立个体。
- 2,290个 bark units 不能当作2,290个独立生物学样本；跨种子/重复的 OOF 汇总和逐犬方向才是本轮稳定性判断的依据。

`age_months`、`breed`、GMM transcript、文件名、路径、`dog_id`、`video_id` 和 bark sequence 标识均不得作为模型输入。它们只能用于审计、分组或结果分层，防止标签捷径。

## 现有资产能否复用

| 资产 | 结论 | 用法与边界 |
|---|---|---|
| IDEA-067 固定 manifest | 可直接复用 | 锁定 `runs/idea067_external_ast_representation_v1/canine/manifest.csv`，SHA-256 为 `591b4f5c7b2d69ac76f297b3121dd185f962cc24ba8ccb99cb159d01f5098ad4`。不得按结果重采样。 |
| 犬 AST embedding | 可直接复用 | `ast_standard_recording_embeddings.npz` 含2,290×768表示，SHA-256 为 `129ab12adc68f4d77323867851dc093c896dc0a561cc1706d1fea5163ac45344`。它由冻结的 AudioSet-AST、1.28秒窗/0.64秒 hop及 recording 内算术平均生成。 |
| 猫的20维声学特征 NPZ | **不可直接复用** | 其行只对应792条 MeowAgeNet 猫叫。可复用特征定义和无标签提取代码，但必须从犬音频重新提取并建立新的 recording-id 哈希锁。 |
| C1 已训练参数 | 不复用 | 本轮迁移的是公式与归纳偏置，不是猫分类头或猫数据上学到的权重。A0、U1、C1 均从同一新随机初始化在犬训练角色上拟合。 |
| IDEA-067 logistic AST 结果 | 只作历史参照 | Macro-F1 为 `0.2127±0.0073`，dummy 为 `0.0950`；由于新神经管线还需内部 validation 作早停，训练角色不完全相同，不把该数值作为 C1 的正式配对 gate。 |

已完成的只读技术核对显示：2,290个固定样本均为16 kHz、单声道、PCM-16；时长最小0.076秒、中位0.252秒、最大7.708秒，其中31条短于0.1秒。manifest 与 AST cache 的 recording、dog、source path 顺序逐项一致，2,290个音频 SHA-256 全部唯一，未发现跨犬重复音频。

## 犬声学特征提取

冻结复用 IDEA-068 的20维定义，不在犬结果上修改频率范围、窗口或特征集合：

```text
F0位置/离散/动态：log_f0 q10/median/q90/IQR/std/MAD/slope/|delta| median
周期与声质：period variation、voiced fraction、voiced-probability mean/std、
             periodicity median、HNR median、amplitude variation
能量与频谱：log-RMS median/IQR、spectral-tilt median/IQR、spectral-flatness median
```

固定设置为16 kHz、`librosa.pyin`、F0范围60–2000 Hz、1024点 frame、160点 hop、200–4000 Hz谱倾斜范围。特征由原始 bark unit 波形计算，不使用标签；短音频不重复拼接来人为增加 F0 观测。无法估计的值保留为 NaN，随后只用当前训练角色的中位数填补并只用当前训练角色的均值/标准差标准化。若某一训练角色整列均缺失，则固定以0填补、scale=1，并在审计中显式记录。

当前2,290条音频无需重采样。若未来完整数据审计发现其他采样率，则先转单声道，再使用与 IDEA-068 相同的 polyphase 方法重采样到16 kHz；不得因为结果不佳而改变 F0 上下限。若完整数据的编码或采样率不兼容，则须在任何标签评估前单独修订并重新锁定提取协议和源文件哈希。

## 配对管线

所有神经管线共享训练角色标准化后的768维冻结 AST embedding：

```text
h = ReLU(W_ast x),  W_ast: 768→128
```

之后经过同一个 BatchNorm、dropout和 `128→5` 分类头。

### A0-dog：AST-only匹配对照

```text
h_A0 = h
```

### U1-dog：参数匹配的宽无界年龄残差

```text
c = GELU(W_age a),  W_age: 20→60
u = W_r c,           W_r: 60→128
h_U1 = h + u
```

### C1-dog：冻结公式的有界宽年龄残差

```text
c = GELU(W_age a)
q = stopgrad(sqrt(mean_j(h_j^2) + 1e-8))
r = 0.25 × q × tanh(W_r c)
h_C1 = h + r
```

U1 与 C1 的年龄输出层权重和偏置均零初始化。相同完整种子下，U1/C1 的全部同形参数初值必须逐张量相同，A0/U1/C1 的优化前 logits 必须完全一致；构造模型后重置训练 RNG，以配对 dropout、dog batch 顺序和优化随机流。

五分类下预期可训练参数为 A0 `99,333`，U1/C1 各 `108,401`；runner 单元测试必须重新计算并锁定，任何不一致均在训练前停止。U1 用于区分“加入年龄声学分支”与“0.25×RMS相对有界”两种贡献，不能省略后再把 C1 增益归因于有界机制。

训练折类别先验 dummy 使用同一 test fold，并使用完整 outer-train（fit+validation）角色的标签频率；dummy 没有需要 validation 选择的参数，这一定义与 IDEA-067 的训练折先验一致，也避免因人为丢弃 validation 标签而削弱对照。IDEA-067 的平衡多项逻辑 AST probe保留为历史参照，不与新管线混成同一模型族。

## 无泄漏切分与固定训练

外层完全复用 IDEA-067 的三组 `StratifiedGroupKFold(5)`：split seeds 为 `[17, 43, 101]`。每个 repeat 的五折合并成覆盖全部2,290条记录的一套 OOF 预测。正式 runner 必须从 IDEA-067 预测文件重建并核对 `(repeat, recording_id, dog_id, fold)`，而不是生成一组更有利的新外层划分。

每个 outer-train 内再固定一次按 `dog_id` 分组的四折切分：

```text
StratifiedGroupKFold(n_splits=4, shuffle=True,
                     random_state=outer_split_seed + 1000 + outer_fold)
取第0折为 validation，其余为 fit
```

validation 只用于选择最低“逐犬等权交叉熵”的 checkpoint；test 在 checkpoint 冻结后只预测一次。为了保持流程简单且避免新增二阶段训练规则，本轮不在选择 epoch 后用 outer-train 重训。已做只读重建：15个 outer fold 的 fit/validation/test 均无犬只交叉且五类齐全；正式执行前仍须把角色表落盘、哈希锁定并由 runner 再验证一次。

固定沿用 MeowAgeNet 年龄模块的训练设置：Adamax、学习率0.006、epsilon `1e-7`、gradient clip 1.0、最多50 epochs、patience 8、dropout `0.44571035356880917`、每个 batch 最多4只完整的狗、按 fit bark-unit 类别频率计算的 balanced cross-entropy。必须实现犬专用 `DogSetDataset`：一个 sample 是一只狗在当前角色中的全部 bark units，但每条 bark 保留自己的 `age_group` 标签；不得复用假设“一只动物只有一个标签”的猫 `CatSetDataset`。每批损失按所有 bark 的“类别加权损失和／类别权重和”计算。loader 必须确定性地避免单狗末批：当余数为1时将倒数第二批与末批重新分成3/2或等价的无单狗方案；允许末批为2或3只狗。不得让同一狗的一部分单位进入 validation/test。

validation checkpoint 的“逐犬等权交叉熵”固定定义为：先对每条 bark 取真实类别负对数概率，再在每只狗内部求均值，最后对 validation dogs 等权平均。不得先把同一狗的概率平均后赋予单一标签，也不得调用要求单动物单标签的 `calls_to_animals` 或 `animal_cross_entropy`。

所有 AST/年龄特征的标准化量、NaN填补量、类别权重和 checkpoint 判断量只由当前 fit 角色计算。外层 test 不参与早停、阈值、缺失值处理、标准化或任何超参数选择。

## 新训练种子与计算预算

固定字符串：

```text
IDEA-075-dog-C1-age-sensitive-AST-benchmark-v1
```

其 SHA-256 为：

```text
3b3e238bb521156e998999e095d0a0ffb8118182d3181fccd56cc72c9dfe75ff
```

按连续大端 uint32 对10000取模并跳过历史基础种子冲突，前三个值为：

```text
base seeds = [8075, 4270, 1872]
```

完整训练种子为 `base_seed + 10000×repeat_index + 100×outer_fold`。45个完整种子互不重复，且当前审计未发现与 IDEA-068 至 IDEA-074 已登记完整种子冲突。

正式预算：

```text
3 neural pipelines × 3 base seeds × 3 split repeats × 5 outer folds
= 135 neural fits
```

dummy 不计神经 fit；特征提取一次。禁止追加“直到变正”为止的新种子。

## 指标与汇总层级

主要指标为 bark-unit OOF Macro-F1。每个 `base_seed × split_repeat` 先合并五个 outer test fold，形成9个等权主分析单元；不得把45个 fold 或6,870/20,610行重复 OOF 预测伪装成独立重复。

同时报告：

- balanced accuracy、五分类 CE、multiclass Brier；
- 每类 recall，特别是 senior recall；
- 逐犬 bark-unit accuracy 的等权平均；
- 每个 `dog_id × age_group` 内概率算术平均后的 Macro-F1/ balanced accuracy；
- C1−A0、U1−A0、C1−U1 的逐犬 CE差值与 accuracy差值正/平/负计数；
- 15个共同 `repeat × outer_fold` 单元先跨三个基础种子平均后的配对差值及最差单元；
- C1 的 `||r||/||h||` 分布、最大逐维预算违规数；机制审计不参与选模。

## 预注册 gate

### 第一层：C1 整包年龄敏感融合相对 A0 是否有效

全部条件同时满足才通过：

1. 9个 seed×repeat 等权的平均 C1−A0 Macro-F1 至少 `+0.005`；
2. 3/3基础种子的平均差值严格大于0；
3. 至少6/9个 seed×repeat 差值严格大于0；
4. 至少10/15个共同 split-cell 差值不小于0；
5. 最差共同 split-cell 不低于 `-0.03`；
6. C1 的等权平均 CE 与 Brier 均不劣于 A0；
7. C1−A0 的平均 balanced-accuracy 差值不小于0；
8. 每个基础种子的 pooled senior-recall 差值不低于 `-0.02`。

### 第二层：C1 的“有界”机制是否优于 U1

只有第一层通过后才解释这一层：

1. 平均 C1−U1 Macro-F1 严格大于0；
2. 至少2/3基础种子的平均 C1−U1 严格大于0；
3. 至少5/9个 seed×repeat 的 C1−U1 严格大于0；
4. C1 的等权平均 CE 与 Brier 均不劣于 U1。

第一层通过、第二层失败时，只能说“20维年龄敏感声学融合在犬任务上有探索性价值”，不能说 RMS-relative tanh bound 得到跨物种验证。第一层失败但存在正向均值或局部优势时，仍完整报告为正向但未过门槛的结果。

### 第三层：外部任务下限

C1 必须在9/9个 seed×repeat 中都高于对应训练先验 dummy 的 Macro-F1，才可使用“稳定高于先验基线”的表述。与 IDEA-067 logistic AST `0.2127` 的比较只作描述，因为其训练/验证流程不同，不作为 C1−A0 配对 gate。

即使三层全部通过，本轮仍为探索性子集结果。只有另行预注册并运行完整79,142-unit协议，才可讨论更强的犬 benchmark 结论。

## 执行前审计与停止规则

正式训练前必须完成并保存以下审计：

1. 锁定元数据、manifest、2,290个源音频、AST cache、IDEA-068特征定义、runner、roles CSV及协议哈希；
2. 验证 manifest、AST cache、犬声学特征的 recording-id 一一对应且顺序一致；
3. 无标签地提取20维犬特征，报告每维 NaN数、全有限记录数、全缺失列和每条记录有效帧数；出现读取失败、Inf、ID缺失/重复或源哈希漂移时停止；
4. 重建15个外层和15个内层角色，确认 dog overlap 为0、每个 fit/validation/test 均含五类，并锁定每角色的犬数与类别数；
5. 验证A0/U1/C1参数量、初始同形状态、优化前 logits、每epoch dog顺序和 recording coverage 哈希；
6. 验证所有 C1 预测中逐维 `|r_j| ≤ 0.25×RMS(h)`，并报告扰动强度；
7. 外层 test 预测一旦生成，不得修改特征、cap、宽度、优化器、种子、split或 gate。

样本级技术事实已足以制定当前子集协议，但以下信息在扩展到完整79,142 units前仍不足：完整 shard 的采样率/编码覆盖、完整音频可提取率、完整数据的重复音频情况，以及官方 `age_group` 构造规则。它们必须先做只读清单与音频头/哈希审计；若不兼容，再在任何完整集标签评估前发布带理由的新协议版本。

## 结果解释与下一步

- 第一层与第二层都通过：C1 仍只升级为“值得进入完整犬数据确认”的探索性候选；冻结公式后制定完整集协议。
- 第一层通过、第二层失败：保留年龄声学融合方向，C1 不获得优于无界 U1 的机制主张。
- C1−A0 为正但未过 gate：如实保留效应、稳定性和失败条件，不重调 `0.25` cap或F0范围。
- C1 与 U1 都未优于 A0：说明当前猫数据上提出的年龄归纳偏置未在犬子集迁移；不以类别重采样、标签重分箱或只挑 senior split 挽救。
- 只有在本轮基本有效性通过后，才进入完整犬数据；CatMeows 情境保持性和 AST 内部末层注入仍是相互独立的后续问题。
