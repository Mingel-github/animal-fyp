# IDEA-012 / IDEA-048 Pilot 讨论回传稿

日期：2026-08-26
实验协议：`meowagenet-locked-v1`
用途：供本轮结果讨论、导师回传和下一阶段路线选择使用。

## 1. 结论摘要

本轮在固定的 cat-ID folds 和 animal-level 评价协议下，完成了 VGGish+MLP、
标准 AST、IDEA-012 的 time-fine AST、等 token 数的 frequency-fine 对照，
以及只根据 inner validation 逐折选择 AST 变体的 nested-selected pipeline。

当前结论分为两层：

1. **IDEA-012 在当前 pilot 中没有获得预期的净提升。** time-fine AST 的
   animal-level macro F1为0.6469，标准AST为0.6525，差值为-0.0056，因此当前
   结果更支持保留standard geometry。不过，time-fine明显好于同token数的
   frequency-fine，提供了一个有用的方向性信号：若增加patch密度，时间方向比
   频率方向更符合这批数据。
2. **IDEA-048 得到的是“接近VGGish、优势维度不同”的结果。** nested-selected
   AST的primary macro F1为0.6666，VGGish+MLP为0.6846，相差1.8个百分点，位于
   预设的`±0.03`实践差异范围内；同时nested AST取得最高balanced accuracy，
   standard AST取得最高QWK。

总体而言：**VGGish目前在精确的三分类综合表现上领先；AST在类别召回平衡、
senior识别和有序年龄结构上显示出优势。表格呈现的重点不是AST“有没有价值”，
而是两类音频表示形成了清楚、可利用的互补错误画像。**

## 2. 本轮回答的研究问题

### IDEA-012

研究问题：在其他条件相同的情况下，将 AST 的时间 stride 从10减小到5，增加
时间方向的 patch 密度，是否能够提高猫年龄三分类性能？

核心比较：`AST time-fine − AST standard`。

### IDEA-048

研究问题：在统一、无 cat-ID 泄漏的协议下，通过 inner validation 选择 AST
候选变体后，最终 AST pipeline 是否能够超过 VGGish+MLP baseline？

核心比较：`AST nested-selected − VGGish+MLP`。

这里的“超过”以事前冻结的 primary metric——animal-level macro F1——判断。
Balanced accuracy 和 QWK 用来解释模型行为，不在结果出来后替换主指标。

## 3. 数据口径

数据中几个数字代表不同层级，不能互换：

| 数字 | 含义 | 本轮用途 |
| ---: | --- | --- |
| 937 | 官方 VGGish CSV 中的 embedding 行数 | 原 notebook 复现参考 |
| 112 | 官方发布标签中的 `cat_id` 数 | 原 notebook 复现参考 |
| 793 | 清理前的原始音频条目数 | 数据来源审计 |
| 936 | 排除重复别名 `049A` 后的 VGGish embedding 行数 | 锁定协议的 VGGish 输入 |
| 111 | 去重后的真实分析动物数 | 正式划分和 primary evaluation 单位 |
| 792 | 去重后的唯一原始叫声数 | AST 输入和 manifest 主体 |
| 843 | 792段叫声经过1.28秒滑窗后形成的 AST segment 数 | AST encoder 推理单位 |

937不等于猫数，也不等于原始叫声数：VGGish CSV 的一行是一个 embedding
样本，一段较长叫声可能对应多个 embedding。正式比较必须先在同一只猫内部
聚合预测，再对111只猫计算指标，避免“叫声多的猫”在评价中获得更大权重。

类别动物数为：kitten 15只、adult 62只、senior 34只。类别明显不均衡，这也是
同时报告 macro F1、balanced accuracy 和逐类 recall 的原因。

## 4. 冻结实验流程

1. 依据官方数据 manifest 和 checksum 确认输入文件，排除重复别名 `049A`。
2. 固定4个 outer cat-ID folds；每折测试27～28只猫，同一只猫的所有叫声只能
   出现在同一侧。
3. 每个 outer-training 集内部再划出17只猫作为 inner validation，用于早停和
   AST 变体选择；outer test 不参与选择。
4. 分别提取清洗后的 VGGish embedding，以及 standard、time-fine、
   frequency-fine 三种冻结 AST embedding。
5. 所有表示使用同一 MLP 分类头和同一 seed 规则训练，降低分类器差异带来的
   干扰。
6. 将同一 `cat_id` 的类别概率取均值，再生成每只猫的最终预测。
7. 一次性计算 outer animal-level 指标，并对主要差值进行按动物配对的
   bootstrap。

正式协议要点：

- Primary unit：animal；
- Primary metric：macro F1；
- Secondary metrics：balanced accuracy、QWK、逐类 recall；
- 实践差异阈值：`δ_task = 0.03` macro F1；
- 训练 seed：`42 + outer_fold`；
- MLP：Dense(128) + BatchNorm + Dropout，Adamax，class weights balanced；
- early stopping：监控 inner `val_loss`，patience 30；
- AST encoder：冻结，本轮不是完整 encoder fine-tuning。

## 5. AST 配置与计算量对照

AST checkpoint 为 `MIT/ast-finetuned-audioset-10-10-0.4593`。音频重采样到
16 kHz；使用1.28秒窗口和0.64秒 hop。长叫声的多个 segment embedding 先在
call 内平均。

| 变体 | Frequency stride | Time stride | Patch tokens | 角色 |
| --- | ---: | ---: | ---: | --- |
| AST standard | 10 | 10 | 144 | 标准 AST geometry |
| AST time-fine | 10 | 5 | 276 | IDEA-012 主候选 |
| AST frequency-fine | 5 | 10 | 276 | 与 time-fine 等 token 数的方向对照 |

三种 AST 均使用16×16 patch kernel，并复用相同 checkpoint 的 patch projection
和 Transformer encoder 权重；位置编码网格根据输入 geometry 做双线性插值。
encoder 可训练参数为0。

2026-08-25版冻结特征实际在CPU上提取，耗时约为：standard 43.93秒、time-fine 90.79秒、
frequency-fine 92.21秒。time-fine 的 token 数和耗时约为 standard 的两倍，
但本轮没有换来 primary metric 增益。

## 6. Animal-level 主结果

| Pipeline | Macro F1 | Balanced accuracy | QWK | Kitten recall | Adult recall | Senior recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VGGish+MLP | **0.6846** | 0.6754 | 0.5263 | 0.6667 | **0.7419** | 0.6176 |
| AST standard | 0.6525 | 0.7063 | **0.5723** | **0.8000** | 0.6129 | **0.7059** |
| AST time-fine | 0.6469 | 0.7018 | 0.5474 | **0.8000** | 0.6290 | 0.6765 |
| AST frequency-fine | 0.6167 | 0.6670 | 0.4729 | **0.8000** | 0.6129 | 0.5882 |
| AST nested-selected | 0.6666 | **0.7082** | 0.5432 | **0.8000** | 0.6774 | 0.6471 |

Nested selection 只依据 inner validation：Fold 0选择standard，Fold 1选择
time-fine，Fold 2选择frequency-fine，Fold 3选择standard。

四折 macro F1：

| Pipeline | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
| --- | ---: | ---: | ---: | ---: |
| VGGish+MLP | 0.8125 | 0.5937 | 0.6966 | 0.6183 |
| AST standard | 0.7737 | 0.6232 | 0.6653 | 0.5844 |
| AST time-fine | 0.7737 | 0.7078 | 0.5788 | 0.5934 |
| AST frequency-fine | 0.6595 | 0.5974 | 0.6016 | 0.5844 |

## 7. 主要差值与结论边界

| 比较 | Macro F1 差值 | 动物分层配对 bootstrap 95%区间 | 当前判断 |
| --- | ---: | ---: | --- |
| Time-fine − Standard | -0.0056 | [-0.0557, 0.0421] | IDEA-012 无提升信号 |
| Nested AST − VGGish | -0.0180 | [-0.1100, 0.0805] | 性能接近，但未超过 baseline |
| Frequency-fine − Standard | -0.0359 | [-0.0904, 0.0128] | 频率方向更细没有优势 |

区间较宽，说明111只猫对几个百分点差异的分辨能力有限。本轮最贴合数据的描述是：
IDEA-012未出现预期净提升；IDEA-048与VGGish点估计接近，同时呈现balanced
accuracy和QWK方面的优势画像。区间信息用于表示这种结论仍有较大抽样不确定性。

## 8. 读表前需要理解的术语

### Pipeline

这里的 pipeline 不是单指分类头，而是从音频输入到动物预测的完整路径：

`音频/已有特征 → VGGish或AST表示 → 同一个MLP分类头 → 叫声概率 → cat-ID内聚合`。

各 pipeline 使用同一个 MLP 和同一套 folds，主要区别来自前面的音频表示及 AST
geometry。因此表格比较的重点是：不同表示把哪些年龄信息保留下来、又容易混淆
哪些年龄段。

### Animal-level

一只猫可能有多段叫声，甚至一段叫声还可能被切成多个 segment。Animal-level
评价先把同一只猫的所有预测概率取平均，最后每只猫只贡献一个预测结果。因此，
叫声较多的猫不会比叫声较少的猫拥有更多“投票权”。表中的分母始终是111只猫，
而不是936个VGGish embedding或843个AST segment。

### Recall、precision 与 F1

- **Recall（召回率）**回答“某个真实年龄组里，有多少被模型找出来”。例如 kitten
  recall 0.8000表示15只真实kitten中识别对12只。
- **Precision（精确率）**回答“所有被模型叫作某年龄组的猫里，有多少确实属于
  该组”。模型如果把很多adult也预测成kitten，kitten recall可能上升，但kitten
  precision会下降。
- **F1**是precision和recall的调和平均。只有“找得多”且“叫得准”同时成立时，
  F1才高；其中一项偏低就会拉低F1。

### Macro F1

先分别计算kitten、adult、senior三个F1，再让三个类别各占三分之一后取平均。
它不会因为adult有62只、kitten只有15只，就让adult在总分中占更大比例。本轮将
它设为主指标，是因为它同时考虑每一类的漏判和误报。

### Balanced accuracy

Balanced accuracy是三个类别recall的等权平均：

`(kitten recall + adult recall + senior recall) / 3`。

它集中回答“模型是否能均衡地找回三个年龄组”，不使用precision。它很适合暴露
普通accuracy可能掩盖的少数类问题：如果模型主要认对数量最多的adult，却漏掉
大量kitten，普通accuracy仍可能不低，balanced accuracy则会明显下降。

### QWK

QWK是quadratic weighted kappa，即**二次加权Kappa**。它同时考虑：

1. 真实标签和预测标签的一致程度；
2. 年龄类别具有`kitten → adult → senior`的顺序；
3. 错误跨越了几个年龄级别；
4. 在当前真实类别和预测类别分布下，随机情况下本来会有多少一致。

相邻错误，如kitten错成adult，距离为1；跨两级错误，如kitten错成senior，
距离为2，在二次权重下具有4倍的距离惩罚。QWK可写成“1减去实际加权分歧与随机
期望加权分歧的比值”：1代表完全一致，0代表只达到当前边际分布下的随机一致，
负值表示比随机一致更差。

所以，QWK胜出说明模型不只是做对了多少，还说明它的整体预测更保留年龄顺序。
需要注意，QWK经过随机期望校正，不等于简单数一数跨级错误；两个模型即使
exact正确数接近，QWK也可能因为完整混淆结构和预测分布不同而拉开。

### Nested-selected AST

它不是把三个AST输出平均的ensemble。每个outer fold中，只看该fold的inner
validation结果，在standard、time-fine和frequency-fine之间选一个，再将所选
模型用于该fold的outer test。它代表的是“带内部模型选择的AST流程”，而不是
第四种独立的AST表示。

## 9. 从具体预测数理解表格

下面补充原结果表没有直接展示的混淆数据。每个单元格均为动物数，行是真实类别，
斜线后的顺序固定为“预测kitten / adult / senior”。

| Pipeline | 真实kitten（15只） | 真实adult（62只） | 真实senior（34只） | 总预测正确 |
| --- | --- | --- | --- | ---: |
| VGGish+MLP | 10 / 3 / 2 | 2 / 46 / 14 | 1 / 12 / 21 | **77/111** |
| AST standard | 12 / 3 / 0 | 10 / 38 / 14 | 3 / 7 / 24 | 74/111 |
| AST time-fine | 12 / 3 / 0 | 11 / 39 / 12 | 4 / 7 / 23 | 74/111 |
| AST frequency-fine | 12 / 2 / 1 | 9 / 38 / 15 | 4 / 10 / 20 | 70/111 |
| AST nested-selected | 12 / 2 / 1 | 8 / 42 / 12 | 3 / 9 / 22 | 76/111 |

例如，AST standard的`10 / 38 / 14`表示：62只真实adult中，10只被预测为
kitten、38只预测正确、14只被预测为senior。

各类别的precision与F1进一步揭示了“高recall是否伴随较多误报”：

| Pipeline | Kitten precision / F1 | Adult precision / F1 | Senior precision / F1 |
| --- | ---: | ---: | ---: |
| VGGish+MLP | **0.7692 / 0.7143** | 0.7541 / **0.7480** | 0.5676 / 0.5915 |
| AST standard | 0.4800 / 0.6000 | 0.7917 / 0.6909 | 0.6316 / **0.6667** |
| AST time-fine | 0.4444 / 0.5714 | **0.7959** / 0.7027 | **0.6571 / 0.6667** |
| AST frequency-fine | 0.4800 / 0.6000 | 0.7600 / 0.6786 | 0.5556 / 0.5714 |
| AST nested-selected | 0.5217 / 0.6316 | 0.7925 / 0.7304 | 0.6286 / 0.6377 |

### Macro F1：VGGish的优势来自kitten精度和adult稳定性

VGGish的macro F1为0.6846，是三个类别F1中表现最完整的一条pipeline：

- 对kitten采取相对保守的判断，只预测了13只kitten，其中10只正确，因此kitten
  precision达到0.7692，kitten F1为全表最高的0.7143；
- 62只adult识别对46只，adult recall为0.7419、F1为0.7480，都是全表最高；
- senior是它相对较弱的部分，只识别对21/34只，senior F1为0.5915。

这说明VGGish的决策特点是：**不轻易把猫判成kitten，同时能较稳地维持adult
边界**。它在kitten recall上不如AST，但误报kitten很少，最终获得更高kitten F1；
再加上adult优势，macro F1排在第一。

### Balanced accuracy：nested AST的优势是三个年龄组找得更均衡

Nested AST的三类正确数是12/15、42/62、22/34，对应recall 0.8000、0.6774、
0.6471，平均后balanced accuracy为0.7082，是全表最高。它一共预测正确76只猫，
只比VGGish少1只，但正确结果在三个年龄组之间分布得更均衡。

Standard AST也有相同特征：它把kitten正确数从VGGish的10只提高到12只，把
senior从21只提高到24只；代价是adult从46只降到38只。Balanced accuracy让
每个年龄组各占三分之一，因此kitten和senior的改善足以使它从VGGish的0.6754
升到0.7063。

这里揭示的是两种不同的取舍：VGGish更擅长维持adult和kitten预测的precision，
AST更敏感地检出年龄分布两端。对于不希望漏掉kitten或senior的应用，AST的
balanced accuracy优势具有直接意义。

### QWK：standard AST的优势是年龄顺序一致性

QWK排序为：standard AST 0.5723、time-fine 0.5474、nested AST 0.5432、
VGGish 0.5263、frequency-fine 0.4729。

Standard AST虽然总共预测正确74只，少于VGGish的77只，但QWK更高。这说明在
校正类别和预测分布所产生的随机一致后，standard AST的完整混淆结构更符合
`kitten—adult—senior`这一有序轴。结合逐类结果看，它对年龄两端的识别更强：
kitten识别12/15、senior识别24/34，而且没有把真实kitten直接预测成senior。

因此，QWK胜出带来的信息是：**standard AST更像是在学习连续的年龄相关声学
结构，而VGGish更像是在优化三个离散类别的精确边界。** 这是对模型行为的合理
解释而非机制证明，但它为后续ordinal head或连续年龄建模提供了清楚动机。

### 为什么balanced accuracy高、macro F1却不一定高

Standard AST预测了25只kitten，但实际只有15只kitten；其中12只预测正确，另外
13只是误报，所以kitten recall达到0.8000，而precision只有0.4800。VGGish只
预测13只kitten，找回10只，recall较低但precision达到0.7692。

因此：

- balanced accuracy更奖励AST“多找回了2只kitten和3只senior”；
- macro F1还会看到AST为此产生的kitten误报以及adult漏判；
- QWK进一步询问这些错误在年龄顺序上离正确答案有多远。

三个指标并不冲突，它们是在观察同一张混淆矩阵的三个侧面。

## 10. 各 pipeline / 模块的优势画像

### VGGish+MLP：离散三分类最稳

- Macro F1最高：0.6846；
- 总预测正确数最高：77/111；
- kitten F1和adult F1最高；
- 特别适合作为当前三分类任务的强baseline。

它的主要缺口在senior：senior recall和F1均低于standard/time-fine AST。也就是
说，它的总体分数较高，但对年龄较大猫的覆盖不如标准AST。

### AST standard：序数结构和年龄两端最强，计算上也更经济

- QWK最高：0.5723；
- senior recall最高：0.7059，正确识别24/34只；
- kitten recall并列最高：0.8000；
- 在AST变体中无需增加token；首轮CPU特征提取约43.93秒，约为两个fine变体的一半。

它的主要取舍是把较多adult分到kitten或senior，使adult recall降到0.6129。
因此它体现的是“年龄两端敏感、adult边界较松”的表示特征。

### AST time-fine：时间方向比频率方向更值得保留

- 与standard一样识别对12/15只kitten；
- adult precision为全表最高的0.7959，senior F1并列最高0.6667；
- 在相同276个patch tokens下，macro F1、balanced accuracy和QWK都高于
  frequency-fine。

Time-fine没有超过standard，但它比frequency-fine高3.03个百分点macro F1、
3.48个百分点balanced accuracy和7.45个百分点QWK。这说明额外token若要分配，
放在时间方向比放在频率方向更符合本数据；只是这种方向性优势尚不足以抵消两倍
token和推理时间，也没有形成IDEA-012预期的净提升。

### AST frequency-fine：作为方向对照提供了明确的负结果

- Kitten recall仍为0.8000，说明AST家族对kitten的敏感性仍在；
- 但senior只识别对20/34只，QWK、macro F1均为全表最低；
- 与time-fine使用相同token数和近似计算时间，因此差异不是简单由模型规模造成。

它的价值主要在实验解释：更密的频率重叠没有带来整体收益，反而削弱senior和
年龄顺序表现。这让“时间方向相对更有用”的判断有了等计算量对照。

### AST nested-selected：最接近VGGish的折中方案

- Balanced accuracy最高：0.7082；
- 总预测正确76/111，只比VGGish少1只；
- Macro F1为0.6666，与VGGish相差1.8个百分点；
- 三类预测数量为23/53/35，比standard AST的25/48/38更接近真实类别数量
  15/62/34，adult recall也由standard的0.6129恢复到0.6774。

它把AST的年龄两端敏感性与较好的adult恢复结合起来，是当前AST路线中最均衡的
实际pipeline。它没有拿到最高QWK，也没有拿到最高macro F1，但在两个目标之间
形成了最好的折中。

## 11. 表格传递的总体信息

| Pipeline | 最突出的优势 | 主要代价/特点 | 最适合说明什么 |
| --- | --- | --- | --- |
| VGGish+MLP | Macro F1、总正确数、kitten/adult F1 | Senior覆盖较弱 | 当前离散三分类baseline最强 |
| AST standard | QWK、senior recall、计算效率 | Adult recall较低 | 年龄序数结构和两端年龄敏感性 |
| AST time-fine | 同算力方向对照中全面胜过frequency-fine | 未超过standard，耗时约翻倍 | 时间信息比额外频率重叠更有价值 |
| AST frequency-fine | 保持较高kitten recall | 总体指标最低 | 排除“增加任意patch密度都会改善”的解释 |
| AST nested-selected | Balanced accuracy、三类折中 | 不是单一表示，依赖inner选择 | 当前最均衡的AST工作流 |

这张表不是简单给出一个“唯一赢家”，而是显示出两类表示的互补性：

- **VGGish长于类别边界和precision**，尤其是kitten与adult；
- **AST长于年龄两端的召回与有序关系**，尤其是standard AST的senior recall和QWK；
- **Nested selection能够回收部分adult能力**，形成balanced accuracy最好的折中；
- **Time-fine相对frequency-fine的全面领先**，说明时间方向确实比频率方向更值得
  后续研究，只是本轮步长修改本身还没有带来最终macro F1提升。

论文中可以把这一结果概括为：**VGGish与AST并非只有高低关系，而是形成不同的
错误画像。VGGish更擅长精确地区分当前三个离散年龄标签；AST更容易识别kitten和
senior，并保留年龄顺序信息。下一阶段最有潜力的方向，是利用两者的互补性改善
adult边界与边缘年龄召回之间的平衡。**

## 12. GPU硬件纠正与重跑结果

2026-08-26重新核验确认本机配有NVIDIA GeForce RTX 4060 Ti 8GB。原先“本机
没有NVIDIA GPU”的判断是硬件审计错误；8月25日实验走CPU，是因为环境安装了
`torch 2.2.2+cpu`，脚本也没有把模型和batch迁移到CUDA。

切换到`torch 2.2.2+cu121`并增加显式device控制后，使用相同checkpoint、输入、
geometry、folds、seed和MLP完成全量重跑。standard/time-fine/frequency-fine
推理时间从43.93/90.79/92.21秒降至2.81/5.19/5.21秒，加速15.63～17.68倍。

GPU与CPU embedding只存在`1e-6`量级的平均浮点差异；所有111只猫的最终预测
标签、animal-level指标、nested选择和主要差值完全一致。因此，GPU重跑确认了
第一轮结果的设备一致性，同时打开了有限fine-tuning、PEFT和token-level pooling
等下一阶段路线。

## 13. 本轮限制

- 本轮是frozen-encoder、每折单seed的pilot，不代表完整AST fine-tuning的能力
  上限。首轮使用CPU是当时CPU-only PyTorch环境和脚本执行路径所致；本机已确认
  配有RTX 4060 Ti 8GB，可用于GPU一致性重跑和后续有限fine-tuning。
- 每个 inner validation 只有17只猫，候选排序容易受单只动物影响；四折结果的
  波动也较明显。
- AST checkpoint 来自 AudioSet，输入位置网格从较长音频设置插值到1.28秒窗口。
- outer test 已经查看，不宜继续围绕当前 outer 结果无边界调 stride。若继续调参，
  应明确登记为新的 exploratory protocol，并在新的确认实验中评价。
- 原 notebook 的约0.70三分类结果使用937行官方 VGGish CSV、作者 split 修改和
  embedding/call reporting unit；本轮0.6846使用936行清洗视图、固定111猫 folds
  和 animal-level macro F1。两者口径不同，不能解释为原复现结果退化。

## 14. 建议回传讨论的问题

1. 论文后续目标应坚持“macro F1 必须超过 VGGish”，还是可以把“更均衡的年龄类
   召回和更合理的序数错误”设为一个独立贡献？
2. 是否优先尝试利用这种互补性，例如概率融合、ordinal classification head，或
   针对 adult/边缘类的损失设计，而不是继续细调 patch stride？
3. 如何利用已确认的RTX 4060 Ti 8GB进行有限的AST encoder fine-tuning，以检验
   frozen representation是否限制了IDEA-012/048？
4. 2026-08-26已经收到IDEA-003、013和019的完整材料；下一轮应结合GPU条件、
   当前代码接口和主性能目标，从中明确冻结一个候选。

## 15. 可复核材料

- 详细技术报告：`reports/02_locked_protocol_ast_results.md`
- 冻结协议：`configs/protocol/meowagenet_locked_v1.json`
- 汇总结果：`metadata/experiments/meowagenet_locked_v1_results.json`
- 数据来源与口径：`metadata/datasets/meowagenet/dataset_source.md`
- 数据 manifest：`metadata/datasets/meowagenet/data_manifest.csv`
- GPU AST审计：`runs/ast_locked_v1/gpu_rerun_2026-08-26/ast_embedding_audit.json`
- GPU重跑比较：`runs/locked_comparison_gpu_verified_2026-08-26/comparison_summary.json`

建议对外使用的简短结论：

> 在固定cat-ID划分和animal-level评价下，IDEA-012的time-fine AST没有带来相对
> 标准AST的净提升，但等计算量对照显示时间方向明显优于频率方向。IDEA-048的
> nested-selected AST与VGGish的macro F1相差1.8个百分点，并取得最高balanced
> accuracy；standard AST则取得最高QWK和senior recall。结果显示VGGish更擅长
> 精确的离散类别边界，AST更擅长年龄两端召回和年龄顺序建模，两类表示具有明确
> 的互补潜力。
