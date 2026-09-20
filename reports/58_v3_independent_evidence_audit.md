# v3 主稿独立证据审计附件

日期：2026-09-18  
状态：证据账本已锁定；等待牛马1主稿逐项审阅  
边界：不覆盖 v2/v2.1；不编辑 v3 主稿；不启动 GPU；不重新访问外层测试。

## 1. 审计结论

v3 最稳妥的证据主线不是“某个 Adapter 或年龄模块已经被确认”，而是：

1. **冻结 AST 表示路线相对 VGGish 获得了正式、方向一致的内部验证优势。** Formal-v2.1 中 adapter pipeline 相对 VGGish 的 Macro-F1 平均差为 `+0.07648`，9/9 套 complete OOF 为正；matched AST head-only 相对 VGGish 也为 `+0.07128`。这支持把 **AST representation/head route** 列为 primary candidate。
2. **Probe-guided Adapter 的独立增量未被确认。** Pilot 中 Adapter−head 为 `+0.0315`；formal 校准为 `+0.00519`、5/9 为正、95% CI `[-0.04581,+0.05850]`；确定性新种子复现转为 `−0.0057`、5/9 为正。因此，AST 整体收益不能归因于当前 Adapter。
3. **C1 有小幅历史正均值，但最终预注册 gate 失败。** IDEA-071–081 中可比较的 72 个 seed×repeat cell 描述性均值为 `+0.007207`，方向为 41 正、6 平、25 负；24 个基础种子为 15 正、9 负。它们全部来自同一 111 只猫，不能当作 72 个独立样本。C1 的正式状态仍由 IDEA-076 决定：exploratory positive，不是 primary candidate。
4. **IDEA-077–081 的 AST 本体改造均为 negative result。** 多层混合、pre-last special-token 注入、空间 patch adapter、末层 LoRA、tail ConvPass 均未通过预设 gate。
5. **IDEA-082 的 G3 谱/能量组是 exploratory lead，而非声源归因。** G3 real−A0 为 `+0.01611` 且 utility gate 通过，但 real−shuffled information gate 因 5/9 正向、worst split `−0.05106` 而失败；没有任何组达到 source-supported。
6. **CatMeows 和犬 benchmark 只能作为 external boundary。** CatMeows 是情境分类而非年龄任务；犬任务是低优先级、跨物种、固定 2,290-unit 子集。二者均不得与 MeowAgeNet 年龄 Macro-F1 数值合并。

机器可读版本：`metadata/experiments/meowagenet_v3_independent_evidence_ledger.json`。

## 2. v3 术语状态

| 术语 | 使用条件 | 本项目可放入此类的证据 | 禁止写法 |
|---|---|---|---|
| `primary candidate` | 锁定主要比较支持，且方法特异性结论未被后续确认推翻 | 冻结 AST 表示 + matched head 路线，相对 VGGish | “Probe-guided Adapter 已被独立确认” |
| `exploratory positive` | 均值、局部 cell、utility gate 或跨任务信号为正，但稳定性、校准、机制或独立性 gate 未全部通过 | C1 历史信号；IDEA-082 G3；部分 ConvPass/LoRA 正均值；tuned head seed-17 HPO | “已证实”“稳定复现”“外部确认” |
| `negative result` | 预注册候选或机制的必需 gate 失败 | Adapter-specific increment；IDEA-077–081；IDEA-075 C1−A0；IDEA-082 source attribution | 只挑正 seed/正类别来改判 |
| `external boundary` | 另一数据集或任务用于限制泛化范围，必须注明任务/物种/样本边界 | CatMeows context transfer；犬五阶段年龄子集 | 把 CatMeows 称为猫龄外部验证，或把犬结果数值合并进猫龄主效应 |

论文中可以同时出现“AST route 为 primary candidate”和“当前 Adapter 为 negative result”：前者描述表征/管线，后者描述 Adapter 相对 matched head 的附加机制。

## 3. Probe-guided Adapter 证据链

### 3.1 Pilot

在同一 111 cats 的 exploratory pilot 中：

| Pipeline | Macro-F1 |
|---|---:|
| VGGish + MLP | 0.6846 |
| Frozen AST head-only | 0.7260 |
| Probe-guided AST Adapter | 0.7575 |

Adapter−VGGish 为约 `+0.0729`，Adapter−head 为 `+0.0315`。候选 placement 在查看 pilot 后选出，因此该结果只能用于候选形成和探索性论证。

### 3.2 Formal-v2.1

Formal-v2.1 完成 108/108 fits，形成每 pipeline 9 套 complete OOF；每套都是同一 111 cats 的完整预测。

| Pipeline | Macro-F1 mean ± SD | Balanced accuracy | QWK |
|---|---:|---:|---:|
| VGGish + MLP | 0.6525 ± 0.0462 | 0.6525 | 0.5334 |
| AST head-only | 0.7238 ± 0.0335 | **0.7597** | **0.6374** |
| Probe-guided Adapter | **0.7290 ± 0.0428** | 0.7419 | 0.6373 |

- Adapter−VGGish：`+0.07648`，9/9 正；层级 paired bootstrap 95% CI 为 `−0.00635` 到 `+0.16851`。
- Adapter−head：`+0.00519`，5/9 正；95% CI 为 `−0.04581` 到 `+0.05850`。

因此 H048 支持 AST pipeline 相对 VGGish，H019 不支持把增益特异地归因于 Adapter。

### 3.3 HPO 与确定性复现

HPO 属于 formal 之后的 exploratory selection。仅 seed 17 的三 repeats 中，tuned head 为 `0.7488`，Adapter 为 `0.7367`，Adapter−head 为 `−0.0121`。它支持把 tuned head 作为后续性能候选，但不能覆盖 formal。

同种子复跑诊断发现 head-only 的 12/12 预测逐位一致，而 Adapter 的 12/12 均不同；Adapter 均值从历史 `0.7348` 变为 `0.7246`。启用 deterministic algorithms、cuBLAS workspace 和 math attention 后，同 cell 连续拟合完全一致。因此这次同种子差异是**计算确定性诊断**，不能计作一轮新的效果估计。

确定性新基础种子 151/307/509 的 9 套 complete OOF 中：

| Pipeline | Macro-F1 mean |
|---|---:|
| AST head-only | 0.7253 |
| Probe-guided Adapter | 0.7195 |
| Adapter−head | −0.0057 |

方向为 5 正、4 负。结论是当前两层 bottleneck Adapter 有 split-dependent positives，但没有稳定独立增益。

## 4. A0/C1 全部可比轮次

以下轮次共享同一 792 calls、111 cats 和 nested role bank；主要单元是每轮锁定的 base_seed×repeat cell，不是新动物。

| Round | Cells | A0 F1 | C1 F1 | C1−A0 | 正/平/负 | 正/负 base seeds | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| IDEA-071 | 9 | 0.73118 | 0.72996 | −0.00121 | 5/0/4 | 2/1 | negative/no primary gate |
| IDEA-072 | 9 | 0.71760 | 0.72671 | +0.00911 | 7/0/2 | 2/1 | exploratory positive，gate fail |
| IDEA-073 | 9 | 0.74575 | 0.77059 | +0.02484 | 7/2/0 | 3/0 | exploratory positive |
| IDEA-074 | 9 | 0.73735 | 0.72484 | −0.01251 | 2/1/6 | 0/3 | negative，gate fail |
| IDEA-076 | 18 | 0.73894 | 0.74793 | +0.00899 | 9/3/6 | 4/2 | exploratory positive，final gate fail |
| IDEA-080 control | 9 | 0.73570 | 0.74659 | +0.01089 | 5/0/4 | 2/1 | descriptive only |
| IDEA-081 control | 9 | 0.73018 | 0.73873 | +0.00855 | 6/0/3 | 2/1 | descriptive only |

两种跨轮描述只可作为敏感性概览：

- IDEA-071–076：54 cells，平均 `+0.006368`，30 正、6 平、18 负；18 个基础种子为 11 正、7 负。
- 加入 IDEA-080/081 factorial control：72 cells，平均 `+0.007207`，41 正、6 平、25 负；24 个基础种子为 15 正、9 负。

IDEA-076 是预先声明的 final confirmation。其平均差达到 `+0.00899`，但 9/18 seed×repeat 为正，worst split 为 `−0.04913`，CE 略差，且 seed 8807 senior-recall 差为 `−0.03333`；最终主 gate 失败。后续 080/081 的 C1 是其他 factorial 实验中的 control，不能事后替代 IDEA-076 的决策。

## 5. 可合法汇总与不可合并项

### 可合法汇总

- Formal-v2.1 内，可报告 9 个 paired complete-OOF estimates，并使用已锁定的 hierarchical paired bootstrap；同时说明动物仍只有 111 只。
- 每个 C1 round 内，可按协议对 base_seed×repeat 单元等权汇总 C1−A0。
- C1 跨轮可以给出明确标为 **descriptive sensitivity inventory** 的均值、符号计数和 base-seed 方向；不得做独立研究 meta-analysis 推断。
- CatMeows 和犬结果可以分别在 task-specific 表中报告，各自保留动物数、类别、dummy 和任务含义。

### 不可合并

- 不得把 folds、seed×repeat cells 或 612/1,224 个重复 animal occurrences 当作独立动物扩大 n。
- 不得把 pilot、formal 和 deterministic replication 写成三个独立队列；它们重复使用同一 111 cats。
- 不得把 formal complete outer-test OOF 与 IDEA-071–082 的 inner-validation screens 当成同一证据层级。
- 不得把 CatMeows 三分类情境、犬五分类年龄和 MeowAgeNet 三分类年龄的 Macro-F1 合成一个效应。
- 不得用 CatMeows 或 C1−dummy 的结果“挽救”MeowAgeNet 上失败的 Adapter/C1 paired gate。
- 不得根据有利 seed、senior recall、CE/Brier 或 post-outcome feature group 改写预注册失败结论。

## 6. IDEA-077–082 本体与声学消融

| IDEA | 候选 | A0 F1 | 候选 F1 | 主差值 | 结论 |
|---|---|---:|---:|---:|---|
| 077 | Last-four global LayerMix | 0.74157 | 0.73135 | −0.01022；3/9 正 | Negative；所有 gate 条件失败 |
| 078 | Pre-last special-token age injection | 0.74087 | 0.72076 | −0.02011；5/9 正 | Negative；还比等容量 post-pool control 低 −0.01035 |
| 079 | Spatial patch adapter | 0.74576 | 0.73898 | −0.00678 | Negative；比 pointwise control +0.00320 但不稳定 |
| 080 | Last-block Q/V LoRA | 0.73570 | 0.73827 | +0.00257；5/9 正 | Negative；概率质量变差，LoRA+C1−C1 为 −0.01156 |
| 081 | Tail ConvPass | 0.73018 | 0.73652 | +0.00635；5/9 正 | Negative；worst split −0.03361，gate fail；interaction −0.00552 |
| 082 | G3 spectral/energy real | 0.73991 | 0.75602 | utility +0.01611 | Exploratory lead；real−shuffled +0.01966 但 information gate fail |

IDEA-082 四个组均未达到 `source_supported`。G3 的 utility gate 通过值得保留，但 information contrast 只有 5/9 为正，worst split `−0.05106`；因此只能写“G3 是下一步外部验证线索”，不能写“谱/能量源已被确认”。

## 7. CatMeows 与犬 benchmark 的任务边界

### CatMeows

- 440 recordings、21 cats；split unit 为 cat。
- 三分类目标：梳毛、等待食物、隔离。
- Frozen AST Macro-F1 `0.5214 ± 0.0085`；prior dummy `0.2229`；差值 `+0.2985`，3/3 repeats 为正。
- 合法结论：AST 能把情境/行为相关信息迁移到未见猫。
- 非法结论：猫龄模型的外部验证。CatMeows 没有年龄标签。

### 犬五阶段年龄子集

- 2,290 bark units、125 dogs；split unit 为 dog；标签为 puppy/juvenile/adolescent/adult/senior。
- 这是从 79,142 units 中预先固定的资源受限子集。
- IDEA-067 Frozen AST：Macro-F1 `0.2127`，dummy `0.0950`，差值 `+0.1177`，但 balanced accuracy 约 `0.214`，绝对性能弱。
- IDEA-075 C1−A0：`+0.0002234`，4/9 正，CE/Brier 略差；主 gate FAIL。C1−dummy 为 `+0.132964` 且 9/9 为正，仅说明明显高于类别先验。
- 决策：不升级 C1，不进入完整 79,142-unit benchmark。

优先级必须保持为：MeowAgeNet 猫龄主证据 > CatMeows 情境保持边界 > 犬跨物种年龄边界。

## 8. 主稿逐项审阅清单

主稿出现后，必须逐项检查：

- [ ] 摘要是否把 AST route 和 Adapter-specific effect 分开；
- [ ] Formal-v2.1 是否明确为 pilot-informed internal repeated validation；
- [ ] 是否同时报告 Adapter−VGGish `+0.0765` 与 Adapter−head `+0.0052`／CI 跨 0；
- [ ] 同种子 non-deterministic rerun 是否只作诊断，而非新效果估计；
- [ ] deterministic new-seed Adapter−head `−0.0057` 是否保留；
- [ ] C1 是否标为 exploratory positive，且 IDEA-076 final gate fail 未被 080/081 control 反向改判；
- [ ] 72 C1 cells 是否避免“72 独立样本”措辞；
- [ ] IDEA-077–081 是否全部按 negative result 报告；
- [ ] IDEA-082 是否写明 G3 utility pass / information fail / no source supported；
- [ ] CatMeows 是否只称 context transfer；
- [ ] 犬结果是否保留 2,290-unit subset、125 dogs、五阶段、低优先级边界；
- [ ] 是否没有跨任务数值 pooling、挑 seed、挑 secondary metric 或事后改 gate。

任何一项违反即给修订清单；全部满足才可 PASS。
