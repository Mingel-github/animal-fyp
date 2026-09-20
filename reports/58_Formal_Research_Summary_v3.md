# MeowAgeNet Formal / Research Summary v3（中文主稿）

> 文档性质：截至 IDEA-082 的综合研究主稿；保留 v2 / v2.1 历史文件，不回写旧结论。  
> 当前决策：**A0 是唯一的 primary / 当前主模型与稳健参照；C1 是 exploratory-positive（探索性正向）的主要研究候选，但绝非 primary，也尚未被确认可替代 A0。**  
> 首要边界：后期各轮虽使用新初始化种子，仍反复使用同一批 792 条叫声、111 只猫和同一套角色系统；不得把 seed、repeat、fold 或重复出现的猫当成新增独立样本。[数据与独立性来源](42_IDEA-072_C1_independent_seed_confirmation_results.md#新种子聚合单位与预算)

## 1. 执行摘要

当前最可靠的结论不是“某个复杂适配器已经胜出”，而是以下四层证据：

1. **数据集主任务证据。** 冻结 AudioSet-AST 表示配合轻量分类头（A0）持续构成最稳健的 MeowAgeNet 年龄分类参照。训练轮次应按最低动物级验证交叉熵选择；该规则曾以平均 Macro-F1 `+0.0059`、`4/6` 配对为正通过预设流程门，并被后续 tuned frozen-AST reference 采用。[IDEA-056](29_IDEA-056_checkpoint_selection_confirmation_results.md#2-六组新的-complete-oof-配对)
2. **探索性正向主要研究候选。** C1 是一个读取 20 维无标签年龄敏感声学特征、相对 AST 隐状态 RMS 施加 `0.25` 上限的有界加法残差。它在多轮新种子中反复出现正平均方向，但 IDEA-076 的最终预注册门因稳定性、最差 split、CE 和 senior-recall 安全条件失败，故只能列为 exploratory-positive 的主要研究候选，绝非 primary，也不能称为已确认改进。[结构与最终门来源](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#预注册-gate-判定)
3. **机制证据。** IDEA-082 不支持把 C1 的方向性收益简单归因于纯 F0。G3（spectral-energy / voice-quality）是最强线索：相对 A0 为 `+0.016106` 且通过实用门，但相对 matched shuffled 虽为 `+0.019658`，只有 `5/9` seed×repeat 为正、最差 split 为 `−0.051064`，信息门失败，因此 `source_supported=false`。[IDEA-082](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#3-预注册机制门)
4. **历史 PEFT 负结果／混合证据。** Probe-guided adapter 的轨迹从 pilot 相对 head-only `+0.0315`，变为 formal-v2.1 的 `+0.005195`、`5/9` 为正；随后一次**未强制确定性的同 seed rerun**给出约 `−0.0091`，它是计算非确定性诊断、不是新的效果估计。真正采用确定性设置的新种子复验为 `−0.0057`、`5/9` 为正。因此 adapter 的独有增量应记为 negative result / historical mixed evidence；冻结 AST 的价值不应归因于该 adapter。[pilot](08_IDEA-048_stage_checkpoint.md)；[formal-v2.1](12_formal_v2_1_core_results.md#4-两项锁定-primary-contrasts)；[计算诊断与确定性新种子复验](35_AST_adapter_deterministic_replication_results.md)

## 2. 证据边界与术语

### 2.1 四类证据必须分开

| 层级 | 回答的问题 | 可作何种结论 | 不可作何种结论 | 主要来源 |
|---|---|---|---|---|
| 历史 outer-test / complete-OOF | 在既定 111 猫折上，早期方案相对表现如何 | 支持 AST 路线优于旧 VGGish、记录方法演变 | 不能作为从未见过该数据的独立确认；多轮复用不能累加样本量 | [formal-v2 冻结说明](09_formal_protocol_v2_freeze.md#2-pilot-与正式-v2-的证据边界)、[formal-v2.1 结果](12_formal_v2_1_core_results.md) |
| 后期主任务 inner-validation | A0、C1 及新模块在锁定 train/validation 角色上是否稳定 | 比较同轮、配对、新种子下的方向、尾部风险和概率质量 | 不能把新种子称为新动物或外部复现 | [IDEA-072 统计口径](42_IDEA-072_C1_independent_seed_confirmation_results.md#新种子聚合单位与预算)、[IDEA-076 边界](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#跨轮描述性更新) |
| 机制证据 | 年龄声学分支的收益来自哪些输入组或约束 | 允许报告 real−A0、real−shuffled 和结构对照 | 双门未过时不得作来源归因或因果声称 | [IDEA-082](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#5-结论与下一步边界) |
| 外部 benchmark / external boundary | 冻结 AST 或 C1 能否迁移到其他数据/物种 | CatMeows 支持跨猫情境表征；犬数据提供弱年龄信号和跨物种边界 | CatMeows 不证明猫年龄；犬子集不证明实用跨物种年龄模型 | [IDEA-067](37_IDEA-067_external_AST_representation_results.md#结果边界与可复用价值)、[IDEA-075](45_IDEA-075_dog_C1_age_sensitive_AST_benchmark_results.md#研究边界) |

### 2.2 探索性与确认性

- “新种子”只减少初始化偶然性，不增加独立动物；同一批 111 猫上的 `seed×repeat` 是配对稳定性单元，不是独立样本。[IDEA-069](39_IDEA-069_A1_independent_seed_replication_results.md#统计口径与独立性)
- 只有事前锁定且所有 gate 同时满足时，才能使用“通过”或“确认”措辞；平均值为正但任一稳定性、安全性或概率质量条件失败时，只能报告“正向信号”或“探索性候选”。[IDEA-076](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#预注册-gate-判定)
- 本文不对跨轮同一 111 猫的结果进行伪显著性检验，不把 54-cell 或截至 IDEA-081 的 72-cell 汇总解释为独立动物样本；两者都只是描述性台账。[IDEA-076 的 54-cell 边界](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#跨轮描述性更新)、[v3 的 72-cell 派生记录](../metadata/experiments/meowagenet_formal_research_summary_v3.json)

## 3. 当前代号表与模型公式

### 3.1 代号表

| 代号 | 含义 | 当前身份 |
|---|---|---|
| A0 | 冻结 AST 最终表示 + `768→128→3` 分类头；按动物级验证 CE 选择 checkpoint | **当前主模型、主要参照**；可作为默认报告与部署候选 [来源](29_IDEA-056_checkpoint_selection_confirmation_results.md#6-阶段决策与下一步) |
| A1 | `20→32→128` 无界年龄声学加法残差 | 早期强正向线索，但跨 split 与 CE 不稳定 [来源](39_IDEA-069_A1_independent_seed_replication_results.md#科学解释与后续价值) |
| B1 | `20→32→128`、通道缩放范围 `[0.75,1.25]` 的有界调制 | 校准线索，分类收益不足以取代 A1/A0 [来源](40_IDEA-070_bounded_age_conditioned_AST_modulation_results.md#结论) |
| C1 | `20→60→128`、RMS-relative、`tanh` 有界加法残差 | **exploratory-positive 主要研究候选**；绝非 primary / confirmed，最终确认门失败 [来源](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#最终研究状态) |
| U1 | 与 C1 等宽、等参数的无界加法残差 | 容量对照；说明 C1 结果不能简单全归因于 bound [来源](42_IDEA-072_C1_independent_seed_confirmation_results.md#有界机制false) |
| D1 | 有界 scale + shift 双路径 | 未优于 C1；额外 scale 路径不保留 [来源](41_IDEA-071_bounded_dual_path_age_AST_fusion_results.md#结论) |
| G1/G2/G3/G12 | F0、稳定性/周期性、谱能量/声质、F0+稳定性分组 | 机制消融；四组均未获来源支持，G3 是最强探索性线索 [来源](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#3-预注册机制门) |

> 命名注意：本文的 C1 始终指 IDEA-071 起的“有界宽年龄加法残差”。IDEA-056 报告中也曾用 C1 指“动物级 CE checkpoint 选择规则”，两者不是同一对象。[IDEA-056 命名来源](29_IDEA-056_checkpoint_selection_confirmation_results.md#结论摘要)

### 3.2 A0 与 C1

令冻结 AST call embedding 为 \(x\in\mathbb{R}^{768}\)，训练角色内标准化后的 20 维无标签声学特征为 \(a\)，则共同主路径为：

\[
h=\operatorname{ReLU}(W_{ast}x+b_{ast}),\qquad h\in\mathbb{R}^{128}.
\]

A0 不增加年龄分支：

\[
h_{A0}=h.
\]

C1 定义为：

\[
c=\operatorname{GELU}(W_1a+b_1),\quad W_1:20\rightarrow60,
\]

\[
q=\operatorname{stopgrad}\left(\sqrt{\frac{1}{128}\sum_{j=1}^{128}h_j^2+10^{-8}}\right),
\]

\[
r=0.25q\tanh(W_2c+b_2),\qquad h_{C1}=h+r.
\]

公式中的 `stopgrad` 防止主路径通过放大自身间接放宽预算；每维写入被限制在当前样本隐藏 RMS 的 `25%` 尺度。A0 与 C1 的可训练参数量分别为 `99,075` 与 `108,143`，年龄输出层从零影响起点初始化。[公式、参数与设计来源](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#预注册设计)

训练继续使用全局类别平衡的 call-level CE，checkpoint 固定为最低未加权 inner-validation animal CE；推理时先取得 call 概率，再对同猫 calls 作算术平均。[训练口径来源](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#预注册设计)

20 维特征覆盖 log-F0 分位数/离散度/时间变化、有声比例、voicing probability、周期性/HNR、RMS 变化、谱倾斜与谱平坦度；提取不读取年龄标签，缺失值与标准化统计只来自训练角色。[特征来源](38_IDEA-068_age_sensitive_acoustic_AST_results.md#2-无标签声学特征审计)

## 4. 主任务证据：年龄声学线如何演变

### 4.1 C1 之前：IDEA-068 至 IDEA-070

| 轮次 | 关键比较 | 结果 | 阶段含义 | 来源 |
|---|---|---|---|---|
| IDEA-068 | A1−A0 | 平均 fold Macro-F1 `+0.0418`；`6` 正、`4` 平、`2` 负；最差 fold `−0.0411`，CE 略差 `+0.000329` | 显式年龄声学特征有强方向性价值，但未过完整 gate | [报告](38_IDEA-068_age_sensitive_acoustic_AST_results.md#3-主要结果) |
| IDEA-069 | A1−A0 | seed×repeat 均值 `+0.010639`；`6/9` 为正、`3/3` base seed 为正；仅 `7/12` split 非负，最差 `−0.038621`，CE 变差 | 正向方向跨初始化延续，但不是跨数据确认 | [报告](39_IDEA-069_A1_independent_seed_replication_results.md#主要结果) |
| IDEA-070 | B1−A0 | `+0.005090`；`6/9` 正、`9/12` split 非负；最差 `−0.037598`；CE/Brier 优于 A0，但 B1−A1 为 `−0.012483` | 有界缩放改善概率质量，却损失加法残差的分类能力 | [报告](40_IDEA-070_bounded_age_conditioned_AST_modulation_results.md#主要结果) |

这三轮共同促成 C1：保留 A1 “可写入新方向”的加法能力，同时用 RMS-relative `tanh` 预算约束幅度，并扩大年龄编码宽度以匹配后续机制对照。[IDEA-071 设计](41_IDEA-071_bounded_dual_path_age_AST_fusion_results.md#c1有界宽加法对照)

### 4.2 C1 按轮结果

下表中的每个值都来自同轮 A0/C1 严格配对；不同轮使用新 base seeds，但仍是同一批 111 猫。不能把各行简单视为相互独立的重复试验。[跨轮边界](../metadata/experiments/meowagenet_C1_evidence_ledger_pre_IDEA076.json)

| 轮次 | C1−A0 Macro-F1 | seed×repeat 正/平/负 | 预注册身份/结论 | 来源 |
|---|---:|---:|---|---|
| IDEA-071 | `−0.001214` | `5/0/4` | 首次 C1；近似持平 A0，明显优于当轮 A1，但未晋级 | [台账](../metadata/experiments/meowagenet_C1_evidence_ledger_pre_IDEA076.json)、[报告](41_IDEA-071_bounded_dual_path_age_AST_fusion_results.md#主要结果) |
| IDEA-072 | `+0.009112` | `7/0/2` | 正平均且 Brier 最优；单一种子、split 与 CE 条件失败 | [报告](42_IDEA-072_C1_independent_seed_confirmation_results.md#seedrepeat-主要结果) |
| IDEA-073 | `+0.024839` | `7/2/0` | 当轮最强分类候选；但该轮主 gate 针对 S1，C1 结果仍属综合性支持 | [台账](../metadata/experiments/meowagenet_C1_evidence_ledger_pre_IDEA076.json)、[报告](43_IDEA-073_soft_radial_budget_residual_results.md#跨轮综合更新c1-升为当前主要候选) |
| IDEA-074 | `−0.012506` | `2/1/6` | 全新种子反向，C1 新种子复现 gate 失败，身份下调为探索性候选 | [报告](44_IDEA-074_C1_hybrid_call_cat_factorial_results.md#结论) |
| IDEA-076 | `+0.008989` | `9/3/6`（18 单元） | 最终预注册确认：均值门通过，但稳定性、最差 split、CE、senior 安全失败；主 gate=false | [报告](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#主要结果18-个-seedrepeat-等权均值) |
| IDEA-080 | `+0.010894` | `5/0/4` | LoRA 因子实验中的描述性新种子支持；不是新的 C1 confirmation | [metadata](../metadata/experiments/meowagenet_idea080_ast_last_block_lora_c1_factorial_v1_results.json)、[报告](53_IDEA-080_AST_last_block_LoRA_C1_factorial_results.md#主要结果) |
| IDEA-081 | `+0.008553` | `6/0/3` | ConvPass 因子实验中的描述性新种子支持；由九个配对单元重算，不替代 IDEA-076 gate | [锁定 summary](../runs/meowagenet_idea081_ast_tail_convpass_c1_factorial_v1/initial_evaluation_summary.json)、[pipeline 均值](53_IDEA-081_AST_tail_ConvPass_C1_factorial_results.md#3-四管线平均结果) |

IDEA-071 至 IDEA-074 的预先冻结台账合计 36 个 seed×repeat 单元，描述性均值为 `+0.005058`，方向为 `21/3/12`；加入 IDEA-076 后形成冻结的 54-cell 汇总，描述性均值为 `+0.006368`，方向为 `30/6/18`。再纳入 IDEA-080/081 的各 9 个新种子单元，截至 IDEA-081 的 72-cell **纯描述性**汇总为 `+0.007207031785`，方向为 `41/6/25`。这说明“中心方向长期略正且异质性明显”，但所有单元仍复用同一 111 猫：36、54 或 72 都不是独立样本；72-cell 也不重开、不覆盖 IDEA-076 的正式失败判定。[预 IDEA-076 台账](../metadata/experiments/meowagenet_C1_evidence_ledger_pre_IDEA076.json)；[IDEA-076 的冻结 54-cell 更新](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#跨轮描述性更新)；[截至 IDEA-081 的 72-cell 派生记录](../metadata/experiments/meowagenet_formal_research_summary_v3.json)

### 4.3 当前主任务判读

- **A0 保持当前主模型。** 它结构最简单、在多轮新模块比较中稳定作为强参照，且 checkpoint 选择规则已通过单独流程确认。[IDEA-056](29_IDEA-056_checkpoint_selection_confirmation_results.md#结论摘要)
- **C1 保持 exploratory-positive 主要研究候选，但绝非 primary / confirmed。** 支持它的是跨多组新初始化反复出现的正平均方向、IDEA-076 的 `+0.008989` 与 IDEA-080/081 的再次正向；限制它的是最终 gate 失败、显著 split 尾部风险、CE 非一致改善以及个别 senior-recall 安全失败。[IDEA-076](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#正向结果及其限制)
- **模型选择规则。** 若需要一个当前可报告、可复现、无需额外机制假设的主结果，选 A0；若研究目标是验证“显式年龄声学信息能否补足 AST”，则把冻结公式的 C1 与 A0 成对报告，但不得只报 C1 的有利轮次。

## 5. 机制证据：不能把 C1 简化为“纯 F0 模型”

IDEA-082 将既有 20 维特征拆成 real 与 role-local shuffled 的匹配对照。每组必须同时通过 `real−A0` 实用门与 `real−shuffled` 信息门，才能把增益归因于该组真实信息。[设计来源](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#3-预注册机制门)

| 特征组 | real−A0 | real−shuffled | 关键门结果 | 机制结论 | 来源 |
|---|---:|---:|---|---|---|
| G1 F0 level/contour | `+0.007925` | `+0.007799` | 两门均失败 | 不能称为纯 F0 来源 | [IDEA-082](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#g1f0-levelcontour) |
| G2 stability/periodicity/voicing/HNR | `+0.006044` | `+0.000179` | 两门均失败 | 真实信息相对 shuffled 几乎无净增益 | [IDEA-082](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#g2source-stabilityperiodicityvoicinghnr) |
| G3 spectral-energy/voice-quality | `+0.016106` | `+0.019658` | 实用门 PASS；信息门因 `5/9` 为正、最差 split `−0.051064` 而 FAIL | 最强探索性线索，但 `source_supported=false` | [IDEA-082](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#g3spectral-energyvoice-quality) |
| G12 F0+stability | `+0.001947` | `−0.007927` | 两门均失败 | 组合未恢复完整 C1 的方向与稳定性 | [IDEA-082](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#g12f0-stability) |

因此，论文中的机制表述必须是：**“C1 使用 F0、周期性与声质代理特征；分组消融未确认任何单一组为完整收益来源。G3 提供最强探索性线索，但未通过 matched-shuffled 稳定性门。”** 不能写成“C1 的收益由 F0 驱动”，也不能把训练后猫级 F0 与年龄的相关性解释成因果机制；猫级 F0 中位数与年龄等级的 `Spearman ρ=−0.693` 只是一项结果后诊断。[相关性边界](38_IDEA-068_age_sensitive_acoustic_AST_results.md#5-结果后的机制诊断)

## 6. PEFT、内部表示与注入路线

### 6.1 Probe-guided adapter：完整演变

Formal-v2.1 的两项 primary contrast 必须分开：H048（adapter−VGGish）为 `+0.076476`、`9/9` 为正，hierarchical paired bootstrap 95% CI 为 `[−0.006354,+0.168513]`；它主要支持“AST 路线相对旧 VGGish 的内部优势”，但区间仍跨 0。H019（adapter−matched AST head-only）仅为 `+0.005195`、`5/9` 为正，95% CI 为 `[−0.045810,+0.058502]`，区间同样跨 0；结合后续确定性新种子负均值，adapter 的**独有增量**应记为负结果／历史混合证据，而不是 AST 路线成功的原因。[formal-v2.1 primary contrasts](12_formal_v2_1_core_results.md#4-两项锁定-primary-contrasts)

| 阶段 | Adapter−head-only | 方向信息 | 当前解释 | 来源 |
|---|---:|---|---|---|
| IDEA-019 pilot | `+0.0315` | 单套 111-cat outer-test OOF | 达到当时实际意义阈值，但 probe-guided 仅比 random adapter 高约 `0.0030`，不能证明 probe 指导独有贡献 | [pilot](07_IDEA-019_PEFT_placement_results.md#1-结论摘要)、[阶段总结](08_IDEA-048_stage_checkpoint.md) |
| Formal-v2.1 H019 | `+0.005195` | `5/9` complete OOF 为正；95% CI `[−0.045810,+0.058502]` 跨 0 | adapter 独有增量很小且 split-dependent | [formal-v2.1](12_formal_v2_1_core_results.md#h019adapter-独立贡献假设) |
| 同 seed、未强制确定性的 rerun／计算诊断 | 约 `−0.0091` | 原差值约 `+0.0011` 反向；head 逐位复现，adapter 原实现存在 CUDA/attention 非确定性 | 仅用于暴露计算非确定性，不是新的效果估计 | [计算诊断](35_AST_adapter_deterministic_replication_results.md#同种子复跑与确定性诊断) |
| 真正 deterministic 的新种子复验 | `−0.0057` | `5/9` 为正、`4/9` 为负，范围 `−0.0823` 至 `+0.0320` | 当前 adapter 独有增量为负结果／历史混合证据 | [新种子复验](35_AST_adapter_deterministic_replication_results.md#新种子确定性复现) |

### 6.2 其他预注册路线

| 路线 | 关键配对结果 | 结论 | 来源 |
|---|---|---|---|
| LoRA（IDEA-050，候选选择式） | LoRA−head `−0.0313`；三次 repeat 均值为 `−0.0043/−0.0465/−0.0433` | 参数高效但总体更差，不保留当轮候选 | [IDEA-050](20_IDEA-050_AST_LoRA_initial_results.md#4-animal-level-complete-oof-主结果) |
| LoRA（IDEA-080，固定 block-12 Q/V rank-6） | L1−A0 `+0.00257`；CL1−C1 `−0.01156`；交互 `−0.01413` | 主效应、组合与交互 gate 全失败 | [IDEA-080](53_IDEA-080_AST_last_block_LoRA_C1_factorial_results.md#主要结果) |
| Last-4 LayerMix | L1−A0 `−0.01022`，仅 `3/9` 为正；八项 gate 全失败 | 全局、样本无关的层混合不保留 | [IDEA-077](48_IDEA-077_AST_last4_global_layer_mix_results.md#主-gate-审计) |
| Pre-last special-token 年龄注入 | T1−A0 `−0.020109`；T1−等参数 P1 `−0.010354` | 冻结 block 12 前修改两个 special tokens 未形成稳健收益 | [IDEA-078](50_IDEA-078_AST_prelast_special_token_age_injection_results.md#主要结果九个-seedrepeat-等权均值) |
| Spatial patch adapter | S1−A0 `−0.00678`；S1−pointwise `+0.00320` 但仅 `4/9` 为正 | 空间候选门与机制门均失败 | [IDEA-079](49_IDEA-079_AST_spatial_patch_adapter_results.md#主要结果) |
| Tail ConvPass | V1−A0 `+0.00635`，但仅 `5/9` 为正且 senior 安全失败；CV1−C1 `+0.00083`；交互 `−0.00552` | 存在弱均值信号，但固定尾层 ConvPass 与 C1 组合均不保留 | [IDEA-081](53_IDEA-081_AST_tail_ConvPass_C1_factorial_results.md#4-预注册对比) |
| Soft radial residual | S1−A0 `+0.005644`、`6/9` 为正，但仅 `1/3` base seed 为正、最差 split `−0.07129`；S1−U1 `−0.004098` | U1 分类能力保留层通过，但整体 gate 与 C1 层失败；不晋级 | [IDEA-073](43_IDEA-073_soft_radial_budget_residual_results.md#预设-gate-判定) |
| Hybrid Call+Cat × C1 | C1-H−C1-C `−0.001106`；交互 `+0.006182`，但仅 `7/12` split 非负、最差 `−0.068152` | 局部 Brier/交互方向保留，三个预设层级均失败；不并入 C1 | [IDEA-074](44_IDEA-074_C1_hybrid_call_cat_factorial_results.md#结论) |

这些结果共同支持一个克制判断：在当前小样本、动物分组任务中，移动 AST 内部表示或增加 PEFT 参数通常没有稳定优于冻结 AST；C1 的相对优势在于它引入了有明确领域含义且受幅度约束的外部声学线索，而不是单纯增加 backbone 可训练容量。

## 7. 外部 benchmark

### 7.1 CatMeows：优先于犬数据的其他猫数据

CatMeows 使用 440 条录音、21 只猫，任务是梳毛、等待食物、隔离三类情境识别；三次 animal-grouped 评估中，冻结 AST 的 Macro-F1 为 `0.5214±0.0085`，先验 dummy 为 `0.2229`，平均差 `+0.2985`，三次均为正。[CatMeows 结果](37_IDEA-067_external_AST_representation_results.md#catmeows未见猫的情境分类)

这支持“AST 表示包含可跨未见猫迁移的情境/行为信息”，但任务标签不是年龄，因此不能用来证明 AST 或 C1 已学到猫年龄机制。[边界](37_IDEA-067_external_AST_representation_results.md#结果边界与可复用价值)

### 7.2 犬年龄 benchmark：最低优先级的跨物种探索

IDEA-067 在固定资源子集的 2,290 个 bark units、125 只狗、五年龄阶段上，冻结 AST 相对先验 dummy 的 Macro-F1 平均高 `+0.1177`，但 AST 平衡准确率仅约 `0.214`，senior recall 为 `0.104–0.172`，只能称为弱、方向一致的年龄信号。[IDEA-067 犬结果](37_IDEA-067_external_AST_representation_results.md#canine-age-transition犬只年龄阶段分类)

IDEA-075 进一步比较 A0/U1/C1：C1−A0 仅 `+0.000223`，`4/9` seed×repeat 为正，CE 与 Brier 均略差；C1 却在 `9/9` 单元高于 outer-train prior dummy，平均优势 `+0.132964`。因此外部任务下限通过，但 C1 整包相对 A0 与有界机制相对 U1 均失败。[IDEA-075](45_IDEA-075_dog_C1_age_sensitive_AST_benchmark_results.md#结论)

跨物种结果不能反向改变 MeowAgeNet 的模型排序，也不能用犬数据“挽救” IDEA-076 或 IDEA-082 的 gate。证据优先级固定为：**MeowAgeNet 主任务 > 其他猫数据 > 犬数据**。[优先级来源](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#优先级与证据边界)

## 8. 历史 outer-test 证据

历史 outer-test / complete-OOF 结果记录了模型开发轨迹，但全部围绕同一 111 猫展开；formal-v2 明确将其定义为 pilot-informed prospective evaluation，而非从未接触数据的独立验证。[协议边界](09_formal_protocol_v2_freeze.md#2-pilot-与正式-v2-的证据边界)

| 阶段 | 关键结果 | v3 中的用途 | 来源 |
|---|---|---|---|
| IDEA-019 pilot | head-only `0.7260`，probe-guided adapter `0.7575`，差 `+0.0315` | 说明早期 adapter 候选如何产生；不作为最终机制结论 | [IDEA-019](07_IDEA-019_PEFT_placement_results.md#1-结论摘要) |
| Formal-v2.1 | VGGish `0.6525`，AST head-only `0.7238`，adapter `0.7290`；H048 `+0.076476`、`9/9` 为正、95% CI `[−0.006354,+0.168513]`；H019 `+0.005195`、`5/9` 为正、95% CI `[−0.045810,+0.058502]` | H048 支持 AST 路线相对旧 VGGish 的内部优势；H019 与后续复验不支持 adapter 独有稳定增量，两区间均跨 0 | [formal-v2.1](12_formal_v2_1_core_results.md#4-两项锁定-primary-contrasts) |
| IDEA-056 checkpoint 规则 | animal-CE 选择相对 call-CE 选择 Macro-F1 `+0.0059`，`4/6` 配对为正 | 固化 A0 后续 checkpoint 规则 | [IDEA-056](29_IDEA-056_checkpoint_selection_confirmation_results.md#2-六组新的-complete-oof-配对) |

任何论文总表都应把这组历史 outer-test 结果与后期 `outer_test_accessed=false` 的主任务/机制实验分栏，避免把经过多轮观察的同一批猫包装成新的独立测试集。

## 9. 当前模型层级

| 层级 | 模型/证据 | 允许的用途 | 禁止的解读 |
|---|---|---|---|
| Tier 1：当前主模型 | **A0 tuned frozen AST** | 默认主结果、稳健参照、后续外部验证基线 | 不声称已在真正独立的新猫年龄数据上完成确认 |
| Tier 2：exploratory-positive 主要研究候选 | **C1 bounded RMS-relative additive residual** | 与 A0 成对报告；用于冻结外部/前瞻验证；保留正平均方向 | 绝非 primary / confirmed；不称“显著优于 A0”“最终确认成功”或“机制已证实” |
| Tier 3：机制线索 | **G3 spectral-energy/voice-quality** | 作为未来外部数据上的冻结消融候选 | 不称 C1 收益已被归因于 G3 |
| Tier 4：探索性历史证据 | A1、B1、U1、S1、Hybrid；probe-guided adapter 独有增量为负结果／历史混合证据 | 方法讨论、负结果、假设生成 | 不进入当前主模型排序 |
| Closed / no-go 配置 | 已测试 LoRA、LayerMix、token injection、spatial adapter、ConvPass、D1 等 | 作为预注册负结果与设计边界 | 不在同一数据上事后改超参数并称为原实验延续 |

该层级不是按单轮最高分排序，而是综合预注册 gate、配对稳定性、概率质量、尾部风险、可解释边界和重复使用数据的事实。

## 10. v2 → v3 变更清单

1. **主叙事从 adapter 转向 frozen AST + 年龄声学。** v2/v2.1 以 probe-guided adapter 为最高平均候选；v3 根据计算非确定性诊断与真正 deterministic 的新种子复验，将其独有增量记为负结果／历史混合证据，并以 A0 为唯一 primary、C1 为 exploratory-positive 主要研究候选。[adapter 复验](35_AST_adapter_deterministic_replication_results.md#科学解释与可复用价值)
2. **冻结 checkpoint 选择口径。** v3 明确以后期采用的最低动物级验证 CE 为 A0/C1 共同参考规则，而不是旧的 call-level 选择。[IDEA-056](29_IDEA-056_checkpoint_selection_confirmation_results.md#结论摘要)
3. **加入 IDEA-068 至 IDEA-076 的完整年龄声学演变。** 不只呈现 C1 有利轮次，同时保留 IDEA-074 的反向结果与 IDEA-076 最终 gate 失败。[C1 台账](../metadata/experiments/meowagenet_C1_evidence_ledger_pre_IDEA076.json)、[最终确认](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#最终研究状态)
4. **加入 IDEA-080/081 的新种子描述性更新。** 两轮 C1−A0 再次为正，但不重开或覆盖 IDEA-076 的最终判定。[IDEA-080](53_IDEA-080_AST_last_block_LoRA_C1_factorial_results.md#主要结果)、[IDEA-081](53_IDEA-081_AST_tail_ConvPass_C1_factorial_results.md#3-四管线平均结果)
5. **加入 IDEA-082 的来源归因边界。** v3 明确拒绝“纯 F0 驱动”叙事；G3 最强但信息门失败。[IDEA-082](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#5-结论与下一步边界)
6. **完整纳入 PEFT/表示负结果。** LoRA、LayerMix、special-token injection、spatial adapter、ConvPass 均按锁定配对和 gate 报告，不因局部正均值而晋级。
7. **重新划分证据类型。** 历史 outer-test、后期 inner-validation、机制消融和外部 benchmark 分开书写；不再把跨轮同一 111 猫描述成扩大样本量。

## 11. 论文可用与不可用表述

### 11.1 可用表述

- “在同一 MeowAgeNet 动物分组内部验证框架中，冻结 AST 一直是最稳健的参照；一个受 RMS 相对幅度约束的年龄声学加法残差在多组新初始化中反复呈现正平均方向，但未通过最终预注册稳定性与安全门。”
- “C1 在 IDEA-076 中相对 A0 的平均 Macro-F1 差为 `+0.008989`，但仅 `9/18` seed×repeat 严格为正，最差 split 为 `−0.049131`，且 CE 与一个 base seed 的 senior recall 安全条件失败，因此不构成确认性胜出。”[来源](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#预注册-gate-判定)
- “分组消融未确认 F0、稳定性或谱能量/声质中的任何单一组为 C1 收益来源；G3 是最强探索性线索，但 matched-shuffled 信息门未通过。”[来源](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#5-结论与下一步边界)
- “冻结 AST 在 CatMeows 未见猫情境分类中稳定高于先验 dummy，这支持跨个体情境表征，而非猫年龄表征。”[来源](37_IDEA-067_external_AST_representation_results.md#catmeows未见猫的情境分类)
- “当前 probe-guided adapter 的独有增量是负结果／历史混合证据；固定 LoRA、LayerMix、special-token injection、spatial adapter 与 tail ConvPass 均未显示稳定、可确认的 A0 增益。”

### 11.2 不可用表述

- 不可写：“C1 已显著优于 A0”或“C1 已成为确认的新主模型”。
- 不可写：“54 次独立实验/54 个独立样本证明 C1 有效”；这些单元复用相同 111 猫。[来源](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#跨轮描述性更新)
- 不可写：“F0 是 C1 增益的原因”或“G3 已证明声质机制”；IDEA-082 的所有 `source_supported` 均为 false。[来源](../metadata/experiments/meowagenet_idea082_age_acoustic_group_ablation_v1_results.json)
- 不可写：“formal-v2.1 是独立外部验证”；它是 pilot-informed、同一 111 猫上的 repeated internal validation。[来源](09_formal_protocol_v2_freeze.md#2-pilot-与正式-v2-的证据边界)
- 不可写：“adapter 带来了 AST 的主要提升”；formal-v2.1 中 head−VGGish 为 `+0.0713`，adapter−head 仅 `+0.0052`，后者又在确定性复验中反向。[formal-v2.1](12_formal_v2_1_core_results.md#5-secondary-metrics-与类别优势)、[复验](35_AST_adapter_deterministic_replication_results.md#新种子确定性复现)
- 不可用犬 benchmark 或 CatMeows 情境结果替代 MeowAgeNet 年龄主任务的确认。

## 12. 下一步计划

1. **停止在 MeowAgeNet 上追加 seed、重分组或结果驱动微调 C1/G3。** IDEA-076 已声明停止同数据 seed 扩展，IDEA-082 也禁止按本轮结果重组特征或改门槛。[IDEA-076](46_IDEA-076_C1_Meow_final_seed_confirmation_results.md#结论)、[IDEA-082](57_IDEA-082_age_acoustic_mechanism_group_ablation_results.md#5-结论与下一步边界)
2. **优先获取真正独立的猫年龄数据。** 最小确认设计应冻结 A0/C1 公式、20 维特征、预处理、checkpoint 规则、主指标和安全门；新动物不得参与任何现有调参。
3. **若只能做机制复验，冻结 G3 real-vs-shuffled。** 在外部或前瞻收集数据上原样检验 G3，不在当前 111 猫上继续挑组。[IDEA-082 推荐](../metadata/experiments/meowagenet_idea082_age_acoustic_group_ablation_v1_results.json)
4. **CatMeows 用于保真而非年龄确认。** 检查 C1/G3 是否损害冻结 AST 的跨猫情境表征；若标签不含年龄，不作年龄效力推断。
5. **犬数据保持次级。** 只有在新的预注册设计与资源允许时才进入完整犬 benchmark；当前 2,290-unit 子集只保留为弱跨物种基线。[IDEA-075](45_IDEA-075_dog_C1_age_sensitive_AST_benchmark_results.md#最终研究状态)
6. **论文主结果建议成对呈现 A0/C1。** 主表报告 A0；研究候选表报告 C1 的逐轮配对差、正/平/负、CE/Brier、最差 split 与 senior-recall 安全项。所有历史 outer-test 结果单列为“development history”。

## 13. 最终结论

v3 的最窄、最可靠结论是：**冻结 AST（A0）是唯一的 primary / 当前主模型；C1 是具有反复正平均信号但未完成确认的 exploratory-positive 主要研究候选，绝非 primary / confirmed。** 年龄声学信息值得继续研究，但现有证据不允许把收益简单归因于 F0，也不允许宣称任何单一特征组已被确认。G3 是下一步最值得在真正外部/前瞻数据上原样复验的机制线索。Probe-guided adapter 的独有增量应作为负结果／历史混合证据，后续 LoRA、LayerMix、token injection、spatial adapter、ConvPass 等路线也应作为完整、诚实的探索性或负结果保留，而不进入当前主模型层级；CatMeows 与犬数据只构成 external boundary。
