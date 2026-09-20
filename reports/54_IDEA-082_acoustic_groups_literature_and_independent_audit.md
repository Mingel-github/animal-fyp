# IDEA-082：20 维年龄声学特征分组、文献依据与独立审计

日期：2026-09-18  
角色：独立方法与审计轨  
状态：分组先验已锁定；最终独立审计 **GO（32/32）**；未检查任何特征—标签关联、实验结果或经验相关矩阵；未启动 GPU；未访问 outer test。

## 1. 结论先行

IDEA-068 的 20 维 schema 可在不看标签和结果的前提下，划成互斥且穷尽的三个**主分组**：

1. `f0_level_and_contour`（5 维）：F0 水平、范围与整体轮廓；
2. `source_stability_and_periodicity`（10 维）：F0/周期/振幅稳定性、周期性与 pYIN voicing 可靠性；
3. `spectral_energy`（5 维）：宽带能量、谱倾斜与谱平坦度。

主分组仅用于预注册消融，不表示每个量只有一种物理来源。必须同时保留以下解释边界：

- `log_f0_iqr/std/mad/abs_delta_median` 会同时响应有意的音高轮廓和不稳定的发声/跟踪；
- `period_variation_proxy` 是帧级 F0 周期变化代理，不是真正逐声门周期的 jitter；
- `amplitude_variation_proxy` 是帧级 RMS 变化代理，不是真正逐周期的 shimmer；
- pYIN 的 voiced probability 是模型后验/跟踪置信度，不能直接等同于声带生理稳定性；
- `periodicity_median` 与 `hnr_db_median` 来自同一帧级自相关量，信息近冗余；
- 谱倾斜同时受声源、声道滤波和录音链影响，当前提取器不能将三者解耦；
- 谱平坦度既是谱能量分布指标，也会响应非周期噪声。

机器可读的先验已锁定在 `metadata/experiments/idea082_independent_feature_grouping_review_v1.json`。锁定发生在本轨看到任何 IDEA-082 候选 protocol/preflight 之前。

## 2. IDEA-068 提取器与 cache 核查

### 2.1 固定来源

- 提取器：`scripts/run_meowagenet_idea068_age_sensitive_ast.py`
- 提取器 SHA-256：`3eab879d1a39bfdc4c3d56d3e105d628c0a38222e550760965f71706cccfe745`
- cache：`runs/meowagenet_idea068_age_sensitive_ast_v1/features/age_sensitive_acoustic_features.npz`
- cache SHA-256：`ba951db756436ea00adb5c7241ea0264de9fa0f6674a84f274812434d93e26ba`
- extraction summary SHA-256：`022cc2d54fc2af25430e327ebb60d8daff733d627c76a522d3baf81f3b4b2cda`
- 792 calls、20 features；772 calls 全有限；其余 20 calls 除三项 voicing 指标外均有缺失。
- extraction summary 记录 `label_information_used=false`、`source_hashes_verified=792`。

提取固定为 16 kHz；pYIN 范围 60–2000 Hz；窗长 1024、hop 160；谱倾斜拟合频段 200–4000 Hz。分组依据只有代码公式、信号含义和原始/权威文献，没有查看特征相关矩阵、年龄标签或既有分组消融结果。

### 2.2 逐特征物理含义与主分组

| # | 特征 | IDEA-068 的实际定义 | 主分组 | 必须保留的重叠/限制 |
|---:|---|---|---|---|
| 1 | `log_f0_q10` | voiced `log(F0)` 的 10% 分位 | F0 | F0 低端水平 |
| 2 | `log_f0_median` | voiced `log(F0)` 中位数 | F0 | 典型 F0 水平 |
| 3 | `log_f0_q90` | voiced `log(F0)` 的 90% 分位 | F0 | F0 高端水平 |
| 4 | `log_f0_iqr` | voiced `log(F0)` 的 Q75−Q25 | F0 | 轮廓范围，也混入稳定性/跟踪误差 |
| 5 | `log_f0_std` | voiced `log(F0)` 总体标准差 | 稳定性 | 也测量有意的音调轮廓跨度 |
| 6 | `log_f0_mad` | voiced `log(F0)` 中位绝对偏差 | 稳定性 | 也测量稳健轮廓离散度 |
| 7 | `log_f0_slope` | voiced `log(F0)` 对全 call 归一化时间的 OLS 斜率 | F0 | 跟踪断裂会影响斜率 |
| 8 | `log_f0_abs_delta_median` | 相邻且均 voiced 的 `|Δlog(F0)|` 中位数 | 稳定性 | 快速轮廓变化与不稳定不可分 |
| 9 | `period_variation_proxy` | 相邻 voiced `|Δ(1/F0)|` 中位数 / `median(1/F0)` | 稳定性 | 帧级代理，不是真实 cycle jitter |
| 10 | `voiced_fraction` | 有限正 F0 帧占比 | 稳定性 | 同时反映停顿、call 结构、SNR 和算法可靠性 |
| 11 | `voiced_probability_mean` | pYIN voiced posterior 的均值 | 稳定性 | 算法置信度，不是纯生理量 |
| 12 | `voiced_probability_std` | pYIN voiced posterior 的标准差 | 稳定性 | 同时受分段、SNR 和算法置信度影响 |
| 13 | `periodicity_median` | 在估计 F0 lag 处归一化自相关的中位数 | 稳定性 | 与 HNR 同源、近冗余 |
| 14 | `hnr_db_median` | 同一自相关 `r` 的 `10log10(r/(1−r))` 帧值中位数 | 稳定性 | 与 periodicity 同源；也属谱/噪声质量 |
| 15 | `amplitude_variation_proxy` | 相邻 voiced 帧 `|ΔRMS|` 中位数 / voiced RMS 中位数 | 稳定性 | 帧级 envelope 代理，不是真实 shimmer；也属能量动态 |
| 16 | `log_rms_median` | voiced `log(RMS+1e-10)` 中位数 | 谱/能量 | 受声源幅度与录音增益影响 |
| 17 | `log_rms_iqr` | voiced log-RMS 的 Q75−Q25 | 谱/能量 | 也反映振幅稳定性和 call envelope |
| 18 | `spectral_tilt_median` | 200–4000 Hz 内每帧 `log|X| ~ log f` 斜率的 voiced 中位数 | 谱/能量 | 声源、声道滤波、录音链混叠 |
| 19 | `spectral_tilt_iqr` | 上述谱斜率的 voiced IQR | 谱/能量 | source/filter 混叠，也反映时变构音/稳定性 |
| 20 | `spectral_flatness_median` | voiced 帧 spectral flatness 中位数 | 谱/能量 | 同时反映谱形和非周期噪声 |

## 3. 分组锁定

### 3.1 F0 水平与轮廓（5）

`log_f0_q10`, `log_f0_median`, `log_f0_q90`, `log_f0_iqr`, `log_f0_slope`

这组回答“叫声的典型音高、音高范围和整体升降方向是什么”。IQR 放在这里，因为它首先是稳健的 call 内 F0 范围；但不能把它解释成纯轮廓指标。

### 3.2 声源稳定性、周期性与 voicing 可靠性（10）

`log_f0_std`, `log_f0_mad`, `log_f0_abs_delta_median`, `period_variation_proxy`, `voiced_fraction`, `voiced_probability_mean`, `voiced_probability_std`, `periodicity_median`, `hnr_db_median`, `amplitude_variation_proxy`

这组回答“F0/周期/振幅在帧尺度上是否稳定，以及信号有多像可靠的周期声源”。它是三组里解释边界最多的一组：F0 dispersion 可由轮廓造成，voicing posterior 是算法量，两个 perturbation 指标也不是临床语音学中的逐周期 jitter/shimmer。

### 3.3 谱形与能量（5）

`log_rms_median`, `log_rms_iqr`, `spectral_tilt_median`, `spectral_tilt_iqr`, `spectral_flatness_median`

这组回答“能量有多强、多变，以及频谱能量怎样随频率分布”。当前 20 维 schema 没有 formant、共振峰带宽或显式声道长度，因此本组不能被命名成纯 `filter` 组。

## 4. 原始/权威文献证据

### 4.1 猫叫年龄与 F0

Van Toor et al. 的 MeowAgeNet 正式论文把 F0 随年龄变化作为猫龄预测的声学背景，并明确说明有限的非人类动物数据使迁移学习和谨慎验证很重要；其数据页还明确说明每只猫分配唯一 ID，以便训练/测试时把同猫 calls 放在一起、避免泄漏。已视觉核验本地正式 PDF 第 2、5 页。正式页面：<https://www.nature.com/articles/s41598-025-17986-z>。

Schötz, van de Weijer & Eklund 对 50 只成年家猫、969 个 meow 的线性混合模型显示：年龄对 mean F0 的估计为负（表 2：−7.718，SE 3.766），正文解释为老猫 mean F0 倾向低于年轻猫；年龄对 F0 range 未见显著效应。论文同时强调 context、sex 与个体差异，并在 future work 明确建议分析 source/filter 谱特征、perturbation、spectral tilt、HNR 与 resonance。已视觉核验本地正式 PDF 第 5、7 页。正式页面：<https://www.sciencedirect.com/science/article/pii/S0168159123003180>。

这两篇来源支持“F0 level 是年龄候选信息”，但也反对把 F0 range、声质或谱特征未经独立检验就解释为猫龄机制。

### 4.2 Source–filter 边界

Taylor & Reby 的综述把哺乳动物发声分成声门/声源产生的基频与谐波，以及声道滤波形成的谱包络和共振峰；年龄、性别、体型等信息可能通过两部分共同表达。原始页面：<https://zslpublications.onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-7998.2009.00661.x>。

Briefer 的哺乳动物情绪发声综述进一步把 F0 水平、范围/变化、jitter、shimmer、谱能量、formant 等放在 source–filter 和唤醒度机制下讨论。原始页面：<https://zslpublications.onlinelibrary.wiley.com/doi/10.1111/j.1469-7998.2012.00920.x>。

因此，本实验可以用“F0 / 稳定性 / 谱能量”作为**计算分组**，但没有资格把 `spectral_tilt` 或 `flatness` 声称成已分离的声道滤波参数。

### 4.3 周期性、HNR、jitter 与 pYIN

Boersma 1993 的原始方法以短时自相关估计周期性与 HNR；HNR 可由周期成分与噪声成分之比转换到 dB。IDEA-068 的 `periodicity_median` 和 `hnr_db_median` 正是共享同一自相关 `r`，只是在聚合前作单调变换。原始 PDF：<https://www.fon.hum.uva.nl/david/ba_shs/2010/Boersma_Proceedings_1993.pdf>。

Praat 的官方定义中，local jitter 是逐周期长度差的平均绝对值除以平均周期，并带有合法周期筛选；IDEA-068 则对帧级 `1/F0` 差取中位数，所以只能称 `period_variation_proxy`。官方页面：<https://praat.org/manual/PointProcess__Get_jitter__local____.html>。Praat 对 harmonicity/HNR 的定义同样把它视为周期信号与噪声能量的比例：<https://praat.org/manual/Harmonicity.html>。

Mauch & Dixon 的 pYIN 原始论文把多个 YIN 候选概率化，并经时序模型得到 F0/voicing；因此 `voiced_probability_*` 是算法后验摘要，不是独立测得的生理变量。原始页面：<https://ieeexplore.ieee.org/document/6853678>。

### 4.4 声学描述符与消融原则

openSMILE 的原始论文把音频描述组织为 low-level descriptors 与统计 functionals，支持把 F0、能量、谱和声音质量指标按事先定义的族进行分析。原始页面：<https://portal.fis.tum.de/en/publications/opensmile-the-munich-versatile-and-fast-open-source-audio-feature/>。

IDEA-082 的三组消融必须在同一 split、同一初始化、同一 batch 顺序和同一 checkpoint 规则下配对；否则差异会混入训练随机性。组名和成员必须先于结果锁定，失败后不得再按经验相关、单特征效果或年龄方向重划分。

## 5. 独立审计清单

### 5.1 数据、cache 与标签隔离

- [ ] feature cache 路径、20 维顺序及 SHA-256 与本报告第 2.1 节完全一致；禁止重新提取后仍沿用旧 hash。
- [ ] extraction summary 保持 `label_information_used=false` 和 `source_hashes_verified=792`。
- [ ] 三个主组互斥、并集恰好等于 20 维；不得复制一个重叠特征到多个输入组。
- [ ] 组别只控制输入列；不得把 age label、role、fold、cat ID 或结果指标拼入特征。
- [ ] 对 20 个部分缺失的 call，不能删除到改变各 variant 的验证总体；使用训练折统计量填补。

### 5.2 折内预处理

- [ ] 每个 seed/repeat/fold 独立计算训练猫所对应 calls 的逐列 `nanmedian`。
- [ ] 只用该折训练数据填补后，计算训练折 mean/std；零或近零 std 固定为 1。
- [ ] validation 使用冻结的训练折 median/mean/std；禁止在整套数据或 train+validation 上重算。
- [ ] 若 AST embedding 也标准化，同样只用当前训练折统计量。
- [ ] 每个 fit summary 写出预处理统计量或其 hash，足以重算核对。

### 5.3 同猫分组与泄漏

- [ ] `cat_id` 是 split 和动物级汇总单位；一个 fit 内 train/validation cat 集合交集为空。
- [ ] 一个 cat 的所有 calls 只能处在一个 role；call 重复、cat 重复及预测总体都应检查。
- [ ] role CSV 的路径和 hash 必须锁定；不得从标签重新生成“更均衡”的 split。
- [ ] early stopping 和 checkpoint selection 仅使用 inner validation；`outer_test_accessed=false`、`outer_test_predictions=false`。

### 5.4 配对公平性

- [ ] 同一 seed/repeat/fold 的共同 head 参数逐张量同 hash；组分支的可比参数也采用相同初始化规则。
- [ ] cat-set batch 列表和每个 epoch 的 cat 顺序在各 variant 间一致，且每个训练 cat 每 epoch 恰好出现一次。
- [ ] optimizer、学习率、epoch 上限、patience、损失、checkpoint 规则和 class weights 一致。
- [ ] 若缺省分支以零输入/零残差实现，初始化 logits 与 loss 应在容差内相同；若模型输入维度不同，必须提供等价的匹配初始化证明。

### 5.5 Shuffled control

- [ ] 每个 `(repeat, fold, role)` 单独生成确定性置换；`train` 和 `validation` 不得共享或跨越置换池。
- [ ] 置换只能重排当前 role 的 call feature rows；不能读取年龄标签、预测、损失、全数据距离或跨 role 关系。
- [ ] 每个 role 记录 seed/material、置换 index SHA-256、置换后输入 SHA-256、长度、一一映射、fixed points 和 multiset equality。
- [ ] 同一 cell 的各 feature group 使用同一 role-local mapping，以便 real/shuffled 只改变声学信息配对而不改变样本总体。
- [ ] preflight 必须重放置换并复核 hash；任何跨 role index、重复、遗漏、越界或 hash 不一致均为 NO-GO。
- [ ] 若声学预处理发生在置换之前，训练统计量只得来自 train role；validation 的 shuffled input 仍须使用冻结的 train-role 统计量。

### 5.6 配对汇总层级

- [ ] 最小配对单元为 `(seed, repeat, fold)`，每个 variant 在每个 cell 恰好一个完成 fit。
- [ ] 主指标在 cat-level validation prediction 上计算；call 不作为独立样本扩大 n。
- [ ] 报告逐 cell 差值、均值、正差 cell 数、worst-cell，以及每个 repeat 的 senior recall 差值。
- [ ] pooled CE/Brier 必须保留 cat-level prediction 和配对标签总体；不得只报最优 seed/fold。
- [ ] 缺失或失败的 cell 不能静默丢弃，也不能在看结果后补改 seed、group 或 gate。

## 6. GO / NO-GO 规则

在候选 protocol 和 CPU preflight 出现后，只有以下条件全部具备才给 **GO**：

1. 候选引用同一 cache hash、同一 20 维顺序和预锁定 5/10/5 分组；
2. 代码或可重放 preflight 证实折内 median/imputation/standardization；
3. cat role 零交叉、标签未进入特征、outer test 保持关闭；
4. shuffled controls 是可重放、带 hash 的 role 内独立确定性置换，不使用全数据关系；
5. common initialization、batch order 和 checkpoint selection 有逐项证据；
6. 配对汇总以 seed/repeat/fold cell 和 cat-level prediction 为单位；
7. preflight 自身的协议、runner、cache、roles 与 manifest hash 相互一致。

以下任一情况直接 **NO-GO**：全数据标准化；以 call 随机切分；feature cache hash 不符；分组成员在看结果后改变；outer-test 标志缺失或为真；不同 variant 使用不同 batch 顺序/初始化而仍声称配对；把帧级 proxy 宣称为真实 jitter/shimmer；把谱倾斜宣称为已分离的 filter 参数。

## 7. 本地正式 PDF 核验记录

- `C:/Users/zhu/Desktop/essay_article/Audio/00_Baseline_Candidates/2025_vanToor_MeowAgeNet_Feline_Age_Prediction.pdf`  
  SHA-256 `9ff1117763624964af18973040919ed17bd3eeb8a6b6ff5e31e3889268f7ffda`；18 页；已视觉核验第 2 页年龄/F0 背景及第 5 页 unique-cat ID 防泄漏说明。
- `C:/Users/zhu/Desktop/essay_article/Audio/01_Application_Papers/Animal_Vocalization/Emotion_Age_And_Attributes/2023_tz_Context_effects_duration_fundamental_frequency_intonation_human_directed_domestic_cat.pdf`  
  SHA-256 `5b2fcacc08c2f360c43c8817172288bf48502f3c2ff7ed40d543b6a1bb8fe038`；已视觉核验第 5 页年龄系数和第 7 页 source/filter、perturbation、tilt、HNR 建议。

这些本地 PDF 是本次动物/猫叫论证的主要材料；网页仅用于链接原始/权威方法来源。博客和二手教程未用于方法判定。

## 8. 最终候选审计结论

最终结论：**GO，可由研究总监另行决定是否授权正式 GPU run。** 本结论不等同于本轨自行授权，且本轨没有启动 GPU。

- 32/32 独立硬检查通过；主轨与独立测试合计 15/15 通过。
- protocol SHA-256：`0b5e4a4c0be591925811a42769555cab7e5ac9ba8cd2781ac91d1d341e782022`
- runner SHA-256：`2ce03190d3079cbfe34fbdd140288b2309dfa12ace54f8d21279e66fa7f2c415`
- CPU preflight SHA-256：`b0967358ed610ce41441cf3bb7bde2cbaa950d38727bb2359eeef7724d783952`
- independent audit SHA-256：`5fb2ae5e8224981146563ce3930df8709ab37ec3ff5942850ff0d5d0513fb4c3`
- feature cache SHA-256：`ba951db756436ea00adb5c7241ea0264de9fa0f6674a84f274812434d93e26ba`
- roles SHA-256：`87deda39808297e1af5b71283e1d7487a7b88d9288cb492c488e3e64fb91c433`

G12 固定采用原 20 维 schema 顺序 `[0..14]`；其信息集合严格等于 G1∪G2。protocol、runner 与 preflight 三者一致。独立审计重放了全部 12 个 `(repeat, fold)` cell 的 train/validation 映射，共 24 个 mapping SHA-256：全部逐个一致、fixed points 为 0、role 内一一映射、未触及 test，且 train/validation cat 集合无交叉。
