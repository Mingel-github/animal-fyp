# IDEA-082 年龄声学机制分组：中文文献依据与预注册边界

日期：2026-09-18  
状态：结果盲、分组已锁定；仅批准协议实现与 CPU preflight

## 结论先行

本实验把 IDEA-068 的 20 维标签盲声学缓存拆为互斥的 5/10/5 三组，并额外预注册一个 15 维 G1+G2 组合。该拆法既遵循声源—滤波器与音高/周期性分析的常用区分，也明确承认若干代理量并非纯生理测量。分组在任何 IDEA-082 模型结果产生之前锁定；不得在结果后改回 8/7/5 或其他 taxonomy。

## 文献依据

1. van Toor、Qazi 与 Paladini（Scientific Reports, 2025）在 MeowAgeNet 原始论文中指出猫的 F0 会随年龄变化，并汇总多物种中随年龄降低的 F0 证据；该论文的数据以 cat ID 分组以避免泄漏，同时明确提出未来应研究叫声的 spectral characteristics。这支持把 F0 作为独立的首要机制组，也支持单列频谱/能量组。来源 PDF SHA-256：`9ff1117763624964af18973040919ed17bd3eeb8a6b6ff5e31e3889268f7ffda`。
2. Schötz 等（Applied Animal Behaviour Science 270, 2024, 106146）分析 50 只成年猫的 969 次 meow，在模型中控制年龄和性别；较老与雄性猫的 mean F0 更低，但 F0 contour 同时随情境变化。论文还将 HNR、spectral tilt、jitter/shimmer 与声压/共振列为需另行分析的 voice-quality/spectral 线索。这同时支持 G1 的年龄相关性和“不要把所有 F0 变异都解释为年龄”的对照需求。来源 PDF SHA-256：`5b2fcacc08c2f360c43c8817172288bf48502f3c2ff7ed40d543b6a1bb8fe038`。
3. Klenova 等（Animals, 2024）在 56 个猎豹 litter、14 个年龄等级、1,977 次 chirp 上发现 `f0max` 与年龄高度负相关，并同时测量 F0 起止/极值、时长、峰频率和能量四分位。频率参数随年龄显著变化，但年长个体的精确年龄预测更困难。这为 F0 level/contour 的跨猫科生理合理性提供外部依据，也提醒我们不能把一个总体方向直接当成各年龄阶段稳定分类增益。来源 PDF SHA-256：`e7a05d6f2bc84ffe761ad2ccc04170647de0a30b810e8e9742085817427ab355`。
4. Zhu 等（ACM MM, 2025）使用带精确出生日期、breed 与 dog ID 的纵向犬声数据，报告不同生命阶段的 bark sequence、元素类型与能量/频谱结构变化，并把 pitch contours、energy distributions 与 formants 列为后续机制分析对象。该证据不是猫科直接验证，因此只用于支持“年龄信号可能分布在相互不同的声学族”这一分组策略，不用于对具体猫组作阳性预测。来源 PDF SHA-256：`05d7fe789c1a585c6ec2a7f97a2ad5f4f0380d1acb9cb8f8256b9209a9d0fba4`。

## 20 维特征到机制组的精确映射

| 索引 | 特征 | 冻结主组 | 操作性含义 / 边界 |
|---:|---|---|---|
| 0 | `log_f0_q10` | G1 F0 | 有声帧 F0 下尾水平 |
| 1 | `log_f0_median` | G1 F0 | 典型 F0 水平 |
| 2 | `log_f0_q90` | G1 F0 | 有声帧 F0 上尾水平 |
| 3 | `log_f0_iqr` | G1 F0 | 鲁棒音高跨度；亦可能含不稳定性 |
| 4 | `log_f0_std` | G2 stability | 全局 F0 变异；亦响应有意 intonation |
| 5 | `log_f0_mad` | G2 stability | 鲁棒 F0 偏差；亦响应 contour dispersion |
| 6 | `log_f0_slope` | G1 F0 | 整体上升/下降轨迹 |
| 7 | `log_f0_abs_delta_median` | G2 stability | 相邻有声帧的典型 F0 变化 |
| 8 | `period_variation_proxy` | G2 stability | 帧尺度周期变化代理，不是真正逐周期 jitter |
| 9 | `voiced_fraction` | G2 stability | pYIN 接受为有声的帧比例，兼受结构/SNR 影响 |
| 10 | `voiced_probability_mean` | G2 stability | 平均 voicing 后验支持，兼具算法置信度含义 |
| 11 | `voiced_probability_std` | G2 stability | voicing 后验随时间的稳定性 |
| 12 | `periodicity_median` | G2 stability | F0 lag 处归一化自相关强度 |
| 13 | `hnr_db_median` | G2 stability | 周期/非周期能量比；与 periodicity 近冗余 |
| 14 | `amplitude_variation_proxy` | G2 stability | 有声帧 RMS 的帧尺度变化，不是真正逐周期 shimmer |
| 15 | `log_rms_median` | G3 spectral-energy | 典型宽带能量，受录音增益影响 |
| 16 | `log_rms_iqr` | G3 spectral-energy | 呼叫内能量跨度，兼具包络/稳定性含义 |
| 17 | `spectral_tilt_median` | G3 spectral-energy | 200–4000 Hz 频谱斜率，声源/滤波器不可完全分离 |
| 18 | `spectral_tilt_iqr` | G3 spectral-energy | 频谱斜率的呼叫内变化 |
| 19 | `spectral_flatness_median` | G3 spectral-energy | tonal 与 noise-like 的频谱分布代理 |

G1 索引固定为 `[0,1,2,3,6]`；G2 为 `[4,5,7,8,9,10,11,12,13,14]`；G3 为 `[15,16,17,18,19]`；G12 为 `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]`。前三组两两不交且覆盖 0–19，G12 是唯一明确重叠组。

## 容量与因果解释边界

| 组 | 输入维数 | 隐层宽度 | 年龄支路参数 | 与完整 C1 9,068 的差 | 总参数 |
|---|---:|---:|---:|---:|---:|
| G1 | 5 | 67 | 9,106 | +38 | 108,181 |
| G2 | 10 | 64 | 9,024 | −44 | 108,099 |
| G3 | 5 | 67 | 9,106 | +38 | 108,181 |
| G12 | 15 | 62 | 9,056 | −12 | 108,131 |

宽度按公式 `w*(d+129)+128` 选择最接近完整 C1 预算的整数；不是按结果调节。组间残余预算差最大为 82 个总参数，所有 real/shuffled 成对比较则严格等参数。因此核心机制结论来自 real−shuffled，而不是不同组之间未经校正的胜负排序。

## 既有结果边界

IDEA-076 的完整 C1 在 18 个 seed×repeat 单元上的 C1−A0 平均 Macro-F1 为正，但只有 9/18 为正，最差 split 明显为负，CE 也未通过；因此完整 C1 只能作为“有方向性但不稳定”的历史参考。本实验不得把局部分组阳性扩写为完整 C1 已确认，也不得继续同数据 seed-bank 扩张来修复既有结论。

## 预注册防泄漏与对照

- 分组只读取 extractor 定义和文献，不读取标签—特征关联、经验相关矩阵或 IDEA-082 结果。
- shuffling 在每个 repeat/fold 的 train 与 validation role 内分别进行，使用标签盲确定性 derangement；test 不加载、不置换、不预测。
- 同一 cell 的四个组复用同一行映射；训练特征多重集、缺失值模式和训练折标准化统计保持完全一致。
- A0 因新 seed 无法严格复用历史结果，故每 cell 只重跑一次，并由所有成对比较共享。
- 冻结 C1 上限、animal-level validation CE checkpoint、fold、batch 顺序、聚合和门槛；不得在看到结果后改组或选组。

## CPU preflight 的 GO 条件

只有以下全部通过才可向总监报告“工程上可运行”：依赖哈希、20 维 schema 与 5/10/5 分区、12 个 role cell 的动物/调用不交、seed 冲突、置换无固定点且不跨 role、real/shuffled 多重集与训练统计一致、九管线初始 logits 一致、同组 real/shuffled 完整初始状态一致、参数预算、零初始化下梯度可达性、C1 相对 RMS 上限。该 GO 不等于允许 GPU；正式运行仍需总监单独放行。

