# IDEA-085 声学时序残差：文献动机与轨迹缓存

日期：2026-09-19  
状态：**逐帧缓存完成并通过 CPU 验证；模型预检与任何 GPU 运行另行授权。**

## 研究问题与边界

现有 C1/U1 的 20 维声学输入包含斜率、四分位距、相邻变化等统计，但仍把每个叫声压缩为一个固定向量，不能保留“何时上升、下降、停顿或改变音质”的完整顺序。IDEA-085 因此只问一个新的内部机制问题：显式保存并建模逐帧声学轨迹，能否提供统计摘要之外的信息。

这不重复 IDEA-052。IDEA-052 处理 AST 表征 token 的池化；IDEA-085 保存的是六条可解释的原始声学时间序列。它也不是外部确认，且不会修改已有实验、v3 或 outer-test 边界。

## 文献动机

[Ryu et al. (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/ryu25_interspeech.html) 在人类语音情绪识别中显式建模 pitch contour，并把 pitch 序列与预训练语音表征通过 cross-attention 结合；其摘要还指出 raw pitch 与 z-score normalization 的作用会随数据集而变化。这支持两个有限的工程动机：轮廓顺序可能包含固定统计不能表达的信息；不应在结果前假定逐叫声 F0 归一化必然更好。该论文研究的是人类情绪，不是猫年龄证据，也不授权复制其任务结论。

[Schötz and van de Weijer (Speech Prosody 2014)](https://www.isca-archive.org/speechprosody_2014/schotz14_speechprosody.html) 报告食物与就医等待两种猫叫语境中，F0 轮廓分别呈更多上升与下降趋势，听者也能高于随机水平区分语境。这表明猫叫的时间性语调可能具有可感知结构，因而为保留完整 contour 提供物种相关动机；但“语境可区分”不等于“年龄可预测”，情绪、语境与年龄主张必须严格分开。

因此，本项目只把文献用于预先固定“保留时间顺序”的工程假设，不把论文结果当作 IDEA-085 的年龄证据，也不据此选择结果后的通道、宽度、分组或门槛。

## 缓存设计

旧缓存 `age_sensitive_acoustic_features.npz` 只有 `[792,20]` 的逐叫声统计，没有逐帧数据，不能直接复用。新缓存从与 IDEA-068 完全相同的 792 个源音频重新提取，固定设置如下：

- 16 kHz；pYIN `60–2000 Hz`；frame length `1024`；hop `160`，即原始 `10 ms` 网格；center=true。
- 六通道顺序固定为 `log_f0, voiced_probability, periodicity, log_rms, spectral_tilt, spectral_flatness`。
- 不删除无声或缺失帧，不拼接有效 F0，不做逐叫声 F0 归一化；NaN 原样保存，模型必须另建 finite indicators。
- `log_f0` 仅在 pYIN 给出有效 F0 时有限；`periodicity` 沿用 IDEA-068 的对应周期归一化自相关，因此同样在无有效 F0 时为 NaN。
- `log_rms`、`spectral_tilt` 和 `spectral_flatness` 在所有可计算帧上保存。这里与 IDEA-068 不同：旧 20 维摘要对这三类量只在 finite-F0 voiced mask 上汇总；新缓存保留全帧，但仍可使用 `finite(log_f0)` 重建旧摘要。
- `time_seconds` 只用于审计，明确不输入模型。
- 所有音频在提取前逐个核对源 SHA-256；缓存按明确的 `call_id` 对齐，而不是依赖隐式行号。

NPZ 接口为：

| 数组 | 形状 | 类型 | 用途 |
|---|---|---|---|
| `call_ids` | `[792]` | Unicode | 显式调用身份 |
| `offsets` | `[793]` | int64 | 每个调用在扁平帧矩阵中的边界 |
| `values` | `[57848,6]` | float32 | 六通道轨迹 |
| `feature_names` | `[6]` | Unicode | 锁定通道顺序 |
| `time_seconds` | `[57848]` | float64 | 审计网格；不入模 |

## 全量缓存结果

- 身份：`792 calls / 111 cats`；`792/792` 源文件哈希通过。
- 总帧数：`57,848`。
- 每个叫声帧长 `min/median/max = 9/70/443`。
- 最长序列：`0Y-8week-046A-45.wav`，443 帧，对应网格终点 4.42 秒。
- `log_f0` 与 `periodicity` 有限率均为 `0.8165537270`；缺失 `10,612` 帧。
- `voiced_probability`、`log_rms`、`spectral_tilt`、`spectral_flatness` 有限率均为 `1.0`。
- 20 个叫声没有任何有效 F0；这些样本仍完整保留其原时间网格、voicing probability 与三个全帧谱/能量通道，没有被删除。
- 提取耗时 `351.77 s`，只使用 CPU；标签与实验成绩未参与提取。

抽样重建审计覆盖 44 个调用，包括全部 20 个无有效 F0 调用、最长序列和均匀分布的其他调用。由六通道轨迹重建 IDEA-068 的旧 20 维统计后，NaN 模式逐位一致，最大绝对差 `2.8610e-6`，远小于预注册的 `atol=rtol=2e-4`，结论 PASS。

完整测试为 `6/6 PASS`；独立 `--verify-existing` 再加载验证也通过。测试覆盖锁定依赖、真实调用的 10 ms 网格、静音输入、全帧通道、旧统计重建、schema 和最终缓存结构。

## 产物与哈希

| 产物 | SHA-256 |
|---|---|
| `configs/protocol/meowagenet_idea085_trajectory_extraction_v1.json` | `351c3e115dec997712e3197a557a3b8240157db3559d2be4989ac1492564c69e` |
| `scripts/extract_idea085_acoustic_trajectories.py` | `c2ae23b65afe6c4086fcd62271d67312573437e6d8b8ebbcd3c1ee048177df97` |
| `tests/test_idea085_acoustic_trajectories.py` | `2172a6ce5a887ed359244e5f79284388f1e3e73ce4e0dd70aa4481f2d944401c` |
| `runs/meowagenet_idea085_acoustic_temporal_residual_v1/features/acoustic_trajectories.npz` | `a69b246eba8d773f12a605f2caa9086e2ba0a2d87a3eeeb6ec57175c551a46a7` |
| `runs/meowagenet_idea085_acoustic_temporal_residual_v1/features/trajectory_schema.json` | `d8a53f861bc338b8394b40bc36c7874b948fc6646a684d31d7686e9a0b88f5c9` |
| `runs/meowagenet_idea085_acoustic_temporal_residual_v1/features/extraction_summary.json` | `d7aa9c9d1e76d62fdb6e5f4e1f1accac775386bd3a4e1587fdc7d049d6431ac1` |

## 下一步边界

该缓存只完成了共享、标签盲的输入材料。它不证明时序模型有效，也不授权 GPU。下一步必须先只读审查固定 A0/C1/T1/J1 方案的 CPU 预检，特别确认 T1/J1 的参数与初态配对、J1 仅做整帧顺序打乱、train-only 预处理、padding/mask 安全、outer-test=false 与首 cell/resume 门禁；之后再由研究总监决定是否运行。
