# IDEA-088 clean111 原 baseline 风格对照：最终结果

日期：2026-09-20  
状态：**COMPLETE / PASS_FULL**；60/60 fits 完成，独立全量复核通过，不追加实验

## 1. 结论摘要

在 IDEA-088 锁定的 clean111、原 post-swap 角色投影和原 final 风格训练配方下，公共猫级 probability-mean Macro-F1 的五个 seed 均值为：

- VGGish：`0.698972 ± 0.024127`；
- A0（冻结 AST 特征加分类头）：`0.709418 ± 0.018780`；
- C1（在 A0 基础上加入有界声学残差）：`0.722263 ± 0.024837`。

C1 相对 A0 的配对 seed 差为 `+0.012846 ± 0.026330`，五个 seed 中 3 正、0 平、2 负。本轮没有新增判定门槛。结合差异幅度较小且跨 seed 不一致，本轮结果只能作为**协议敏感性下的描述性证据**，不能据此宣称稳定或普适改进。

A0 与 C1 的 native call Macro-F1 几乎相同，C1−A0 为 `-0.000234 ± 0.007601`。同时 C1 的猫级 CE 与 Brier 略高于 A0。因而 C1 的猫级 Macro-F1 增量主要体现为本协议下 call 概率经逐猫平均后的分类结果变化，不是 call 级性能提升，也不是概率质量/概率误差全面改善。

本轮整体分数没有超过 IDEA-087，但两轮的角色、强制交换、停止规则、输入单位和训练配方不同，绝对分数只能作背景性描述，不能作单因素因果比较。IDEA-088 不新增动物，也不覆盖 IDEA-087 的结论。

## 2. 锁定设计与评价边界

- 数据为 792 个去重 calls、111 个 analysis cats；重复别名 `049A` 被排除。
- 三条固定 pipeline 为 VGGish、A0、C1；5 seeds × 4 folds × 3 pipelines，共 60 fits。
- 原 post-swap 角色由历史运行日志恢复并冻结；runner 不重新运行 SGKF。
- VGGish 使用 129 维输入，即 128 个 embedding 列加 `mean_freq`；native unit 是 embedding row。
- A0/C1 使用 AST call 输入；native unit 是 call。两者共享公共 head 初态、call 顺序和 dropout 随机流。
- 训练固定使用 Adamax、学习率 `0.003109800273709165`、dropout `0.44571035356880917`、batch 128、最多 1500 epochs、training-loss early stopping、`min_delta=0.001`、patience 30、恢复最佳权重；无 validation、HPO、gradient clip 或 AMP。
- 主指标先在每折按猫平均 softmax 概率，再计算猫级指标；每个 seed 先对四折等权平均，最终报告五个 seed-level 均值的均值与样本 SD。
- VGGish row 指标与 AST call 指标分别报告，不把不同观测单位混作同一指标。
- `000A`、`046A` 在所有 cell 中强制留在训练侧；每个 seed 另有两只替换猫测试两次。因此结果**不是 111 猫完整 OOF**，也没有把重复测试猫或 20 folds 当作新增独立动物。

## 3. 公共猫级结果

下表均为“五个 seed 的四折均值”的均值 ± 样本 SD。

| Pipeline | Accuracy | Macro-F1 | Balanced Accuracy | Cross-Entropy | Brier |
|---|---:|---:|---:|---:|---:|
| VGGish | 0.716313 ± 0.025805 | 0.698972 ± 0.024127 | 0.724856 ± 0.010231 | 0.674348 ± 0.007940 | 0.403162 ± 0.009754 |
| A0 | 0.715956 ± 0.011698 | 0.709418 ± 0.018780 | 0.724996 ± 0.031248 | 0.882599 ± 0.050774 | 0.419421 ± 0.009457 |
| C1 | **0.728476 ± 0.026093** | **0.722263 ± 0.024837** | **0.736580 ± 0.034808** | 0.891945 ± 0.056662 | 0.419657 ± 0.011777 |

VGGish 的 CE/Brier 数值优于 A0/C1，但 VGGish 的原生输入是 embedding rows，AST 的原生输入是 calls；这组概率误差指标只在锁定的公共猫级聚合口径下描述，不应外推为模型家族的普适排序。

## 4. 原生输入单位结果

| Pipeline | Native unit | Native-unit Macro-F1 |
|---|---|---:|
| VGGish | embedding row | 0.704045 ± 0.012359 |
| A0 | AST call | 0.666278 ± 0.024984 |
| C1 | AST call | 0.666044 ± 0.025529 |

本轮VGGish特征行级Macro-F1为0.704045，接近历史复现的0.700136。两者均使用129维输入（128维embedding加mean_freq），并以各折F1取平均。本轮从历史日志保留原实际换猫后的划分，仅删除清洗排除的049A；清洗投影、模型随机初始化及类别整数编码等实现设置存在差异。因此历史值用于背景参照，本轮数值按清洗版实验单独报告。

## 5. 五个 seed 的猫级 Macro-F1 与配对差

| Split seed | VGGish | A0 | C1 | C1−A0 猫级 | C1−A0 native call |
|---:|---:|---:|---:|---:|---:|
| 7270 | 0.682902 | 0.698012 | 0.738168 | +0.040155 | -0.005261 |
| 860 | 0.735319 | 0.685483 | 0.717403 | +0.031919 | -0.011045 |
| 5390 | 0.706888 | 0.707257 | 0.682746 | -0.024511 | +0.004136 |
| 5191 | 0.696896 | 0.728247 | 0.725921 | -0.002327 | +0.006931 |
| 5734 | 0.672855 | 0.728088 | 0.747080 | +0.018992 | +0.004071 |
| 均值 ± SD | — | — | — | **+0.012846 ± 0.026330** | **-0.000234 ± 0.007601** |

猫级和 call 级配对差均是 3 正、0 平、2 负，但 call 级均值约为零。该模式支持“C1 的可见差异发生在逐猫概率聚合后的决策层面”这一有限描述，不支持稳定的逐 call 优势。

## 6. 训练与门禁完整性

| Pipeline | Fits | Best epoch：均值（范围） | Stop epoch：均值（范围） |
|---|---:|---:|---:|
| VGGish | 20 | 196.75（107–307） | 226.75（137–337） |
| A0 | 20 | 89.70（67–138） | 119.70（97–168） |
| C1 | 20 | 88.65（67–155） | 118.65（97–185） |

本地最终核验确认：

- 60/60 `fit_summary.json` 均为 `complete`，三条 pipeline 各 20 fits；
- 每个 fit 的 weights、train-only preprocessing、epoch orders、checkpoint lock、unit predictions 和 cat predictions 共 360 个引用产物，文件存在且 SHA-256 与记录一致；
- 60/60 checkpoint 复载 state digest 一致，且每个 fit 的 test prediction 调用数严格为 1；
- 全部 fit 均在记录的 best epoch 后严格 30 epochs 停止；
- 最终 `--resume` 门禁返回 `completed_new_fits=0`、`validated_resume_fits=60`，没有追加训练；
- IDEA-088 原 runner/角色两组测试为 23 passed；加入独立 verifier 的 6 项测试后，三组最终为 **29 passed**。

独立 verifier 未导入训练 runner，重新核验角色/标签、train-only 预处理与类别权重、每一轮 unit order、严格早停、checkpoint-before-test、unit→cat 重算及全部 seed 汇总：

- 状态：`PASS_FULL`；
- fit 指标最大绝对差：`5.39e-9`；
- aggregate 最大绝对差：`1.83e-9`；
- metadata 最大绝对差：`0`；
- 20/20 A0/C1 公共初态和 epoch order 前缀匹配；
- 3965 个历史保护文件的 combined digest 与执行前快照精确一致。

## 7. 产物与冻结身份

| 产物 | 路径 | SHA-256 |
|---|---|---|
| Protocol | `configs/protocol/meowagenet_idea088_original_style_clean_v1.json` | `c12bd2a5a4abe6b4f0af425a153747a7d6c96b6c9214023684ffc33625d21847` |
| Runner | `scripts/run_meowagenet_idea088_original_style_clean.py` | `f1cd6c9993b505e3e3f5f6d19488ef1b755c55eec8c63ffb68b4918e8d0db5cc` |
| Recovered roles | `runs/meowagenet_idea088_original_style_clean_v1/roles/original_clean_roles.json` | `a0ce4989985989ebdd982bbf565e33c080b4e314e232eb875f30fbdb4eac34b8` |
| CPU preflight | `runs/meowagenet_idea088_original_style_clean_v1/preflight/cpu_preflight.json` | `fb4fa4b55762bed639528387858b7f417f452083cde7e778792a40c4392eb449` |
| Aggregate results | `runs/meowagenet_idea088_original_style_clean_v1/idea088_aggregate_results.json` | `746e755a3c22ea1fc721348aef2315eb3b8f3fb1265a69797512101a82e1bd68` |
| Results metadata | `metadata/experiments/meowagenet_idea088_original_style_clean_v1_results.json` | `50d99e2c235460b7f3ed4a06cea31c3aa5892696a87ffe98fc7345265407b21c` |
| Independent audit report | `reports/81_IDEA-088_original_style_clean_independent_audit.md` | `3c4e1f8c9381d9324fa402ee4ede2860bf5853ed1692a25e9eed92e95087a091` |
| Independent verification | `runs/meowagenet_idea088_original_style_clean_v1/independent_verification.json` | `8b005ca520c4ae0edf3911bfd4bf796b8f66c00889fa77c4444e2081c59e1294` |

## 8. 最终决定

IDEA-088 到此关闭：保留 C1 在本锁定协议下相对 A0 的小幅猫级 Macro-F1 正差，连同跨 seed 不一致、native call 近零差及 CE/Brier 略差一并报告。该结果作为协议敏感性证据归档，不替换 IDEA-087，不提出外部泛化、完整 OOF、更多训练数据或全面概率质量改善的主张，也不据此启动额外模型、seed、调参或数据扩展。
