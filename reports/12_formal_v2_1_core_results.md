# MeowAgeNet Formal-v2.1 核心正式实验报告

> 日期：2026-08-28
> Run ID：`meowagenet_formal_v2_1_core`
> 状态：**锁定范围内的 108 个 fold-level fits 已全部完成**
> 证据边界：同一批 111 只分析猫上的三次 repeated、四折 cat-ID-disjoint 内部验证

## 1. 核心结论

Formal-v2.1 给出了两层清晰结论：

1. **性能层面（H048）**：Probe-guided AST adapter 相对唯一 VGGish baseline 的
   animal macro F1 平均提高 **0.0765**，9/9 个 repeat×seed 完整 OOF 比较全部为正。
   Balanced accuracy 平均提高 **0.0894**，QWK 平均提高 **0.1039**。AST 路线的性能
   优势在本轮重复内部验证中方向一致、幅度具有实际意义。
2. **机制层面（H019）**：Probe-guided adapter 相对 matched AST head-only 的 macro
   F1 平均提高 **0.0052**，9 次比较中 5 次为正。两者在 primary metric 上基本并列，
   说明本轮正式提升主要来自 AST 表征与统一分类流程；adapter 提供了最高平均 macro
   F1，但其独立增量具有 split-dependent 特征。

因此，当前论文可以稳定地表述为：

> 在锁定的 repeated animal-level internal validation 中，AST pipeline 相对复现的
> VGGish baseline 获得了持续的性能提升。Probe-guided adapter 得到最高平均 macro
> F1，而 matched head-only AST 在 balanced accuracy 及部分年龄类别召回率上更强；
> adapter 的独立机制优势仍小于 AST backbone 路线本身的贡献。

这份结果是阶段性正式证据，不会关闭后续新模型、新 idea 或外部验证路线。

## 2. 锁定执行范围

正式运行严格使用 execution lock 中共同确认的最小核心：

- pipelines：`vggish_mlp`、`ast_head_only`、`ast_probe_guided_adapter`；
- split repeats：0、1、2；
- outer folds：0、1、2、3；
- base model seeds：17、43、101；
- optional modules：空；
- 总量：3 pipelines × 3 repeats × 4 folds × 3 seeds = **108 fits**；
- primary metric：111-cat complete OOF 上的 animal macro F1；
- primary contrasts：H048（adapter−VGGish）和 H019（adapter−head-only）。

运行从 2026-08-28 12:40 开始，于 13:10 完成，整体约 30 分钟。AST 计算使用
NVIDIA GeForce RTX 4060 Ti；软件、硬件、runner 与 recipe 哈希均由 execution lock
记录。

## 3. 完整 OOF 汇总

每个 pipeline 形成 3 repeats × 3 model seeds = 9 套完整 OOF；每套 OOF 恰好包含
111 个唯一 cat_id，类别分布为 kitten 15、adult 62、senior 34。

| Pipeline | Macro F1，均值 ± SD | 范围 | Balanced accuracy，均值 ± SD | QWK，均值 ± SD |
|---|---:|---:|---:|---:|
| VGGish + MLP | 0.6525 ± 0.0462 | 0.5980–0.7184 | 0.6525 ± 0.0493 | 0.5334 ± 0.0597 |
| AST head-only | 0.7238 ± 0.0335 | 0.6629–0.7616 | **0.7597 ± 0.0247** | **0.6374 ± 0.0387** |
| Probe-guided AST adapter | **0.7290 ± 0.0428** | 0.6633–0.7846 | 0.7419 ± 0.0482 | 0.6373 ± 0.0569 |

表格传达了三个互补信息：

- **Macro F1** 对三个年龄类别等权，adapter 的均值最高，说明它取得了本轮最好的
  类别整体折中。
- **Balanced accuracy** 是三个类别召回率的平均值，head-only 最高，说明它对每个
  类别“找回真实样本”的平均能力最强。
- **QWK** 对年龄类别之间的距离进行加权，错成相邻年龄组的惩罚小于跨级错误；
  head-only 与 adapter 几乎相同，说明两种 AST 方案在年龄顺序一致性方面不相上下。

## 4. 两项锁定 primary contrasts

| Hypothesis | Paired comparison | Macro F1 平均差 | 正向 OOF | 10,000 次 hierarchical paired bootstrap 95% CI |
|---|---|---:|---:|---:|
| H048 | Adapter − VGGish | **+0.0765** | **9/9** | −0.0064 到 +0.1685 |
| H019 | Adapter − AST head-only | +0.0052 | 5/9 | −0.0458 到 +0.0585 |

### H048：性能改进假设

9 个 paired OOF 差值全部为正，范围为 +0.0133 到 +0.1539。这个模式说明增益并非由
单个 lucky seed 产生；在每个 repeat×seed 单元中，adapter 都超过对应的 VGGish。
平均差 +0.0765 也明显高于协议中的实际意义参考幅度 0.03。

层级 paired bootstrap 区间的下界略低于 0，体现当前证据仍来自同一批 111 只猫和
三个 split repeats。合适的结论是：**本轮观察到方向完全一致、幅度较大的内部验证
优势，同时保留总体效应大小的不确定性。** 9/9 正向结果与多项 secondary metrics
共同构成 H048 的强支持信号。

### H019：adapter 独立贡献假设

Adapter−head-only 的平均差为 +0.0052，正向次数为 5/9；不同 repeat 的平均差分别
约为 +0.0437、−0.0102、−0.0179。Adapter 在 repeat 0 明显领先，在 repeats 1 和 2
与 head-only 互有胜负。因此，本轮适合把 probe-guided adapter 保留为**最高平均
macro F1 的正式候选**，同时把 H019 解释为：adapter 的额外收益具有数据划分依赖性，
AST backbone 与 matched classifier head 已经提供了主要性能增益。

## 5. Secondary metrics 与类别优势

| Paired comparison | Δ Macro F1 | Δ Balanced accuracy | Δ QWK |
|---|---:|---:|---:|
| Adapter − VGGish | **+0.0765** | **+0.0894** | **+0.1039** |
| Head-only − VGGish | **+0.0713** | **+0.1072** | **+0.1040** |
| Adapter − Head-only | +0.0052 | −0.0179 | −0.0001 |

Adapter 相对 VGGish 的 balanced accuracy 在 9/9 个 OOF 中为正，QWK 在 8/9 中为
正；head-only 相对 VGGish 的三项指标也都呈现稳定优势。两种 AST pipeline 共同支持
AST 路线的性能结论。

三个类别的平均 recall 为：

| Pipeline | Kitten recall | Adult recall | Senior recall |
|---|---:|---:|---:|
| VGGish + MLP | 0.6296 | 0.6971 | 0.6307 |
| AST head-only | **0.8815** | 0.6559 | **0.7418** |
| Probe-guided AST adapter | 0.8222 | **0.7007** | 0.7026 |

这说明不同 AST 模块具有各自优势：head-only 更擅长识别 kitten 和 senior，adapter
在 adult 上取得最高召回，并以更均衡的 precision/recall 组合获得最高 macro F1。
模型并非简单地“一个全面压倒另一个”，而是围绕相同强势 AST 表征形成不同的类别
决策取向。

## 6. Repeat 与 pilot 对照

| Pipeline | Repeat 0 Macro F1 | Repeat 1 | Repeat 2 |
|---|---:|---:|---:|
| VGGish + MLP | 0.7040 | 0.6248 | 0.6286 |
| AST head-only | 0.7198 | **0.7543** | **0.6972** |
| Probe-guided AST adapter | **0.7635** | 0.7441 | 0.6793 |

Pilot 中的 macro F1 为 VGGish 0.6846、head-only 0.7260、adapter 0.7575；formal
均值分别为 0.6525、0.7238、0.7290。绝对分数随 repeated splits 改变，但最关键的
相对关系具有很强连续性：

- pilot 的 adapter−VGGish 差约为 +0.0729；
- formal 的平均差为 +0.0765；
- 两阶段的相对提升幅度非常接近；
- pilot 的 adapter−head-only 差为 +0.0315，formal 缩小到 +0.0052，说明单次 pilot
  高估了 adapter 相对 matched AST control 的独立增量。

Formal-v2.1 因而保留了 pilot 中最重要的性能信号，同时校准了机制归因。

## 7. Probe-guided layer 选择

12 个 repeat×fold probe 选择的层（1-based）为：

| Repeat | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|---|
| 0 | 8 + 11 | 8 + 11 | 8 + 11 | 10 + 12 |
| 1 | 11 + 12 | 10 + 12 | 11 + 12 | 11 + 12 |
| 2 | 5 + 7 | 8 + 11 | 6 + 7 | 11 + 12 |

第 11 层出现 8 次、第 12 层出现 6 次，后部层占主导；repeat 2 同时选择了第 5–7
层，说明最佳 placement 会随训练猫和验证猫的组成而变化。这个结果支持“probe 能做
fold-specific placement selection”，同时也解释了 adapter 增益的 split dependence。

## 8. 完整性与可追溯性审计

独立复算确认：

- 108/108 fit summaries 完整，失败数为 0；
- 27/27 complete OOF prediction files 完整；
- 每套 OOF 均为 111 个唯一 cat_id，无重复或缺失；
- 每套标签分布均为 15/62/34；
- 所有概率为有限数，最大行概率和误差为 `1.4e-7`；
- 从 OOF CSV 重新计算的 macro F1、balanced accuracy 和 QWK 与
  `formal_summary.json` 完全一致；
- `run_manifest.json` SHA256：
  `68a73d9cc8ca6b24006695bffc177da50756c092129cee3fb22c9ebd78226586`；
- `run_summary.json` SHA256：
  `55f4be6f10ddbe78225ddb94daacb84cc310dbf08c819c7ee1d39c6b8334566b`；
- `formal_summary.json` SHA256：
  `d0eba94ef7ca1ae9089e80a96d0906836f33dc51306b7a5b357333983bbf30c1`。

原始运行产物继续保存在本机忽略目录 `runs/meowagenet_formal_v2_1_core/`；Git 中保存
本报告和机器可读审计快照，既避免提交逐样本预测，又能通过哈希确认本报告对应的原始
正式结果。

## 9. 论文使用建议

论文主表适合同时保留 VGGish、AST head-only 和 Probe-guided adapter：

- VGGish 是复现基线；
- head-only 分离 AST 表征收益；
- adapter 展示当前最高 macro F1 候选及其有限的增量。

建议把 H048 写为本阶段得到支持的主要性能发现，把 H019 写为经过正式重复实验校准的
机制发现。后续工作可以继续尝试 adapter 设计、新 pooling、新 pretrained encoder 或
外部数据验证；新增结果使用新的 run/protocol 标识，不覆盖本次 formal-v2.1 记录。
