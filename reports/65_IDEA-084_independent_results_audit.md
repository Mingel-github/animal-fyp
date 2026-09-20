# IDEA-084 全量结果独立复算审计

日期：2026-09-19  
审计角色：牛马1（独立只读复核）  
结论：**PASS；官方汇总与独立复算在 `1e-12` 容差内零差异。三条预注册综合 gate 均未通过，但各轴结果应分别解释，不能把 gate fail 简化为“没有作用”。**

## 独立性与完整性

本审计没有导入或调用 IDEA-084 runner/aggregate。复算脚本逐项读取 144 个 `fit_summary.json`、288 个保存的 validation prediction CSV，验证每个预测文件的 SHA-256，并由调用级概率重新生成动物级概率与预测，再从动物级记录独立计算所有指标。

- 144/144 fit identities 完整、无重复，覆盖 `4 pipelines × 3 base seeds × 3 repeats × 4 folds`；full seed 逐格符合锁定公式。
- 288/288 prediction hashes 与 fit ledger 一致。
- 144/144 动物级预测可由调用级概率均值重建；概率有限、归一且保存的类别等于 argmax。
- 每格 validation 动物与锁定 role 表一致；四管线的动物 ID 与真值标签逐格配对。
- protocol 与 runner 哈希仍分别为 `0c4b50f2a6903bb5e55157591e5a93400c56602161fa2b1f8c8fd0d112bb7eef`、`fa3913fd7b2a7a66f3a4e61deb150ef7358cc32ee5b65eafdfe9d2c7e2abe12f`；outer test 未访问。
- 独立结果与官方 `initial_evaluation_summary.json` 的核心均值、逐 fold、9 个 seed×repeat、12 个 repeat×fold split、三组纠错剖面和 gate 条件全部一致；difference count=`0`。

## 四管线均值

以下为 9 个等权 seed×repeat 估计的均值：

| 管线 | Macro-F1 | Balanced accuracy | Animal CE | Brier |
|---|---:|---:|---:|---:|
| A0 | 0.727352 | 0.761111 | 0.699483 | 0.420760 |
| C1 | 0.729273 | 0.762963 | 0.697506 | 0.417705 |
| P1 | 0.726634 | 0.766667 | 0.698193 | 0.418447 |
| R1 | 0.733229 | 0.771296 | 0.701961 | 0.420721 |

## 三条预注册比较

### P1 − C1：不支持方法改进

- Macro-F1 均值差 `-0.002639`，9 个 seed×repeat 为 `4/0/5` 正/平/负；三个 base-seed 均值仅一个为正。
- Balanced accuracy 数值上提高 `+0.003704`，但 CE 数值上变差 `0.000687`，Brier 数值上变差 `0.000742`。
- 12 个 split 中 `6/12` 非负，最差为 repeat 2 / fold 0 的 `-0.036159`。
- senior recall 的三个 base-seed 差为 `-0.016667/-0.033333/+0.066667`，其中一个超过预注册安全下界。
- 612 个重复动物出现中，P1 纠正 C1 错误 14 次、引入 15 次新错误，净值 `-1`；9 个 seed×repeat 的净纠错方向为 `4/1/4`。

P1 的 BA 有小幅正向变化，但分类主指标、概率质量、尾部稳定性与 senior 安全性没有形成一致改进，因此 P1−C1 gate fail。

### P1 − A0：分类近乎持平，保留概率质量与 BA 的部分效用

- Macro-F1 均值差 `-0.000718`，方向为 `3/3/3`；这不是预注册的 F1 提升。
- Balanced accuracy 数值上提高 `+0.005556`；CE 数值上降低 `0.001291`，Brier 数值上降低 `0.002313`。
- 12 个 split 中 `8/12` 非负，最差为 repeat 0 / fold 2 的 `-0.028042`，均满足对应的 split 数量与尾部门槛。
- senior recall 三个 base-seed 差为 `+0.033333/0/+0.016667`，安全轴通过。
- P1 纠正 A0 错误 8 次、引入 9 次，净值 `-1`；净纠错方向为 `1/5/3`。

因此 P1−A0 综合 gate 仍 fail，主要由平均 F1 与正向 seed×repeat 数不足导致；但 CE、Brier、BA、senior 和 split 安全轴的正向数值应作为明确的部分效用保留，而不是写成“无效”。

### P1 − R1：不支持真实 15/5 分组的特异价值

- Macro-F1 均值差 `-0.006595`，方向为 `1/2/6`；三个 base-seed 均值全为负。
- Balanced accuracy 数值上降低 `0.004630`；但 CE 数值上降低 `0.003768`，Brier 数值上降低 `0.002274`。
- 12 个 split 中 `7/12` 非负，最差为 repeat 1 / fold 1 的 `-0.061250`。
- senior recall 三个 base-seed 差为 `-0.016667/-0.016667/+0.016667`，均在安全下界内。
- P1 纠正 R1 错误 6 次、引入 10 次，净值 `-4`；净纠错方向为 `2/2/5`。

R1 在分类 F1/BA 上更好，而 P1 在 CE/Brier 上更好，构成分类与概率质量的方向分离。固定真实分组没有优于同容量固定随机分组，故不能提出该 15/5 分组的机制特异性主张；单个 R1 本身也不能构成完整因果否定。

## 最终判断

三条预注册比较均未通过完整 gate。IDEA-084 不应升级为优于 C1 的新主方法，也没有证据支持真实 15/5 分组相对固定随机分组的特异优势。与此同时，P1 相对 A0 的 BA、CE、Brier 与 senior 安全轴，以及 P1 相对 R1 的 CE/Brier 优势，属于真实、已复算的数值模式，可作为后续假设材料；不得用本次结果后追加权重、宽度、分组、seed 或门槛搜索。

本实验仍基于同一 111 只猫的重复划分，不是外部确认；所有 fold、split 与重复动物出现均为描述性单位。

## 审计产物与哈希

| 产物 | SHA-256 |
|---|---|
| `scripts/verify_idea084_grouped_dual_branch_results.py` | `649ca0dcdbdc20a2e9265de8b8edc810fb87f5857f434e8013422f4060cd603a` |
| `runs/meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1/run_manifest.json` | `f773401e39778a1207020201acba4a1bea5c8228ed951c2b9deab551a9478c37` |
| `runs/meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1/initial_evaluation_summary.json` | `fb4dea0b4b452561a800b6cb4c9f3a7c29ac0f1f79a74bd67c1b3c674092cd46` |
| `runs/meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1/run_summary.json` | `6bc04fe30d4facbfec09868c22da96ad6660b1fb35f3c2a6307d60cf7a684429` |
| `runs/meowagenet_idea084_grouped_dual_branch_acoustic_residual_v1/independent_results_audit.json` | `8ebe20a875a6f3b766141551d6914820e5994a101703d2edf2ce696324c82b6a` |
