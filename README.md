# Animal Vocalization FYP

This repository contains the code, configuration, experiment metadata, and
reports for an animal-vocalization final-year project.

The first reproduction target is van Toor et al.'s feline age-prediction
pipeline. The authors' repository is kept as an upstream Git submodule under
`src/baselines/feline-age-prediction`; its code is not vendored as original
project work.

## 当前研究入口 / Current research overview

更新日期：2026-09-21。当前阶段是 **MeowAgeNet 猫年龄分类的声学残差融合研究与论文整理**。
统一核心对照已完成并通过独立复核；最新主结果来自 IDEA-089。
本研究版本位于
[`research/ast-acoustic-idea065-088-20260920`](https://github.com/Mingel-github/animal-fyp/tree/research/ast-acoustic-idea065-088-20260920)
分支；分支名保留了早期归档范围，该分支现已包含 IDEA-089。

研究问题是：**显式声学描述量如何补充冻结 Audio Spectrogram Transformer（AST）的表示，
以及不同融合方式带来怎样的分类收益与波动？** 当前方法在冻结 AST 之后的分类隐藏表示处
引入声学分支。AST 骨干保持冻结，可训练部分是分类／融合头。早期的 Probe-guided AST
adapter 属于历史探索，其结果单独保留。

### 最新结果

下面是同一内部协议下的六条完整流程。数值为 9 份完整猫级评估的均值 ± 样本标准差，
单位为百分数；Macro-F1 对三个年龄类别的 F1 等权平均，Accuracy 表示猫级正确率。

| 方法名称 | Macro-F1 (%) | Accuracy (%) |
| --- | ---: | ---: |
| VGGish 分类基线 | 66.37 ± 4.32 | 67.07 ± 3.63 |
| VGGish 与作者频率标量融合 | 68.82 ± 4.50 | 70.07 ± 3.95 |
| 冻结 AST 分类基线 | 72.87 ± 3.63 | 72.17 ± 3.81 |
| AST 与声学特征直接拼接 | 73.30 ± 3.35 | 72.47 ± 3.28 |
| 加性声学残差融合 | **74.47 ± 3.79** | **73.87 ± 3.93** |
| 幅度受限的声学残差融合 | 73.49 ± 4.73 | 72.97 ± 4.64 |

加性声学残差融合把 20 项声学描述量经过 `20→60→128` 的非线性分支，得到对 AST
分类隐藏表示的加性修正；幅度受限版本使用相同分支，并以主表示的 RMS 尺度和 `tanh`
限制修正量。直接拼接则把同一组声学描述量与 AST 表示拼接后送入分类头。

相对于冻结 AST 分类基线，加性声学残差融合的平均 Macro-F1 提高 **1.60 个百分点**，
9 组配对中 7 组为正；幅度受限版本提高 **0.62 个百分点**，5 组为正。两者的描述性
配对区间均跨过零，当前证据体现为正向平均收益及对划分／训练随机性的敏感性。
加性版本本轮平均分类表现最高；相对加性版本，幅度受限版本的平均交叉熵略低，F1 波动更大。
完整正负结果及各年龄组表现见[最新中文结果报告](reports/87_IDEA-089_unified_core_results.md)。

### 评估口径与证据范围

- 数据包含 111 只分析猫。AST 使用清洗后的 792 条去重叫声；VGGish 使用作者表示表
  按清洗后的猫 ID 保留的 936 条记录。二者在猫身份和标签上对齐，VGG 行与 AST 叫声的
  逐条对应关系未完整恢复。
  因此跨家族结果是完整流程比较，AST 家族内的融合对照使用相同输入缓存。
- 沿用 3 组既有猫隔离划分，每组 4 折，训练种子为 17、43、101。每折训练、验证和
  测试猫互不重叠。训练侧用验证猫级交叉熵确定轮数，再从头在训练加验证数据上按固定轮数
  重训，锁定模型后进行测试。两模型家族保留各自训练配方，具体差异在报告中披露。
- 每只猫的类别概率由其输入单位的概率取算术平均。每组划分／种子先合并 4 个测试折，
  得到 111 猫各出现一次的完整折外预测（OOF），再计算指标。主表汇总这 9 份完整结果。
- 六条流程共完成 216 次选轮训练和 216 次固定轮数重训。每条流程的 999 次猫预测出现
  来自同一批 111 只猫；独立动物数是 111。该数据也参与过历史开发，本轮定位为统一内部验证。
- 作者频率列 `mean_freq` 的代码线索指向平均基频；确切生成链及逐行音频映射仍有缺口。
  正文使用“作者提供的频率标量”这一可核实描述，并与本项目的 20 维声学特征区分。

### 新协作者与论文写作者的阅读顺序

1. [统一对照结果与解释](reports/87_IDEA-089_unified_core_results.md)：正文主表、模型定义、
   训练配方、波动、历史协议差异和相关文献入口。
2. [基线与协议设计核查](reports/84_IDEA-089_baseline_protocol_design_audit.md)和
   [独立复核](reports/86_IDEA-089_independent_engineering_audit.md)：数据来源、猫角色、
   特征对应关系及独立结果复算。
3. [机器可读结果](metadata/experiments/meowagenet_idea089_unified_core_v1_results.json)、
   [冻结协议](configs/protocol/meowagenet_idea089_unified_core_v1.json)及
   [执行代码](scripts/run_meowagenet_idea089_unified_core.py)：核对数值、实现和训练规则。
   若报告、配置或代码存在疑问，列明具体差异再核实。
4. [历史 v3 汇总](reports/58_Formal_Research_Summary_v3.md)及其引用的专题报告：用于
   交代既往探索。v2.1、v3、调参阶段、原作者风格评估和本轮结果分别保留协议标签。

论文标题、摘要、正文和主表使用**描述性方法名称**；内部实验代号仅用于代码与记录追溯，
不作为论文方法名称。确有必要使用正式缩写时，先定义完整名称并保持全文一致。
下方映射专供复现者定位既有代码，不代表已经确定的论文品牌名，也不改动冻结文件中的键名。

<details>
<summary>复现附表：当前方法名称与 IDEA-089 内部标识的映射</summary>

| 方法名称 / Descriptive name | IDEA-089 内部别名 | 结果 JSON / 代码标识 |
| --- | --- | --- |
| VGGish 分类基线 / VGGish classifier | VGG128 | `VGG128_no_f0` |
| VGGish 与作者频率标量融合 / VGGish with author-provided frequency scalar | VGG129 | `VGG129_with_f0` |
| 冻结 AST 分类基线 / Frozen AST classifier | A0 | `A0_ast_only` |
| AST 与声学特征直接拼接 / AST–acoustic feature concatenation | D0 | `D0_direct_concat` |
| 加性声学残差融合 / Additive acoustic residual fusion | U1 | `U1_wide_unbounded_additive` |
| 幅度受限的声学残差融合 / Bounded acoustic residual fusion | C1 | `C1_bounded_wide_additive` |

此映射只适用于 IDEA-089。不同历史实验可能复用同一代号表示不同结构，应按实验编号
和具体定义辨认。例如早期猫权重或局部适配对照中的 `C1` 与当前幅度受限声学残差不同。
内部键名中的 `f0` 是既有命名，其物理来源表述以上方证据范围为准。

</details>

### 写作与交接约定

- 正文以当前统一协议的完整结果为主，历史探索作为单独标注的背景或补充材料。
- 同时呈现平均收益、配对差、标准差和不确定性区间；SD描述波动，不能直接解释为偶然概率。
- 方法位于冻结 AST 后的融合／分类头；两条分支来自同一音频的不同表征。
  声学生理解释作为研究动机或待验证假设，分类结果与机制归因分别论述。
- 以 MeowAgeNet 猫年龄任务为核心；其他猫情境任务、狗实验和新动物验证具有各自的
  研究问题与优先级。引用文献前核对原文，区分已有思想与本项目的具体实现贡献。
- GitHub 提供报告、代码、协议和紧凑结果。原始音频、模型权重及逐猫预测保留在本地；
  新增逐猫分析或图表需要相应数据，缺失材料列入待确认项。
- 先整理完整初稿和证据对应关系。补充实验另行明确问题与固定预算，历史审计报告和
  已冻结结果保持原样。

## Setup

Clone the project together with the baseline submodule:

```powershell
git clone --recurse-submodules <project-repository-url>
cd <project-directory>
```

Create the reproduction environment with Conda or Miniforge:

```powershell
conda env create `
  --prefix .\environment\.conda\meowagenet-repro `
  --file .\environment\meowagenet-repro.yml
```

The environment directory is intentionally excluded from Git. Recreate it
from the YAML file instead of copying local binaries.

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/` | Data, model, and experiment configuration |
| `data/` | Local datasets; ignored until licensing and privacy are reviewed |
| `environment/` | Reproducible environment definitions |
| `metadata/` | Dataset and experiment metadata suitable for version control |
| `notebooks/` | Project-owned exploratory notebooks |
| `plan/` | Idea handoffs, experiment plans, protocol narratives, and amendments |
| `reports/` | Completed results and conclusions; also immutable legacy-path copies required by historical audits |
| `runs/` | Generated outputs; selected formal JSON audit logs are versioned |
| `scripts/` | Command-line entry points and utilities |
| `splits/` | Reproducible train/validation/test split definitions |
| `src/` | Project source and external baseline references |
| `tests/` | Automated tests |

## Data policy

Keep raw audio, personal data, credentials, trained checkpoints, and per-sample
predictions in the local research environment. Version compact formal audit
logs, acquisition instructions, licenses, checksums, and preprocessing steps so
collaborators can reconstruct and verify permitted results.

The initial baseline assessment is in `reports/00_baseline_assessment.md`.

## Historical research archive / 历史研究记录

以下为早期研究阶段的原始记录，保留当时的结果和决策。原文中的 `current`、`active`
和 `next` 均指记录当时的状态；当前主线和写作入口以上方概览为准。历史代号按对应
实验解释，跨协议分数分别呈现。

<details>
<summary>展开历史阶段记录：formal v2.1、Probe-guided adapter 及早期后续探索</summary>

The reusable idea-space taxonomy, priority navigation, Idea Card template, and
`IDEATE / PLAN / RUN` handoff prompt for new collaborators and Agents are in
`plan/AST_idea_space_and_agent_workflow.md`.

The MeowAgeNet dataset manifest, checksums, cat-ID-disjoint folds, VGGish
baseline, standard AST comparisons, and the IDEA-013/003/019 candidate studies
are recorded in this repository.

The feasibility-pilot phase is complete. The confirmed formal route is:

- research goal: **IDEA-048**, improving MeowAgeNet prediction performance;
- method route: **IDEA-019**, low-parameter AST adaptation;
- current reference implementation: **Probe-guided AST adapter**;
- working claim: low-parameter AST adaptation can improve animal-level feline
  age classification under the declared internal validation protocol.

`H` in the protocol means **Hypothesis**, a research hypothesis to be tested.
`H048` is the performance hypothesis derived from IDEA-048; `H019` is the
adapter-contribution hypothesis derived from IDEA-019.

IDEA-003 is paused and excluded from formal v2.1 because its pilot did not
improve overall performance. Probe-guided layer selection remains a replaceable
implementation candidate before the external execution lock; it is not a
frozen claim of unique layer semantics.

The completed core execution design is formal v2.1:

- amended protocol: `configs/protocol/meowagenet_formal_v2_1.json`;
- readable amendment: `plan/10_formal_protocol_v2_1_amendment.md`;
- execution-lock template:
  `configs/protocol/meowagenet_formal_v2_1_execution_lock_template.json`;
- completed execution lock:
  `configs/protocol/meowagenet_formal_v2_1_execution_lock.json`;
- deterministic split bank: `splits/meowagenet_formal_v2_*`;
- amendment metadata:
  `metadata/experiments/meowagenet_formal_v2_1_amendment.json`.
- candidate execution recipe:
  `configs/experiment/meowagenet_formal_v2_1_probe_guided_candidate_v1.json`;
- guarded formal runner: `scripts/run_meowagenet_formal_v2_1.py`;
- inner-only runner smoke record: `reports/11_formal_v2_1_runner_smoke.md`;
- formal core results: `reports/12_formal_v2_1_core_results.md`;
- machine-readable result audit:
  `metadata/experiments/meowagenet_formal_v2_1_core_results.json`;
- tracked formal execution logs:
  `runs/meowagenet_formal_v2_1_core/**/*.json`.

IDEA-049 begins a separate exploratory pretrained-backbone screening after the
formal-v2.1 checkpoint:

- readable plan: `plan/13_IDEA-049_backbone_screening_plan.md`;
- protocol: `configs/protocol/meowagenet_idea049_backbone_screening_v1.json`;
- first candidate recipe:
  `configs/experiment/idea049/ssast_base_patch400_frozen_v1.json`;
- SSAST checkpoint card: `metadata/models/idea049/ssast_base_patch400.json`.
- independent runner: `scripts/run_meowagenet_idea049.py`;
- SSAST initial-screening report:
  `reports/14_IDEA-049_SSAST_initial_screening_results.md`;
- machine-readable SSAST result:
  `metadata/experiments/meowagenet_idea049_ssast_initial_v1_results.json`.

The SSAST candidate completed its 792-call frozen embedding cache, inner-only
smoke test, and 12 initial-screening fits. Its three seed-17 complete-OOF animal
macro F1 values were 0.6371, 0.6983, and 0.5521 (mean 0.6292), compared with
0.7337 for the paired AST head-only reference. It is retained as a completed
`screened_not_better` candidate. PaSST then entered as the second candidate.

The second IDEA-049 candidate, PaSST, has also completed its resource audit,
792-call embedding cache, inner-only smoke test, and 12 initial-screening fits:

- runner: `scripts/run_meowagenet_idea049_passt.py`;
- recipe: `configs/experiment/idea049/passt_s_ap476_frozen_v1.json`;
- checkpoint card: `metadata/models/idea049/passt_s_ap476.json`;
- Chinese result report:
  `reports/15_IDEA-049_PaSST_initial_screening_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea049_passt_initial_v1_results.json`.

PaSST achieved seed-17 complete-OOF animal macro F1 values of 0.6319, 0.6741,
and 0.6619 (mean 0.6560, sample SD 0.0217). It improved over the SSAST screen
mean by 0.0268 while remaining 0.0777 below the paired AST head-only mean. The
candidate is retained as `screened_not_better`.

The third candidate, PANNs CNN14, completed the same resource audit, 792-call
embedding cache, inner-only smoke, and 12 initial-screening fits:

- runner: `scripts/run_meowagenet_idea049_panns.py`;
- recipe: `configs/experiment/idea049/panns_cnn14_frozen_v1.json`;
- checkpoint card: `metadata/models/idea049/panns_cnn14_audioset.json`;
- Chinese result report:
  `reports/16_IDEA-049_PANNs_CNN14_initial_screening_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea049_panns_initial_v1_results.json`.

PANNs achieved seed-17 complete-OOF animal macro F1 values of 0.6121, 0.5991,
and 0.5540 (mean 0.5884, sample SD 0.0305). Its matched difference from AST
head-only averaged -0.1453 across the three repeats. It is retained as a
completed `screened_not_better` candidate.

The fourth candidate, AVES-base-bio, completed its official resource audit,
792-call embedding cache, inner-only smoke, and 12 initial-screening fits:

- runner: `scripts/run_meowagenet_idea049_aves.py`;
- recipe: `configs/experiment/idea049/aves_base_bio_frozen_v1.json`;
- checkpoint card: `metadata/models/idea049/aves_base_bio.json`;
- Chinese result and stage-closeout report:
  `reports/17_IDEA-049_AVES_initial_screening_and_stage_closeout.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea049_aves_initial_v1_results.json`.

AVES achieved complete-OOF animal macro F1 values of 0.6675, 0.6649, and
0.6865 (mean 0.6730, sample SD 0.0118). It ranks first among the four new
backbones and is close to matched VGGish on macro F1 while improving balanced
accuracy and QWK. Matched AST head-only remains higher by 0.0607 on average.
IDEA-049 initial backbone screening closes after AVES; Conformer and expanded
seeds remain future-work options.

After the IDEA-049 closeout, a separate exploratory AST hyperparameter stage
compared eight inner-only configurations each for AST head-only and the
Probe-guided AST adapter. The search was locked before exploratory outer
evaluation:

- protocol: `configs/protocol/meowagenet_ast_hpo_v1.json`;
- independent runner: `scripts/run_meowagenet_ast_hpo_v1.py`;
- Chinese result report:
  `reports/18_AST_head_and_adapter_hyperparameter_search.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_ast_hpo_v1_results.json`.

The search selected dropout 0.4457 and head learning rate 0.006 for AST
head-only, while the adapter retained its existing configuration. Across three
seed-17 complete OOF evaluations, tuned head-only achieved animal macro F1
0.7488 versus 0.7367 for the adapter. Relative to the matched historical
formal-v2.1 head-only mean, tuned head-only improved macro F1 by 0.0151 and QWK
by 0.0125. This is recorded as a provisional post-formal performance result;
the completed formal-v2.1 evidence remains unchanged.

The next focused method plan is IDEA-050, an AST LoRA study:

- readable plan: `plan/19_IDEA-050_AST_LoRA_plan.md`;
- shared head recipe: dropout 0.4457 and head learning rate 0.006 from
  AST-HPO-v1; the principal HPO signal is the higher head learning rate, while
  dropout is retained as part of the selected combination;
- primary comparison: selected Q/V LoRA against a contemporaneously rerun,
  matched tuned AST head-only control;
- first-stage scope: five bounded LoRA candidates selected within each outer
  fold, followed by three seed-17 nested complete OOF evaluations per pipeline.

The LoRA stage treats the HPO result as exploratory candidate evidence and
keeps formal-v2.1 as the historical formal anchor. LoRA seed expansion to 43
and 101 follows team review of the initial paired result.

IDEA-050 initial nested paired evaluation has now completed:

- executable protocol: `configs/protocol/meowagenet_idea050_ast_lora_v1.json`;
- independent runner: `scripts/run_meowagenet_idea050_ast_lora.py`;
- Chinese result report: `reports/20_IDEA-050_AST_LoRA_initial_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea050_ast_lora_initial_v1_results.json`.

The run completed 60 inner-only candidate fits, 12 pre-outer selection locks,
and 24 outer pipeline fits. Across three seed-17 complete OOF evaluations,
matched tuned AST head-only achieved mean animal macro F1 0.7488 and selected
AST LoRA achieved 0.7174. The paired differences were -0.0043, -0.0465, and
-0.0433. The current five-candidate Q/V LoRA stage therefore closes without
seed-43/101 expansion; its full ablation and audit evidence remains available
for the thesis, while future LoRA variants remain open as later candidates.

The first AST accuracy-enhancement matrix has completed its diagnostics,
inner-only smoke, execution lock, and 48 seed-17 outer fits:

- design plan: `plan/AST_accuracy_enhancement_candidates.md`;
- executable protocol:
  `configs/protocol/meowagenet_ast_accuracy_enhancement_v1.json`;
- independent runner:
  `scripts/run_meowagenet_ast_accuracy_enhancement_v1.py`;
- Chinese result report:
  `reports/21_AST_cat_balancing_and_multilayer_fusion_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_ast_accuracy_enhancement_v1_results.json`.

The matched tuned AST A0 control reproduced the earlier HPO result exactly at
mean animal macro F1 0.7488. Cat-balanced final-layer AST (A1) achieved 0.7765,
with paired gains of 0.0081, 0.0147, and 0.0603 across the three repeats;
balanced accuracy reached 0.7864 and QWK reached 0.6724. All-layer scalar
fusion achieved 0.7027 with class balancing and 0.7230 with cat balancing.
Cat balancing is retained as the provisional performance candidate. Its preset
condition for a later matched A0/A1 expansion to seeds 43 and 101 is met.

The matched cat-balancing seed expansion has now completed:

- executable protocol:
  `configs/protocol/meowagenet_ast_cat_balance_seed_expansion_v1.json`;
- independent runner:
  `scripts/run_meowagenet_ast_cat_balance_seed_expansion_v1.py`;
- Chinese result report:
  `reports/22_AST_cat_balancing_seed_expansion_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_ast_cat_balance_seed_expansion_v1_results.json`.

The expansion added base seeds 43 and 101 with three repeats and four folds,
completing 48 new outer fits. Combined with seed 17, each pipeline now has nine
111-cat complete-OOF evaluations. Cat-balanced A1 achieved mean animal macro
F1 0.7416 versus 0.7385 for matched A0, with six of nine paired comparisons
positive. A1 also achieved higher plain accuracy and lower macro-F1 sample SD;
A0 retained small advantages in balanced accuracy and QWK. The preset stage
gate was met at its boundary. A later code audit found that both weighted
pipelines normalize their weights inside each micro-batch. The stored lookup
weights have the intended global totals, while the effective optimizer
coefficients do not preserve exact equal-cat totals. Combined with the small
mixed-seed effect (+0.0031 macro F1), this stage is retained as an executed
per-mini-batch normalized cat-aware weighting study.

The matched global-weighting correction and retest has now completed:

- design plan: `plan/AST_cat_balance_global_weighting_retest.md`;
- executable protocol:
  `configs/protocol/meowagenet_ast_cat_balance_global_weighting_v1.json`;
- independent runner:
  `scripts/run_meowagenet_ast_cat_balance_global_weighting_v1.py`;
- Chinese result report:
  `reports/23_AST_cat_balance_global_weighting_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_ast_cat_balance_global_weighting_v1_results.json`.

The corrected experiment completed 72 outer fits, 18 complete OOF evaluations,
and nine paired comparisons. Global class-balanced C0 achieved mean animal
macro F1 0.7400 versus 0.7326 for global cat-and-class-balanced C1, a paired
difference of -0.0074. C1 was higher in three of nine comparisons and one of
three base-seed means. The paired cat-cluster bootstrap interval was
[-0.0380, 0.0238]. C0 also led balanced accuracy, QWK, and plain accuracy;
C1 retained a small adult-recall advantage. Under the preset rule, the strict
global cat-balancing stage closes with no improvement evidence. C0 remains the
tuned frozen-AST reference, while future AST accuracy ideas remain open.

IDEA-051 cat-level set aggregation has completed its diagnostic, inner-only
smoke, execution lock, and 36 seed-17 outer fits:

- Idea Card: `plan/IDEA-051_cat_level_set_aggregation.md`;
- executable protocol: `configs/protocol/meowagenet_idea051_cat_set_v1.json`;
- independent runner: `scripts/run_meowagenet_idea051_cat_set.py`;
- Chinese result report:
  `reports/24_IDEA-051_cat_level_set_aggregation_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea051_cat_set_v1_results.json`.

The matched call-probability-mean S0 achieved mean animal macro F1 0.7570,
balanced accuracy 0.7645, and QWK 0.6721. Cat-level hidden-mean S1 achieved
0.6777 macro F1, and learned-attention S2 achieved 0.6688. Their paired mean
differences from S0 were -0.0794 and -0.0883, with all six repeat-level
comparisons favoring S0. S2 learned clearly non-uniform call weights and raised
QWK by 0.0170 relative to S1, providing a useful pooling-mechanism result while
remaining below S0 on the primary endpoint. The preset seed-expansion gate did
not activate, so this implementation closes as a completed prediction-unit and
pooling ablation. S0's animal-level checkpoint selection also produced a
+0.0106 seed-17 mean gain over the earlier strict-global C0 recipe and is
retained as a separate lightweight follow-up candidate.

IDEA-052 AST local acoustic residual has completed its diagnostic, deterministic
inner-only smoke, execution lock, and 36 seed-17 outer fits:

- Idea Card: `plan/IDEA-052_AST_local_acoustic_residual.md`;
- executable protocol:
  `configs/protocol/meowagenet_idea052_ast_local_residual_v1.json`;
- independent runner:
  `scripts/run_meowagenet_idea052_ast_local_residual.py`;
- Chinese result report:
  `reports/25_IDEA-052_AST_local_acoustic_residual_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea052_ast_local_residual_v1_results.json`.

The matched global R0 reproduced the IDEA-051 reference at mean animal macro F1
0.7570. Temporal-mean residual R1 achieved 0.7562 and produced paired differences
of +0.0107, +0.0131, and -0.0264 across the three repeats, for a mean difference
of -0.0009. Temporal-salience residual R2 achieved 0.7431, a mean difference of
-0.0139. Both zero-initialized 128-dimensional gates learned active residual
weights, while neither candidate reached the preset mean-gain threshold for
seed expansion. This parameterization closes as an informative local/global
ablation, and the ordered workflow proceeds to AST-VGGish complementarity,
fusion, or distillation.

IDEA-053 AST-VGGish probability fusion has completed its paired diagnostic,
inner-only smoke, execution lock, and 24 seed-17 model fits across 12 folds:

- Idea Card: `plan/IDEA-053_AST_VGGish_probability_fusion.md`;
- complementarity diagnostic:
  `metadata/experiments/meowagenet_idea053_ast_vggish_complementarity_v1.json`;
- executable protocol:
  `configs/protocol/meowagenet_idea053_ast_vggish_probability_fusion_v1.json`;
- independent runner:
  `scripts/run_meowagenet_idea053_ast_vggish_fusion.py`;
- Chinese result report:
  `reports/26_IDEA-053_AST_VGGish_probability_fusion_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea053_ast_vggish_probability_fusion_v1_results.json`.

Across 333 paired cat evaluations, AST and VGGish were both correct 209 times,
AST alone was correct 43 times, and VGGish alone was correct 21 times. This
established measurable complementarity. The executable fusion selected one AST
probability weight per fold using only 17 inner-validation cats. It achieved
mean animal macro F1 0.7499 versus 0.7570 for AST, with paired differences of
+0.0082, -0.0149, and -0.0147. The fusion changed 13 AST decisions, gaining five
correct predictions and losing eight. The seed-expansion gate did not activate;
the tuned frozen AST remains the reference, and the ordered workflow proceeds
to constrained LayerNorm / SSF / BitFit calibration. Conditional feature fusion
remains available as a later independent idea.

IDEA-054 constrained AST calibration has completed its inner-only smoke,
execution lock, and 48 seed-17 outer fits:

- Idea Card: `plan/IDEA-054_constrained_AST_calibration.md`;
- executable protocol:
  `configs/protocol/meowagenet_idea054_constrained_ast_calibration_v1.json`;
- independent runner:
  `scripts/run_meowagenet_idea054_constrained_ast_calibration.py`;
- Chinese result report:
  `reports/27_IDEA-054_constrained_AST_calibration_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea054_constrained_ast_calibration_v1_results.json`.

The matched frozen-AST A0 achieved mean animal macro F1 0.7570. LayerNorm
tuning achieved 0.7429, block-output SSF achieved 0.7356, and BitFit achieved
0.7372. Their paired mean differences from A0 were -0.0141, -0.0214, and
-0.0198. LayerNorm improved one of three repeats; SSF and BitFit were lower in
all six repeat-level comparisons. All seed-expansion gates closed. The three
parameterizations are retained as a constrained-PEFT ablation, frozen AST
remains the reference, and the ordered workflow proceeds to checkpoint
averaging or output calibration.

IDEA-055 checkpoint ensemble and animal-level class-bias calibration has
completed its inner-only smoke, execution lock, and 12 shared seed-17 head
training trajectories across the 12 outer folds. Four inference pipelines were
derived from those trajectories, producing 48 fold-level predictions:

- Idea Card: `plan/IDEA-055_checkpoint_ensemble_and_class_bias_calibration.md`;
- executable protocol:
  `configs/protocol/meowagenet_idea055_checkpoint_calibration_v1.json`;
- independent runner:
  `scripts/run_meowagenet_idea055_checkpoint_calibration.py`;
- Chinese result report:
  `reports/28_IDEA-055_checkpoint_calibration_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea055_checkpoint_calibration_v1_results.json`.

The matched single-checkpoint A0 reproduced mean animal macro F1 0.7570.
Tail-three checkpoint probability averaging achieved 0.7552, improved balanced
accuracy from 0.7645 to 0.7665, and reduced animal cross-entropy from 0.7160 to
0.7059. Class-bias calibration alone achieved 0.7513, while ensemble plus bias
achieved 0.7457. All seed-expansion gates closed. This completes the current
five-priority exploration sequence as a stage checkpoint: A0 remains the main
performance reference, checkpoint averaging is retained as supporting evidence
for probability quality, and future ideas remain open for later evaluation.

The next confirmation plan is IDEA-056. It compares the historical
call-level-validation-loss checkpoint rule with animal-level-validation-loss
checkpoint selection on new base seeds 43 and 101. Seed 17 remains historical
exploratory evidence and is excluded from the primary confirmation decision:

- readable plan:
  `plan/IDEA-056_animal_level_checkpoint_selection_confirmation.md`;
- executable protocol:
  `configs/protocol/meowagenet_idea056_checkpoint_selection_confirmation_v1.json`;
- independent runner:
  `scripts/run_meowagenet_idea056_checkpoint_selection.py`;
- Chinese result report:
  `reports/29_IDEA-056_checkpoint_selection_confirmation_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea056_checkpoint_selection_confirmation_v1_results.json`.

IDEA-056 completed 24 shared fold trajectories and six new complete-OOF paired
comparisons. Animal-level-CE checkpoint selection increased mean animal macro
F1 from 0.7368 to 0.7426 (`+0.0059`), with 4/6 positive pairs, meeting both
predeclared confirmation thresholds. It also improved plain accuracy by 0.0150,
reduced animal CE by 0.0260, and reduced macro-F1 SD from 0.0321 to 0.0186.
Balanced accuracy changed by -0.0201 and QWK by -0.0029 because the new rule
improved adult recognition while trading kitten and senior recall. Animal-level
CE selection is now the tuned frozen-AST reference checkpoint procedure; C0 is
retained as its matched ablation.

After IDEA-056, the ordered method stage is: diagnose where AST is losing or
underusing information; select one AST-internal direction; test that single
module against a parameter-matched control and original AST; combine modules
only after a stable positive signal.

The active stage plan is documented in
`plan/AST_internal_diagnosis_and_single_module_plan.md`. It treats local
time-frequency patches, intermediate-layer representations, and pretraining
domain mismatch as competing diagnostic directions. No new architecture is
selected until the diagnostic decision record is complete.

The inner-only diagnostic stage has completed across 12 train/validation
splits. The final AST layer outperformed the best standalone intermediate layer
by 0.0297 macro F1. The middle-frequency regional probe achieved 0.7437 versus
0.7649 for the global probe, while middle-time mean replacement reduced true-
class probability in all 12 splits. Last-two-block adaptation improved mean
inner-validation macro F1 by 0.0249, with 5/12 positive and 3/12 tied splits,
while animal CE increased by 0.0023. No axis passed its full diagnostic support
rule; the decision record remains open, with local patch retained as the
strongest provisional mechanism lead:

- protocol: `configs/protocol/meowagenet_ast_internal_diagnosis_v1.json`;
- runner: `scripts/run_meowagenet_ast_internal_diagnosis.py`;
- Chinese report: `reports/30_AST_internal_diagnosis_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_ast_internal_diagnosis_v1_results.json`.

On 2026-09-15, the project team decided to advance both diagnostic leads as
separate experiments. IDEA-057 tests a position-aware local time-frequency
patch branch; IDEA-058 tests constrained top-block adaptation against a
parameter-matched block-location control. The two routes share the IDEA-056
reference and remain independent until each has produced stable evidence:

- joint stage decision: `plan/AST_dual_direction_validation_plan.md`;
- IDEA-057: `plan/IDEA-057_structured_local_patch_branch.md`;
- IDEA-058: `plan/IDEA-058_constrained_top_block_adaptation.md`.

IDEA-057 has completed its bounded inner-only selection, execution lock, and
36-fit R0/M1/C1 complete-OOF evaluation. The selected middle-frequency temporal
strip achieved mean animal macro F1 0.7265, versus 0.7570 for matched R0 and
0.7348 for the parameter-matched position-removed control. Its paired mean
differences were -0.0305 versus R0 and -0.0083 versus C1, so the seed-expansion
gate closed. The structured local-patch implementation is retained as a
mechanism ablation; the independent workflow then proceeded to IDEA-058:

- Chinese result report: `reports/31_IDEA-057_structured_local_patch_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea057_structured_local_patch_v1_results.json`.

IDEA-058 has also completed its bounded selection and 36-fit R0/M1/C1
complete-OOF evaluation. Top-block M1 achieved mean animal macro F1 0.7607
versus 0.7570 for matched frozen R0 and 0.7385 for the parameter-matched
bottom-block C1. M1 exceeded R0 in two of three repeats and C1 in all three,
while its repeat SD increased to 0.0371. The mean M1-R0 gain of 0.0037 remained
below the prespecified 0.005 seed-expansion gate:

- Chinese result report: `reports/32_IDEA-058_constrained_top_block_results.md`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea058_constrained_top_block_v1_results.json`.

A post-result audit found that IDEA-058 selected one global recipe after
aggregating inner-validation results across all outer folds. Because animals
change roles across folds, the next step is a per-outer-fold strict nested
confirmation before extending the method. The complete post-IDEA-058 roadmap
is in `plan/POST_IDEA058_next_stage_plan.md`. It separates four roles: mandatory
evaluation correction, conditional top-block stabilization, the pre-existing
IDEA-039 grouped-augmentation route, and a nuisance-variable diagnostic that
must establish prediction dependence before a new method is proposed. IDEA-021
is excluded from this stage by the current team decision.

The IDEA-058 strict nested confirmation has now completed. Per-fold recipe
selection retained the exact R0 anchor at 0.7570 mean animal macro F1. M1
top-block adaptation achieved 0.7493, a paired mean difference of -0.0077 from
R0 with one of three repeats positive. M1 remained 0.0156 above the matched
bottom-block C1 on average, while the paired bootstrap interval crossed zero.
The strict seed-expansion gate therefore closed, IDEA-058 is recorded as an
exploratory weak-positive that was not confirmed, and top-block stabilization
is paused:

- Chinese result report:
  `reports/33_IDEA-058_strict_nested_confirmation_results.md`;
- executable protocol:
  `configs/protocol/meowagenet_idea058_strict_nested_v1.json`;
- machine-readable result:
  `metadata/experiments/meowagenet_idea058_strict_nested_v1_results.json`.

The active performance plan is now IDEA-039 grouped augmentation. It separates
the effect of a fixed mild augmentation policy from the incremental value of
per-outer-fold nested policy selection and includes an online/no-op execution
control. The readable plan is
`plan/IDEA-039_grouped_augmentation_policy.md`; the current roadmap is
`plan/POST_IDEA058_strict_stage_update.md`. The pre-strict
`plan/POST_IDEA058_next_stage_plan.md` remains unchanged because its SHA-256 is
part of the executed strict protocol.

Formal v2.1 freezes the evidence-critical core while leaving the exact adapter,
three-to-five split repeats, and optional diagnostic modules selectable before
formal outcomes. The minimum core is three pipelines, three repeats, four folds,
and three model seeds, totaling 108 fold-level fits. The earlier strict v2
matrix remains in the repository as a design-history record.

Formal v2.1 is a pilot-informed repeated internal validation on the same 111
cats, not an independent external replication. The locked minimum core has now
completed all 108 fold-level fits and 27 complete OOF evaluations.

The runner exposes separate `inner-only` and `formal` scopes. `inner-only`
trains and validates within the nested development roles and produces no
outer-test predictions. `formal` requires a completed execution lock whose
recipe and runner hashes match before outer-test prediction begins.

The formal-v2.1 aggregate is:

| Pipeline | Animal macro F1, mean ± SD | Balanced accuracy | QWK |
| --- | ---: | ---: | ---: |
| VGGish + MLP | 0.6525 ± 0.0462 | 0.6525 | 0.5334 |
| AST head-only | 0.7238 ± 0.0335 | **0.7597** | **0.6374** |
| Probe-guided AST adapter | **0.7290 ± 0.0428** | 0.7419 | 0.6373 |

The adapter improved macro F1 over VGGish by 0.0765 on average, with all nine
paired OOF comparisons positive. Its incremental difference over matched AST
head-only was 0.0052, so the formal evidence supports the AST route strongly
while treating the adapter-specific contribution as split-dependent.

The prior stage checkpoint remains historical pilot evidence:

| Pipeline | Animal macro F1 | Role |
| --- | ---: | --- |
| Locked VGGish + MLP | 0.6846 | Single formal baseline recipe |
| Probe-guided AST adapter | 0.7575 | Pilot reference candidate |

The pilot checkpoint is documented in
`reports/08_IDEA-048_stage_checkpoint.md`; formal results must not overwrite
that record.

</details>
