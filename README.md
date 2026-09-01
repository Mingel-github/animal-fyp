# Animal Vocalization FYP

This repository contains the code, configuration, experiment metadata, and
reports for an animal-vocalization final-year project.

The first reproduction target is van Toor et al.'s feline age-prediction
pipeline. The authors' repository is kept as an upstream Git submodule under
`src/baselines/feline-age-prediction`; its code is not vendored as original
project work.

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
| `reports/` | Reproduction assessments and results |
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

## Current research status

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
- readable amendment: `reports/10_formal_protocol_v2_1_amendment.md`;
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

- readable plan: `reports/13_IDEA-049_backbone_screening_plan.md`;
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

- readable plan: `reports/19_IDEA-050_AST_LoRA_plan.md`;
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
