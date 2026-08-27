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
| `runs/` | Generated experiment outputs; ignored by Git |
| `scripts/` | Command-line entry points and utilities |
| `splits/` | Reproducible train/validation/test split definitions |
| `src/` | Project source and external baseline references |
| `tests/` | Automated tests |

## Data policy

Do not commit raw audio, personal data, credentials, trained checkpoints, or
generated run artifacts. Record acquisition instructions, licenses, checksums,
and preprocessing steps so collaborators can reconstruct permitted datasets.

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

The active execution design is formal v2.1:

- amended protocol: `configs/protocol/meowagenet_formal_v2_1.json`;
- readable amendment: `reports/10_formal_protocol_v2_1_amendment.md`;
- execution-lock template:
  `configs/protocol/meowagenet_formal_v2_1_execution_lock_template.json`;
- deterministic split bank: `splits/meowagenet_formal_v2_*`;
- amendment metadata:
  `metadata/experiments/meowagenet_formal_v2_1_amendment.json`.
- candidate execution recipe:
  `configs/experiment/meowagenet_formal_v2_1_probe_guided_candidate_v1.json`;
- guarded formal runner: `scripts/run_meowagenet_formal_v2_1.py`;
- inner-only runner smoke record: `reports/11_formal_v2_1_runner_smoke.md`.

Formal v2.1 freezes the evidence-critical core while leaving the exact adapter,
three-to-five split repeats, and optional diagnostic modules selectable before
formal outcomes. The minimum core is three pipelines, three repeats, four folds,
and three model seeds, totaling 108 fold-level fits. The earlier strict v2
matrix remains in the repository as a design-history record.

Formal v2.1 is a pilot-informed repeated internal validation on the same 111
cats, not an independent external replication. Training and formal comparison
run in the external experiment environment after the runner passes checks and
the execution lock is completed.

The runner exposes separate `inner-only` and `formal` scopes. `inner-only`
trains and validates within the nested development roles and produces no
outer-test predictions. `formal` requires a completed execution lock whose
recipe and runner hashes match before outer-test prediction begins.

The prior stage checkpoint remains historical evidence:

| Pipeline | Animal macro F1 | Role |
| --- | ---: | --- |
| Locked VGGish + MLP | 0.6846 | Single formal baseline recipe |
| Probe-guided AST adapter | 0.7575 | Pilot reference candidate |

The pilot checkpoint is documented in
`reports/08_IDEA-048_stage_checkpoint.md`; formal results must not overwrite
that record.
