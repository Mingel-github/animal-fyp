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

As of 2026-08-27, the current IDEA-048 development phase is complete. The
**Probe-guided AST adapter** is retained as the current reference candidate,
while later repetitions, new ideas, and candidate replacement remain open:

| Pipeline | Animal macro F1 | Role |
| --- | ---: | --- |
| Locked VGGish + MLP | 0.6846 | Single formal baseline |
| Probe-guided AST adapter | 0.7575 | Current reference candidate |

The stage checkpoint and its open-ended route are documented in
`reports/08_IDEA-048_stage_checkpoint.md`; the machine-readable record is in
`configs/protocol/meowagenet_idea048_stage_checkpoint_v1.json`.
