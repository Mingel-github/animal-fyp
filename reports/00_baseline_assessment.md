# Baseline assessment: 2025 candidate papers

Date checked: 2026-08-18

## Selection

Use **van Toor et al., "A deep learning pipeline for age prediction from
vocalisations of the domestic feline"** as the first reproduction target.

- The paper and repository are CC BY 4.0.
- The public repository contains labelled audio, precomputed embeddings,
  feature-extraction code, and the model-analysis notebooks.
- Upstream repository: <https://github.com/aster-droide/feline-age-prediction>
- Pinned upstream commit: `3d02295bef1500d2b2500a124596f77010181391`
- Local checkout: `src/baselines/feline-age-prediction`
- The local checkout is a partial/sparse clone. It currently contains the two
  core VGGish analysis notebooks and the matching 1.2 MiB embeddings CSV; the
  large raw-audio blobs remain available from the promisor remote.

Do not use **Huang and Situ, "Beyond Discrete Categories: Multi-Task
Valence-Arousal Modeling for Pet Vocalization Analysis"** as the first
reproduction target. The paper does not publish a code repository or its
LingChongTong audio dataset. Of its 42,553 samples, 25,570 are augmented, so
the reported total is not an independently collected public corpus.

## Local environment audit

| Item | Local machine | Paper environment | Assessment |
| --- | --- | --- | --- |
| OS | Windows 10 Home 22H2, build 19045 | Darwin 21.6.0 | Different but CPU TensorFlow is supported |
| CPU | AMD Ryzen 5 5625U, 6C/12T | Apple arm64 | Sufficient for the small MLP baseline |
| RAM | 15.34 GiB | 16 GiB | Matched closely |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8188 MiB reported by `nvidia-smi` (driver 595.79); integrated graphics may also be present | 0 GPUs | CUDA-capable GPU is available; the original VGGish/TensorFlow reproduction still ran on CPU |
| Python | Anaconda 2025.06 base, Python 3.13.5; not on PATH | Python 3.10.12 | Isolated Python 3.10.12 environment created |
| Git | 2.53.0.windows.2 | Not reported | Ready |
| Free space on E: | 173.7 GiB at audit time | Not reported | Enough for the approximately 415 MiB upstream repository |

## Upstream reproduction details

- Input CSV: `models_analysis_notebooks/vggish-embeddings/vggish_looped_embeddings.csv`
- CSV integrity: Git blob `947a7c9baa983c18009f2a85bc12d95e51fbd48b`
- CSV shape: 937 embedding rows, 128 embedding dimensions, `mean_freq`,
  `gender`, `target`, and `cat_id` columns.
- Official raw-audio audit: 793 `AudioCropped` paths, 55,828,626 bytes,
  kitten/adult/senior counts 135/405/253, and 112 published IDs. The observed
  duration range is 0.0844--4.4253 seconds (mean 0.7247 seconds), matching the
  paper.
- Exact duplicate audit: `0Y-041A-01.wav` and `0Y-049A.wav` are byte-identical
  and have identical VGGish features while carrying different IDs. Preserving
  all official paths gives 793 rows/112 IDs; excluding the later duplicate
  alias gives 792 unique calls/111 analysis IDs. This reconciles the reported
  111 numerically, although the upstream materials do not document the rule.
- Notebook seed generation: NumPy seed 42 produces
  `[7270, 860, 5390, 5191, 5734]`.
- Evaluation: five seeds, four-fold `StratifiedGroupKFold`, grouped by
  `cat_id`.
- VGGish categorical MLP: Dense(128), batch normalisation, dropout
  0.44571035356880917, Dense(3), Adamax learning rate
  0.003109800273709165, batch size 128, up to 1500 epochs, early-stopping
  patience 30.
- Stored categorical notebook result: macro F1 approximately 0.719 and macro
  accuracy approximately 74.10%, consistent with the paper's rounded table.

## Known reproducibility risks

1. The upstream repository has no requirements file or environment lock.
2. VGGish feature extraction is kept in a second repository; the downloaded
   embeddings allow the downstream baseline to be reproduced first.
3. The notebooks contain repeated, manually expanded seed/fold code rather
   than a single parameterised training entry point.
4. The notebooks manually force selected cat IDs (for example `046A`, and in
   some sections `000A`) into the train/validation side by swapping groups.
   This must be preserved for an exact reproduction and separately audited
   for methodology sensitivity.
5. The notebook uses half-open age ranges: kitten `[0, 0.5)`, adult
   `[0.5, 10)`, senior `[10, 20)`. This boundary should not be inferred from
   prose alone.

The environment file `environment/meowagenet-repro.yml` preserves versions
reported by the paper and supplies compatible pins for imported packages that
the authors left unspecified.

## Local smoke test

- Environment prefix: `environment/.conda/meowagenet-repro`
- Verified imports: TensorFlow 2.15.0, NumPy 1.25.2, pandas 1.5.3,
  scikit-learn 1.3.2, imbalanced-learn 0.11.0, and Optuna 3.5.0.
- TensorFlow 2.15 native-Windows environment exposed only the CPU during this
  reproduction. This does not describe the machine's full hardware: a later
  `nvidia-smi` re-audit confirmed an RTX 4060 Ti. PyTorch CUDA is used for the
  AST rerun.
- The 937-row VGGish CSV loaded successfully and reported 112 distinct
  `cat_id` values.
- The pinned 793-file `AudioCropped` dataset was downloaded separately from
  the fixed upstream commit and every file passed Git-blob and SHA-256
  verification. Versioned manifests are in `metadata/datasets/meowagenet`;
  raw audio remains ignored under `data/`.
- A one-epoch categorical-model smoke test used the paper's Dense(128), batch
  normalisation, dropout, Dense(3), Adamax, learning-rate, and batch-size
  settings. The first grouped fold contained 740 training rows and 197
  validation rows, with zero cat-ID overlap. The run completed successfully.
