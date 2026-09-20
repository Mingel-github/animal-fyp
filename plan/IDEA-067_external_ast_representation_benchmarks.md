# IDEA-067 External AST Representation Benchmarks

## Purpose

Test whether the standard AudioSet-pretrained AST representation remains useful
outside MeowAgeNet when evaluation groups are entirely unseen animals.

## Tasks

- **CatMeows:** three-way meow-context classification (brushing, food waiting,
  isolation) using all 440 recordings from 21 cats.
- **Canine Age Transition:** five-way developmental-stage classification using
  a resource-limited, deterministic sample of at most 10 bark units per
  `dog_id × age_group` cell.  Sampling points are evenly spaced over sorted file
  paths, giving 2,290 bark units from all 125 dogs.

## Protocol

Use frozen standard-geometry AST call embeddings and one fixed balanced
multinomial logistic probe.  Run three repeated five-fold
StratifiedGroupKFold evaluations.  The grouping variable is `cat_id` or
`dog_id`; no animal can occur in both train and test within a fold.  The primary
metric is call-level Macro-F1, with balanced accuracy and mean per-animal call
accuracy as secondary summaries.  A train-prior dummy classifier is evaluated
on the identical folds.

The canine result is an exploratory subsample screen, not the full 79,142-unit
benchmark.  Neither task validates feline age prediction directly.  CatMeows
tests cross-cat context information; the canine task tests cross-species and
cross-individual developmental information.
