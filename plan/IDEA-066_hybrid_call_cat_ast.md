# IDEA-066 Hybrid Call+Cat AST

## Question

Can a frozen-AST classifier retain the dense supervision supplied by every call
while directly aligning its training objective with the animal-level decision?

## Mechanism

The classifier produces one probability vector per call.  For every cat in a
training batch, its call probabilities are averaged using exactly the same
arithmetic-mean rule used at inference.  The candidate objective is

`L = 0.5 * L_call + 0.5 * L_cat`,

where `L_call` is globally class-balanced call cross-entropy and `L_cat` is
globally class-balanced cross-entropy on the mean call probabilities of each
cat.  Unlike the negative IDEA-051 pure set models, this does not replace 792
call targets with 111 bag targets; every call retains a direct gradient.

## Matched control

`H0_matched_call_only` uses the same AST embeddings, classifier, cat batches,
call ordering, optimizer, initialization, and animal-level checkpoint rule.  It
differs only by setting the cat-loss coefficient to zero.  This control is
required because batching complete cats changes the BatchNorm context relative
to historical call-minibatch baselines.

## Boundary

The first stage is inner-only screening on the frozen nested train/validation
roles.  It does not read or export outer-test predictions.  The loss mixture is
fixed at 0.5/0.5; there is no coefficient search.  Passing the gate authorizes a
separately locked outer or external benchmark, not an automatic claim of
improvement.
