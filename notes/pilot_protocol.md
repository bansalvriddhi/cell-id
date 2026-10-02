# Frozen evaluation design: CELL-ID linear pilot

This design is written before inspection of any held-out prediction scores. Dataset-specific eligibility, dimensions and assay conditions will be appended after the source-data audit and before the experiment run.

## Source-data addendum, before empirical evaluation

The acquired data are Tahoe-100M author-released `log2FoldChange` estimates, not newly computed mean-expression deltas. The pilot covers ten contexts and 58 common drugs at 0.05 µM on plate 1. All 580 context–drug profiles are present, but some gene estimates are missing. The exact source differential-expression fitting and exposure duration are unverified, and there are no untreated expression vectors. Accordingly the task is retrospective published-signature completion, without a control-expression encoder.

Input is `data/processed_v1/responses.npz`; retain all source NaNs. Inside every outer and inner training fold, eligible genes must have finite estimates in all training context–drug profiles. Among these, select up to 1,000 genes by training residual variance only. The held-out context's gene availability must not influence eligibility or ranking. A target's finite observations define only its evaluation mask, which is common to all methods; report coverage. A probe reveals its values and availability together. Condition the posterior only on that probe's observed genes, without zero-filling missing data.

Before querying a candidate, use its nominal all-selected-gene design matrix; after querying, update the covariance with its actually observed rows. Thus future covariance-based selection may react to acquired missingness but remains independent of response magnitudes. The complete-data derivation describes its nominal objective. The source's LFC standard errors are not used as validated biological-replicate noise.

Freeze the probe/target partition at 29 drugs each, seed 20261002; budgets are 0, 1, 2, 3, 5, 10 and 20. Random acquisition uses 20 seeds. Inner validation selects ranks from 2, 4, 8, capped to available rank, and effective noise multipliers 0.1, 1, 10, evaluated with five random probes and three seeds. Noise baseline is the median selected-gene pre-truncation centered context residual variance, averaged over drugs. Ties use immutable identifiers or recorded deterministic ordering. Context bootstrap uses 2,000 resamples, seed 77. This amendment addresses the actual source's missingness before viewing any prediction scores.

## Question and scope

Can a compact linear-Gaussian response model recover transcriptomic responses in a completely held-out cell context from a small number of measured probes, and does model-based probe design improve over random acquisition? This pilot evaluates static response signatures at the source dataset's documented assay conditions. It does not test clinical response, causal-network recovery, new drug structures, combinations, time evolution, or patient populations.

## Response model

For each outer context holdout, compute the training-context mean signature for each perturbation, denoted mu[p,g]. Center the training context-by-perturbation-by-gene tensor around mu. Learn a low-rank context covariance from these centered profiles, using only the outer training contexts. Write the resulting model as y[p] = mu[p] + A[p] z + epsilon[p], with z ~ Normal(0,I). A[p] maps the context latent coordinates to gene responses for perturbation p. A truncated singular-value factorization of the training residual tensor defines A.

The model conditions on the actual measured signature only after a probe is acquired. Posterior precision is I + sum(A[p].T @ A[p] / tau^2) and the posterior mean is the inverse precision times sum(A[p].T @ (y[p]-mu[p]) / tau^2). The regularization/noise scale is an effective model-mismatch parameter, not a measured biological variance or an empirically calibrated confidence interval. Every unqueried response is predicted as mu[p] + A[p] posterior_mean.

## Comparisons

Use exactly the same trained basis, predictor and candidate/target split for every acquisition policy. Include:

1. Training-context mean, with no adaptation.
2. Uniform random probes, repeated across fixed seeds.
3. Greedy maximum predictive variance (uncertainty sampling).
4. Greedy D-optimal information gain: maximize logdet(I + Sigma A[p].T A[p] / tau^2).
5. Greedy predictive-variance reduction: maximize tr(Q (Sigma - Sigma_new)), where Q is the sum of A[q].T A[q] across the known target identities, and Sigma_new is the posterior covariance after adding p. This objective uses no target-context target responses.

These covariance-based designs are outcome-independent in a linear-Gaussian model. Their probe order can be computed in advance; the posterior mean adapts to observed outcomes. Call this an experimental-design pilot, not a learned outcome-adaptive policy. Chemical-diversity, clustering and nonlinear baselines are not silently approximated by differently named methods.

## Holdouts and leakage control

Hold out each eligible context in turn. Gene selection, representation, training means, covariance and hyperparameter tuning are refitted inside each outer split. No whole-dataset PCA or test-response-based gene selection is permitted. If dimensionality reduction is necessary, gene variance is calculated on outer training responses only.

Before model fitting, divide perturbation identities into a fixed candidate probe pool and a disjoint evaluation panel using a recorded seed and identifier-only ordering. This panel is identical across contexts and policies when the source matrix is complete. The target panel's measured held-out responses never enter selection, tuning or adaptation. The design may use training-context responses for the same perturbation identities. This tests unseen contexts with known perturbations, not unseen perturbations.

When enough contexts exist, tune model rank/regularization on inner context holdouts of the outer training data using random acquisition, not policy-specific performance on the outer test. Fix the candidate grid before seeing outer scores. If the number of contexts is too small for a meaningful inner split, use predeclared fixed parameters and state the limitation.

## Outputs and statistics

Primary metric: gene-wise mean squared error aggregated across the fixed target panel, normalized by the same outer context's no-adaptation MSE. Lower is better; a value of one is the training-context-mean baseline. Average repeated random-seed scores within a context before comparing contexts. Secondary metrics: mean Pearson correlation between predicted and measured delta signatures across genes; sign agreement on measured high-magnitude changes may be descriptive but is not a DEG test.

Budget points: 0, 1, 2, 3, 5, 10, 20, 50, capped by available eligible probes. Report normalized area under the error-versus-budget curve using trapezoidal integration divided by maximum budget, not a novelty claim for the metric. Number of probes counts acquired aggregate signatures, not individual cells or distinct drugs when conditions differ by dose/time.

The unit of analysis is a held-out cellular context, never a gene, cell or random seed. Use a fixed paired context bootstrap for uncertainty on average gains if sufficiently many contexts exist; small-context results are descriptive. Leave-one-context-out models share training contexts, so even these intervals are descriptive across the evaluated contexts rather than independent-cohort population guarantees. No hypothesis tests are required for this exploratory pilot. Any later exploratory analyses are labeled rather than merged with this prespecified primary comparison.

## Algorithmic checks

- Acquired probe identifiers and fixed targets are disjoint and the budget equals the number of distinct acquisitions.
- All policies share their k=0 prediction and target ground truth.
- No-adaptation normalized MSE equals one.
- Posterior covariances remain symmetric positive semidefinite; conditional covariance cannot increase under the assumed model.
- A withheld-response sentinel test changes all unrevealed target responses without changing fitted parameters, selected probes, or predictions.
- Recompute at least one recorded metric independently from saved truth and predictions.
- Record source checksums, exact software versions, seeds, dimensions, regularization, selected probes and split identifiers.
