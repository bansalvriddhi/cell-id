# Exploratory context-offset comparator

This is a separate exploratory comparison requested after the primary pilot, not a primary protocol amendment or a search for favorable results. No tuning or model selection is performed. Scope remains completion of author-released log2FC signatures at plate 1 / 0.05 µM, with source methods and exposure duration unverified.

## Frozen method and deliverables

- [ ] `scripts/context_shift_baseline.py`: independent NumPy CLI, CPU-only, two-thread caps, no imports from or edits to primary code.
- [ ] `results/pilot/context_shift/metrics.csv`: `context,policy,seed,budget,mse,baseline_mse,normalized_mse,target_observed_entries,target_possible_entries,corrected_genes`; 20 original random seed orders, budgets `[0,1,2,3,5,10,20]`; full-precision floats, squared log2FC MSE, dimensionless normalized MSE.
- [ ] `results/pilot/context_shift/per_context.csv`: `context,policy,budget,n_seeds,mse,baseline_mse,normalized_mse`; seeds averaged within context before aggregation.
- [ ] `results/pilot/context_shift/summary.json`: per-budget context mean/min/max and descriptive paired-context bootstrap (2000 resamples, seed77); k5 comparisons against primary random and V-optimal. Positive comparison gain means primary comparator error minus context-offset error. No significance claims.
- [ ] `results/pilot/context_shift/manifest.json`: input and primary file checksums, source script checksum, software versions, exact budgets/seeds, completed status and output hashes. Refuse existing output directories.
- [ ] `results/pilot/context_shift/checkpoints/context_NNN_seed0.npz`: predictions/truth/mask and corrections at k0/k1/k5, fold gene IDs and seed0 probe order; truth retains NaNs.
- [ ] `results/pilot/context_shift/verification.json`: independently read saved predictions, reconstruct every checkpoint correction directly per gene, verify k0 equality, k1 observed residual identity, and direct masked metric calculations; verify all seed-average rows.

For each primary fold reuse its exact selected gene indices, training means and target/probe split. For acquired probes S, correction of gene g is the arithmetic mean of `y[p,g]-mu[p,g]` over finite acquired observations only. If S has no finite observations of g, the correction is zero. Predict every target q as `mu[q,g]+correction[g]`. Targets never enter correction. No shrinkage, weighting or extra hyperparameters. Read the exact primary random selection order, check it against `default_rng(seed).permutation(probes)`, and use the same fixed target observation masks and primary denominator. Missing targets are excluded, never filled with zero ground truth.

Validation covers the actual ten contexts/known drugs/selected genes supplied by the primary run, not unseen drugs, patients, absolute expression or other dose/time regimes. Primary V-optimal comparison mixes a different predictor and a different probe order; primary random comparison uses matched random orders and isolates the adaptation model. Shared-training-fold bootstrap intervals are descriptive, not independent-cohort population inference.

## Completion and verification

All deliverables above were produced. No primary code, primary results, or input files were modified. The script was run once with no tuning; the manifest checks primary output hashes before execution and the independent verifier checks them again afterwards. The new script records 1,400 per-seed metric rows and 70 seed-averaged context/budget rows.

Exact commands, both **exit 0**:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/context_shift_baseline.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/context_shift_baseline.py --verify-only
```

The default run also performs verification and saves `verification.json`. Read-only verification independently reconstructs **30 full checkpoints** (k0/k1/k5, random seed0, ten contexts) using direct per-gene averaging and full masked target differences, not the streaming quadratic scorer. It confirms k0 equality for all 200 context/seed combinations, exact observed residual correction at k1, primary hashes unchanged, and all seed-to-context averages. Maximum absolute discrepancy in reconstructed MSE is **4.440892098500626e-16**. The run checks k1 correction for every context/seed as well.

## Observed exploratory results

Numbers below come from the saved end-to-end script and `results/pilot/context_shift/summary.json`. Random seeds are averaged within each context before context aggregation.

| Probe budget | Context-shift mean normalized MSE |
|---:|---:|
| 0 | 1.000000 |
| 1 | 1.630434 |
| 2 | 1.225142 |
| 3 | 1.082470 |
| 5 | 0.967533 |
| 10 | 0.886650 |
| 20 | 0.844995 |

The unregularized generic offset is **substantially worse at low budgets** (k1–k3 context-average error exceeds the unadapted baseline), but improves at higher budgets. At k5 its exact mean normalized MSE is **0.9675332989464966**, compared with **0.994054748021391** for primary random and **0.9949672557213292** for primary V-optimal. Paired mean gain `primary random − context shift` is **0.02652144907489453**, descriptive 95% interval **[0.00199493425129733, 0.05726817072759071]**. The corresponding V-optimal gain is **0.027433956774832513**, interval **[0.002469249054065693, 0.0583295268817687]**. Individual contexts can still worsen at k5: context-average normalized MSE ranges **0.8669147646242825–1.0259497424512591**. No significance or biological-generalization claims are made.

This exploratory result indicates that a generic per-gene context offset is an important comparator: at five or more probes, the primary low-rank model's small adaptation gains are not evidence that its representation is necessary or superior. This does not establish clinical performance or invalidate the primary selection-policy comparison; the latter held its predictor fixed. No post-result shrinkage, alternative budget selection, or parameter tuning was attempted.

Files touched: `scripts/context_shift_baseline.py`, this note, and `results/pilot/context_shift/{manifest.json,metrics.csv,per_context.csv,summary.json,verification.json,checkpoints/context_000_seed0.npz,...,checkpoints/context_009_seed0.npz}`. The verification JSON is written after the producer manifest and is not included among producer output hashes. The manifest retains exact primary/input/source-script hashes and software versions.
