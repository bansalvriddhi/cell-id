# CELL-ID pilot implementation contract

## Approved mask-aware amendment, frozen before empirical scores

The new actual-data brief supersedes the complete-tensor assumptions below. Input is now `data/processed_v1/responses.npz`: `log2fc`, `lfc_se`, and Boolean `observed` arrays `(C,D,G)`, plus `contexts`, `drugs`, and `gene_names`. `observed` must equal `isfinite(log2fc)`; infinities are rejected. `lfc_se` is shape-checked but is not used as calibrated uncertainty. No normalization is added. Scope is completion of author-provided log2FC signatures at 0.05 µM, plate 1; exposure duration and source DE methods remain unverified as detailed in `notes/data_provenance.md`.

Before each outer **and inner** fit, retain only genes finite across **every training-context × drug** profile, then select up to 1000 by training residual variance. Held-out numerical labels and held-out missingness do not participate in eligibility, ranking, means, factors, or noise. All selected training entries are finite; the effective noise definition otherwise remains unchanged.

Each candidate design score uses its nominal full-selected-gene Gram matrix. After selecting a probe, reveal its finite rows and add **only their** Gram matrix and response sufficient statistic. Later selections can depend on acquired missingness through covariance, but never on response values or unrevealed missingness. V-optimal's design Q remains the nominal full-gene target Gram, not an evaluation-mask-aware Q. This is covariance design with missingness feedback, not a learned outcome-adaptive policy.

The scoring mask is `isfinite(held_out_target_log2fc)` on selected genes, fixed across policies and budgets. Primary MSE pools **observed target-gene entries** and normalizes by the same masked unadapted MSE. Pearson is computed separately for each target using its observed genes; fewer than two observations or constant vectors yield missing correlations. Numerical zero contributions inside masked quadratic algebra are not imputed ground truth; saved truth preserves NaNs. No target observations or no eligible training genes are explicit errors, not fabricated scores.

Additional output checklist:

- [ ] `metrics.csv`: add `target_observed_entries,target_possible_entries,target_coverage,acquired_missing_entries`.
- [ ] `selections.csv`: add per-acquisition `observed_genes,missing_genes`.
- [ ] `coverage.csv`: `context,target,observed_genes,selected_genes,coverage` for the fixed target panel.
- [ ] Model JSON adds `training_eligible_gene_count`; checkpoint NPZ adds `target_observed` and `selected_probe_observed` masks while retaining NaNs in truth.
- [ ] Sentinel changes both hidden target numbers and missingness, then refits/tunes and replays designs without changing fitted values, selections, or predictions.
- [ ] Independent checker verifies masked reconstruction and coverage; new unit checks cover partially/all-missing probes, masked targets, training-only eligibility, and unseen-mask/value isolation.
- [ ] Run actual data once to a fresh `results/pilot/main`, then independently verify. Retain source data untouched. With ten contexts, all shared-training-fold bootstrap intervals are descriptive only.

The earlier checklist and trace below are preserved as implementation history; this amendment governs actual-data execution.

## Deliverables frozen before computation

- [x] `scripts/run_pilot.py`: CPU-only NumPy command-line experiment; default input `data/pilot_tensor.npz`, default new output directory `results/pilot/main`; refuse any existing output directory. No data acquisition, normalization, cloud jobs, or external APIs.
- [x] `scripts/check_pilot.py`: independent saved-result reconstruction and ephemeral in-memory unit checks (`--self-test`); unit checks are not empirical evidence and write no empirical tables.
- [ ] `results/pilot/<run>/manifest.json`: exact input SHA-256, input dimensions, full identifier-only split, configuration, Python/NumPy/platform/thread details, script/protocol hashes, and completion state.
- [ ] `results/pilot/<run>/metrics.csv`: `context,policy,seed,budget,mse,baseline_mse,normalized_mse,pearson_mean,pearson_valid_targets,target_count,gene_count`; deterministic/baseline seed = -1, random seeds = 0..19; float64 values serialized without presentation rounding; undefined Pearson is an empty CSV cell.
- [ ] `results/pilot/<run>/selections.csv`: `context,policy,seed,step,perturbation`; acquisition order, one row per distinct probe, no rows for unadapted.
- [ ] `results/pilot/<run>/per_target_pearson.csv`: `context,policy,seed,budget,target,pearson`; Pearson across selected genes separately for each fixed target; constant vectors yield an explicitly missing coefficient, not zero.
- [ ] `results/pilot/<run>/per_context.csv`: random-seed-averaged context scores, `context,policy,budget,n_seeds,mse,baseline_mse,normalized_mse,pearson_mean`.
- [ ] `results/pilot/<run>/context_auc.csv`: `context,policy,normalized_auc`; trapezoid of context normalized MSE versus actual recorded budgets, divided by maximum recorded budget.
- [ ] `results/pilot/<run>/summary.json`: context-averaged scores, context-paired gains against random and unadapted, context AUCs, uncertainty method and limitations. With >=8 contexts use 2000 paired resamples, RNG seed 77, percentile 95% intervals; otherwise context minimum/maximum only. No significance claims.
- [ ] `results/pilot/<run>/models/context_NNN.{npz,json}`: selected original gene indices and IDs, training context IDs, fitted mean `(P,G_selected)`, loadings `(P,G_selected,r)`, small Gram matrices `(P,r,r)`, eigenvalues, effective noise variance, rank/noise tuning scores and numerical safeguards.
- [ ] `results/pilot/<run>/predictions/context_NNN_k5.npz`: real full target truth `(T,G_selected)`, unadapted predictions of the same shape, and predictions `(4,T,G_selected)` for trace-uncertainty, D-optimal, V-optimal, and random seed 0 at k=5; also identifiers, acquired probe indices, posterior means/covariances.
- [ ] `results/pilot/<run>/checks.json`: probe/target disjointness, distinct acquisition counts, common k=0, covariance PSD/monotonicity, real-data held-out-target sentinel checks. Final independent verification is printed by `check_pilot.py`.

## Validation domain and non-claims

The executable accepts a complete finite floating response tensor `(C,P,G)` plus unique nonempty string arrays `contexts`, `perturbations`, and `genes`. Responses must already be normalized delta/control-relative signatures. It requires >=2 contexts, >=20 perturbations, and >=1 gene. The pipeline does not infer whether the assay normalization, contexts, doses, or biological replicates are suitable; acquisition provenance remains a separate prerequisite. Evaluation is leave-one-context-out for known perturbations at the supplied conditions, on a metadata-only fixed disjoint target panel and training-selected genes. Unit checks exercise algebra, degeneracy, perturbations, and input guards; they cannot establish predictive accuracy on real contexts.

## Numerical and design decisions

1. Sort perturbation identifiers, permute using NumPy `default_rng(20261002)`, assign the first floor(P/2) identifiers to probes and the remainder to targets, then sort each panel by identifier. Require at least ten in each. Contexts are evaluated in identifier order. Budgets are the subset of `[0,1,2,3,5,10,20,50]` no larger than the probe pool (not a new clipped endpoint).
2. For each training split, rank genes by mean-over-perturbations sample variance across training contexts, descending; ties use gene identifier. Keep <=1000 genes, retaining all if fewer. This uses the centered context residual variance and is refitted within every inner split as well as each outer split.
3. Fit the perturbation-specific training mean. For centered training profiles X, the loading factor is `X.T @ U_r / sqrt(Ctrain-1)`, using the context Gram eigendecomposition (equivalent to truncated SVD). The default requested rank is `min(Ctrain-1,8)`; remove only numerically zero eigenmodes. A single training context yields rank zero. Canonicalize eigenvector signs.
4. Effective observation variance is the multiplier times the median, over selected genes, of the mean-over-perturbations training sample variance across contexts. This is **pre-truncation centered residual variance**, not the remaining low-rank reconstruction error (which can vanish at full rank). Default multiplier 1.0. A documented scale-relative floating-point floor handles zero variance; it is not a biological noise estimate.
5. If the original tensor has >=8 contexts, perform full inner leave-one-context-out validation using outer-training data only. Candidate ranks `[2,4,8]` are capped at the inner available algebraic rank; multipliers `[0.1,1,10]`. Tune mean context-normalized MSE using k=5 random probes and seeds `[0,1,2]`. Inner zero-baseline cases with no target variation cannot define normalization and are explicitly omitted from tuning with counts; if all inner baselines vanish retain defaults. Tie-break by smaller rank, then smaller multiplier. Outer test responses never tune these choices.
6. `trace-uncertainty` maximizes `tr(Sigma A[p].T A[p])`; `D-optimal` maximizes `logdet(I + L.T (A[p].T A[p]/tau2) L)`, where `Sigma=L L.T`; `V-optimal` maximizes `tr(Q (Sigma-Sigma_new))` for Q summed across the fixed targets. Ties choose lexicographically first perturbation (numerical near-ties within 1e-12 relative are treated as ties). These orders are covariance-only and outcome-independent. Random uses each seed's uniform permutation of the canonical probe list, shared across contexts for pairing.
7. Precompute drug Gram matrices and eligible-probe sufficient statistics. Although these probe statistics are cached, only add the statistic corresponding to an acquired probe to the posterior; no statistics or labels enter covariance-only selection. Solve the accumulated small precision system directly for each checkpoint posterior mean (rather than multiply an explicit inverse). Evaluate MSE with target quadratic terms; evaluate target-wise correlations with precomputed gene-centered quadratic terms at recorded budgets. No repeated full target-gene arithmetic per seed. Save full predictions at k=5 (always available under the >=10-probe requirement), irrespective of the maximum budget.
8. Normalize within each context by its training-mean MSE on its own selected genes and fixed targets. A zero outer baseline makes this primary endpoint undefined, so fail explicitly rather than substitute fabricated normalized values. Constant prediction/truth vectors have undefined Pearson and are counted/reported as such. Squared-error units are squared input-response units; normalized MSE/AUC and Pearson are dimensionless.
9. Use two threads via BLAS/OpenMP environment caps set before importing NumPy. Preserve source and script checksums. Write only into a fresh output directory; errors leave a manifest marked failed and preserve all partial real work. No significance tests and no claim of generalization to patients, unobserved doses/times, or unseen perturbations.

## Execution trace

Implementation begun with `data/pilot_tensor.npz` absent in the initial directory inventory. Data acquisition is independent; no source-data searches/downloads are part of this branch. Verification and any real run will be appended here after they complete.

### Implementation verification

`PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_pilot.py --self-test` passed **10 tests**, exit code **0**. These are implementation checks, not estimates of biological prediction performance. They cover:

- Input rejection (missingness/nonfinite values, wrong dtype, too few perturbations, duplicate identifiers), identifier-order-invariant metadata split, and output-overwrite refusal.
- Factor covariance against directly computed empirical covariance; small-matrix acquisition objectives against full gene-space conditioning and determinants.
- Target quadratic MSE and Pearson against full vector calculations at every acquisition step, with response scales `1e-5`, `1`, and `1e5` and non-factor/nonlinear perturbations.
- Rank-zero, one-training-context, one-gene/undefined-Pearson, identical-context, and lexical-tie cases; gene cap >1000; rank 8 and 50 acquisitions with strongly heterogeneous gene scales.
- Nested inner-context tuning and target-label sentinel refitting, including unchanged acquisition orders and posterior predictions.
- Full producer serialization **and the independent reader/checker** on in-memory virtual files, exercising 3-context descriptive, 8-context nested/bootstrap, and 2-context rank-zero branches. Nothing from these examples is written to the actual filesystem.
- Random-seed averaging before context aggregation, normalized trapezoidal AUC, and paired bootstrap/min-max branches.

The original degeneracy check failed because averaging identical floating vectors left a tiny residual constant across contexts; an explicit second residual-centering pass now prevents a spurious nonzero mode. A deliberately ill-conditioned stress check also found inverse-times-vector posterior arithmetic less accurate than a direct solve; the implementation was changed to solve the accumulated precision system. The original assertions were retained and now pass. Neither correction used held-out empirical scores.

### Commands for the lead

Run from the project working folder, using a Python 3 environment with NumPy installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_pilot.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_pilot.py --input data/pilot_tensor.npz --output results/pilot/main
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_pilot.py --run results/pilot/main
```

The independent checker reads the original input named in the manifest (or `--input` for an exact-checksum relocated copy), validates output hashes/row coverage/splits/selections, checks training-selected genes and direct-SVD covariance, reconstructs every saved k=5 prediction and MSE/Pearson, and independently recalculates context averages, AUC, and uncertainty. It does not import producer functions when checking real results. Both CLIs provide `--help`. Existing result directories are never reused, resumed, deleted, or overwritten; choose a fresh `--output` for any later authorized rerun.

Output checkboxes above refer to **actual-data** products and remain unchecked until a real run exists. The memory-only tests verify the schema but do not fulfill the empirical-results requirement. Rank/noise tuning is enabled only for >=8 original contexts; no hyperparameter tuning is performed on smaller samples. The only interpretive numerical choice not explicit in the original protocol is the pre-truncation centered-variance definition of the effective noise scale, stated above; it avoids artificial zero noise when all train-context modes are retained.

### Handoff state

At the single final input-availability check, `data/pilot_tensor.npz` was still absent. The conditional launch therefore did not run the empirical experiment and created no `results/pilot/` tables. Both `--help` commands and the complete self-test command exited 0. The lead can execute the two real-data commands above when acquisition supplies the input. No data files, other scripts, or report files were modified by this implementation branch.
