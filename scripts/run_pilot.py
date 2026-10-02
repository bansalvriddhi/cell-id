#!/usr/bin/env python3
"""Reproducible, CPU-only CELL-ID linear covariance design pilot.

Input: author log2fc[C,P,G], observed[C,P,G], lfc_se[C,P,G], contexts[C],
drugs[P], gene_names[G]. Missing LFCs stay NaN. See notes/implementation.md.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("unadapted", "random", "trace-uncertainty", "D-optimal", "V-optimal")
DETERMINISTIC = POLICIES[2:]
CHECKPOINT_POLICIES = (*DETERMINISTIC, "random")
BUDGETS = (0, 1, 2, 3, 5, 10, 20, 50)
SPLIT_SEED = 20261002
RANDOM_SEEDS = tuple(range(20))
INNER_SEEDS = (0, 1, 2)
NOISE_GRID = (0.1, 1.0, 10.0)
METRIC_FIELDS = ("context", "policy", "seed", "budget", "mse", "baseline_mse",
                 "normalized_mse", "pearson_mean", "pearson_valid_targets",
                 "target_count", "gene_count", "target_observed_entries",
                 "target_possible_entries", "target_coverage", "acquired_missing_entries")
CONTEXT_FIELDS = ("context", "policy", "budget", "n_seeds", "mse", "baseline_mse",
                  "normalized_mse", "pearson_mean")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(clean_json(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_json(row[k]) for k in fields})


def validate_arrays(responses, contexts, perturbations, genes):
    if responses.ndim != 3 or responses.dtype.kind != "f":
        raise ValueError("responses must be a floating array of shape (C,P,G)")
    c, p, g = responses.shape
    if c < 2 or p < 20 or g < 1:
        raise ValueError("need C>=2, P>=20 (>=10 probes and >=10 targets), G>=1")
    if np.isinf(responses).any():
        raise ValueError("infinite responses are unsupported; missing entries must be NaN")
    for name, ids, length in (("contexts", contexts, c),
                               ("perturbations", perturbations, p), ("genes", genes, g)):
        if ids.shape != (length,) or ids.dtype.kind not in "US":
            raise ValueError(f"{name} must be a one-dimensional non-object string array")
        text = ids.astype(str)
        if any(not v.strip() for v in text) or len(set(text)) != length:
            raise ValueError(f"{name} must contain unique nonempty identifiers")


def load_input(path):
    with np.load(path, allow_pickle=False) as source:
        missing = {"log2fc", "observed", "lfc_se", "contexts", "drugs", "gene_names"} - set(source.files)
        if missing:
            raise ValueError(f"missing input keys: {sorted(missing)}")
        arrays = [source[key] for key in ("log2fc", "contexts", "drugs", "gene_names")]
        mask = source["observed"]
        if mask.dtype.kind != "b" or mask.shape != arrays[0].shape:
            raise ValueError("observed must be Boolean with the log2fc shape")
        if not np.array_equal(mask, np.isfinite(arrays[0])):
            raise ValueError("observed must equal isfinite(log2fc)")
        if source["lfc_se"].shape != arrays[0].shape:
            raise ValueError("lfc_se shape must match log2fc; uncertainty values are not used")
    validate_arrays(*arrays)
    return arrays[0].astype(np.float64, copy=False), *(x.astype(str) for x in arrays[1:])


def split_perturbations(ids):
    canonical = np.argsort(ids, kind="stable")
    permuted = np.random.default_rng(SPLIT_SEED).permutation(canonical)
    half = len(ids) // 2
    probes, targets = permuted[:half], permuted[half:]
    probes = probes[np.argsort(ids[probes], kind="stable")]
    targets = targets[np.argsort(ids[targets], kind="stable")]
    if min(len(probes), len(targets)) < 10:
        raise ValueError("split requires at least 10 probes and 10 targets")
    assert not set(probes) & set(targets)
    return probes, targets


@dataclass
class Model:
    gene_indices: np.ndarray
    mu: np.ndarray
    A: np.ndarray
    grams: np.ndarray
    eigenvalues: np.ndarray
    residual_variance_median: float
    variance_floor: float
    noise_multiplier: float
    tau2: float
    requested_rank: int
    train_count: int
    eligible_count: int

    @property
    def rank(self):
        return self.A.shape[-1]


def fit_model(train, genes, requested_rank=None, noise_multiplier=1.0):
    """Only training-context responses enter feature selection and factor fitting."""
    n = len(train)
    if n < 1:
        raise ValueError("empty training contexts")
    eligible = np.flatnonzero(np.isfinite(train).all(axis=(0, 1)))
    if not len(eligible):
        raise ValueError("no genes finite across all training context x drug profiles")
    finite_train = train[:, :, eligible]
    eligible_variance = (np.var(finite_train, axis=0, ddof=1).mean(axis=0) if n > 1
                         else np.zeros(len(eligible)))
    variances = np.full(len(genes), np.nan)
    variances[eligible] = eligible_variance
    selected = eligible[np.lexsort((genes[eligible], -eligible_variance))[:1000]]
    x = np.asarray(train[:, :, selected], dtype=np.float64)
    mu = x.mean(axis=0)
    residual = (x - mu).reshape(n, -1)
    # Remove centering roundoff; identical training contexts must have rank zero.
    residual -= residual.mean(axis=0, keepdims=True)
    available = min(n - 1, 8)
    requested = available if requested_rank is None else min(int(requested_rank), available)
    if requested < 0 or noise_multiplier <= 0:
        raise ValueError("rank must be nonnegative and noise multiplier positive")
    gram = residual @ residual.T
    eigenvalues, u = np.linalg.eigh((gram + gram.T) * 0.5)
    eigenvalues, u = eigenvalues[::-1], u[:, ::-1]
    threshold = (np.finfo(float).eps * max(residual.shape) *
                 max(float(eigenvalues[0]), np.finfo(float).tiny))
    rank = min(requested, int(np.sum(eigenvalues > threshold)))
    u = u[:, :rank].copy()
    for j in range(rank):
        pivot = np.argmax(np.abs(u[:, j]))
        if u[pivot, j] < 0:
            u[:, j] *= -1
    factor = (residual.T @ u / np.sqrt(n - 1) if n > 1
              else np.empty((residual.shape[1], 0)))
    A = factor.reshape(*mu.shape, rank)
    grams = np.einsum("pgr,pgs->prs", A, A, optimize=True)
    median = float(np.median(variances[selected]))
    # A relative safeguard, including a representable floor for all-zero inputs.
    scale = max(float(np.mean(x * x)), float(np.max(variances[selected])), 1e-12)
    floor = max(1e-12 * scale, np.finfo(float).tiny)
    tau2 = max(float(noise_multiplier) * median, floor)
    if not np.isfinite(A).all() or not np.isfinite(tau2):
        raise ValueError("nonfinite model arithmetic; input response magnitudes are unsupported")
    return Model(selected, mu, A, grams, eigenvalues[:rank] / max(n - 1, 1),
                 median, floor, float(noise_multiplier), tau2, requested, n, len(eligible))


def change_hyperparameters(model, rank, multiplier):
    r = min(int(rank), model.rank)
    return Model(model.gene_indices, model.mu, model.A[:, :, :r],
                 model.grams[:, :r, :r], model.eigenvalues[:r],
                 model.residual_variance_median, model.variance_floor, multiplier,
                 max(multiplier * model.residual_variance_median, model.variance_floor),
                 int(rank), model.train_count, model.eligible_count)


def inverse_spd(precision):
    value = np.linalg.solve(precision, np.eye(len(precision)))
    return (value + value.T) * 0.5


def check_covariance(old, new):
    if not len(new):
        return 0.0
    tolerance = 1e-9 * max(1.0, float(np.linalg.norm(old, 2)))
    if not np.allclose(new, new.T, rtol=0, atol=tolerance):
        raise AssertionError("posterior covariance not symmetric")
    if np.linalg.eigvalsh(new)[0] < -tolerance:
        raise AssertionError("posterior covariance not PSD")
    minimum = float(np.linalg.eigvalsh(old - new)[0])
    if minimum < -tolerance:
        raise AssertionError("conditional covariance increased")
    return minimum


def plan_order(model, probes, targets, policy, count, seed=-1, reveal=None):
    """Use nominal candidate Gram; only reveal chosen probe's observed-row Gram."""
    if policy not in (*DETERMINISTIC, "random"):
        raise ValueError(f"unknown policy {policy}")
    if not 0 <= count <= len(probes):
        raise ValueError("invalid acquisition count")
    precision = np.eye(model.rank)
    covariance = np.eye(model.rank)
    weighted = model.grams / model.tau2
    Q = model.grams[targets].sum(axis=0)
    remaining = list(map(int, probes))
    random_order = (np.random.default_rng(seed).permutation(probes)
                    if policy == "random" else None)
    order, covariances = [], [covariance.copy()]
    minimum_reduction = 0.0
    for step in range(count):
        if policy == "random":
            chosen = int(random_order[step])
        else:
            scores = []
            cholesky = (np.linalg.cholesky(covariance)
                        if policy == "D-optimal" else None)
            for p in remaining:
                if policy == "trace-uncertainty":
                    score = np.trace(covariance @ model.grams[p])
                elif policy == "D-optimal":
                    matrix = np.eye(model.rank) + cholesky.T @ weighted[p] @ cholesky
                    sign, score = np.linalg.slogdet(matrix)
                    if sign <= 0:
                        raise ArithmeticError("nonpositive D-optimal determinant")
                else:
                    candidate = inverse_spd(precision + weighted[p])
                    score = np.trace(Q @ (covariance - candidate))
                scores.append(float(score))
            best = max(scores)
            tolerance = 1e-12 * max(abs(best), np.finfo(float).tiny)
            chosen = remaining[next(i for i, s in enumerate(scores) if s >= best - tolerance)]
        remaining.remove(chosen)
        actual_gram = model.grams[chosen] if reveal is None else reveal(chosen)
        precision = precision + actual_gram / model.tau2
        new = inverse_spd(precision)
        minimum_reduction = min(minimum_reduction, check_covariance(covariance, new))
        covariance = new
        order.append(chosen)
        covariances.append(covariance.copy())
    assert len(set(order)) == count and not set(order) & set(targets)
    return np.asarray(order, dtype=int), np.asarray(covariances), minimum_reduction


class ProbeStatistics:
    """Evaluator-owned lazy oracle. Candidate policies receive only gram(chosen).

    Caches observed-row calculations across seeds, without exposing unqueried masks.
    No target labels/masks enter this object.
    """
    def __init__(self, model, probes, responses):
        if responses.shape != model.mu[probes].shape:
            raise ValueError("probe response shape does not match selected genes")
        self.model = model
        self._responses = {int(p): responses[i] for i, p in enumerate(probes)}
        self._cache = {}

    def _reveal(self, p):
        p = int(p)
        if p not in self._cache:
            y = self._responses[p]
            mask = np.isfinite(y)
            a = self.model.A[p, mask]
            gram = a.T @ a
            natural = a.T @ (y[mask] - self.model.mu[p, mask]) / self.model.tau2
            self._cache[p] = (gram, natural, mask)
        return self._cache[p]

    def gram(self, p):
        return self._reveal(p)[0]

    def __getitem__(self, p):
        return self._reveal(p)[1]

    def mask(self, p):
        return self._reveal(p)[2]


def probe_statistics(model, probes, responses):
    return ProbeStatistics(model, probes, responses)


def posterior_means(model, order, covariances, stats, budgets):
    natural = np.zeros(model.rank)
    precision = np.eye(model.rank)
    if len(covariances) != len(order) + 1:
        raise ValueError("covariance path does not match acquisition order")
    result = {0: np.zeros(model.rank)}
    for step, probe in enumerate(order, start=1):
        natural = natural + stats[int(probe)]
        precision = precision + stats.gram(int(probe)) / model.tau2
        if step in budgets:
            # Solving directly is more stable than multiplying an explicitly formed inverse.
            result[step] = np.linalg.solve(precision, natural)
    return result


def predict(model, targets, latent_mean):
    return model.mu[targets] + np.einsum("pgr,r->pg", model.A[targets], latent_mean,
                                        optimize=True)


class TargetMetrics:
    """Truth is confined to scoring, never fitting, design, or adaptation."""
    def __init__(self, model, targets, truth):
        mu, A = model.mu[targets], model.A[targets]
        self.mask = np.isfinite(truth)
        self.counts = self.mask.sum(axis=1)
        self.n = int(self.mask.sum())
        if self.n == 0:
            raise ValueError("no observed target-gene entries on training-selected genes")
        # Zero algebraic contributions outside the mask, never zero-filled truth.
        delta = np.zeros_like(truth)
        delta[self.mask] = (truth - mu)[self.mask]
        self.c0 = float(np.sum(delta * delta))
        self.b = np.einsum("pgr,pg->r", A, delta, optimize=True)
        amasked = A * self.mask[:, :, None]
        self.Q = np.einsum("pgr,pgs->rs", amasked, amasked, optimize=True)
        self.baseline_mse = self.c0 / self.n
        muc, ac, yc = np.zeros_like(mu), np.zeros_like(A), np.zeros_like(truth)
        for t, mask in enumerate(self.mask):
            if mask.any():
                muc[t, mask] = mu[t, mask] - mu[t, mask].mean()
                ac[t, mask] = A[t, mask] - A[t, mask].mean(axis=0)
                yc[t, mask] = truth[t, mask] - truth[t, mask].mean()
        self.y2 = np.einsum("pg,pg->p", yc, yc)
        self.mu2 = np.einsum("pg,pg->p", muc, muc)
        self.muy = np.einsum("pg,pg->p", muc, yc)
        self.ay = np.einsum("pgr,pg->pr", ac, yc, optimize=True)
        self.amu = np.einsum("pgr,pg->pr", ac, muc, optimize=True)
        self.acac = np.einsum("pgr,pgs->prs", ac, ac, optimize=True)

    def evaluate(self, mean):
        linear, quadratic = float(2 * self.b @ mean), float(mean @ self.Q @ mean)
        sse = self.c0 - linear + quadratic
        tolerance = 1e-10 * max(self.c0, abs(linear), abs(quadratic), 1e-300)
        if sse < -tolerance:
            raise ArithmeticError("negative quadratic squared error beyond roundoff")
        mse = max(0.0, sse) / self.n
        p2 = (self.mu2 + 2 * self.amu @ mean +
              np.einsum("r,prs,s->p", mean, self.acac, mean, optimize=True))
        covariance = self.muy + self.ay @ mean
        threshold = 1e-13 * np.maximum(self.mu2 + np.abs(p2), 1e-300)
        valid = (p2 > threshold) & (self.y2 > 0) & (self.counts >= 2)
        correlations = np.full(len(p2), np.nan)
        correlations[valid] = np.clip(covariance[valid] /
                                      np.sqrt(p2[valid] * self.y2[valid]), -1, 1)
        average = float(correlations[valid].mean()) if valid.any() else float("nan")
        normalized = mse / self.baseline_mse if self.baseline_mse > 0 else float("nan")
        return mse, normalized, average, int(valid.sum()), correlations


def tune(outer_train, genes, probes, targets, contexts):
    """Nested context validation. Outer held-out data are not an argument."""
    available = min(len(outer_train) - 2, 8)
    ranks = sorted({min(r, available) for r in (2, 4, 8)})
    records = {(r, m): [] for r in ranks for m in NOISE_GRID}
    omitted = []
    for holdout in range(len(outer_train)):
        inner_train = np.delete(outer_train, holdout, axis=0)
        base = fit_model(inner_train, genes)
        observed = outer_train[holdout][:, base.gene_indices]
        for r, multiplier in records:
            model = change_hyperparameters(base, r, multiplier)
            scoring = TargetMetrics(model, targets, observed[targets])
            if scoring.baseline_mse == 0:
                if str(contexts[holdout]) not in omitted:
                    omitted.append(str(contexts[holdout]))
                continue
            stats = probe_statistics(model, probes, observed[probes])
            scores = []
            for seed in INNER_SEEDS:
                order, covariances, _ = plan_order(model, probes, targets, "random", 5, seed,
                                                  reveal=stats.gram)
                mean = posterior_means(model, order, covariances, stats, (5,))[5]
                scores.append(scoring.evaluate(mean)[1])
            records[(r, multiplier)].append({"context": str(contexts[holdout]),
                                             "normalized_mse": float(np.mean(scores))})
    candidates = [{"rank": r, "noise_multiplier": m, "folds": folds,
                   "mean_normalized_mse": float(np.mean([f["normalized_mse"] for f in folds]))
                   if folds else None} for (r, m), folds in records.items()]
    usable = [x for x in candidates if x["mean_normalized_mse"] is not None]
    if usable:
        best = min(usable, key=lambda x: (x["mean_normalized_mse"], x["rank"],
                                          x["noise_multiplier"]))
        selected_rank, selected_multiplier = best["rank"], best["noise_multiplier"]
    else:
        selected_rank, selected_multiplier = min(len(outer_train) - 1, 8), 1.0
    return selected_rank, selected_multiplier, {
        "method": "inner leave-one-context-out, random k=5, seed-then-context mean",
        "seeds": INNER_SEEDS, "ranks": ranks, "noise_multipliers": NOISE_GRID,
        "candidates": candidates, "omitted_zero_baseline_contexts": omitted,
        "fallback_to_defaults": not bool(usable)}


def select_and_fit(train, genes, probes, targets, train_contexts, tune_enabled):
    if tune_enabled:
        rank, multiplier, tuning = tune(train, genes, probes, targets, train_contexts)
    else:
        rank, multiplier = min(len(train) - 1, 8), 1.0
        tuning = {"method": "prespecified defaults; fewer than eight total contexts"}
    return fit_model(train, genes, rank, multiplier), tuning


def sentinel_check(data, holdout, train_indices, genes, probes, targets, contexts,
                   model, tune_enabled, deterministic_orders):
    """Alter hidden numerical targets AND missingness; refit/tune/replay designs."""
    altered = data.copy()
    original = altered[holdout, targets, :].copy()
    changed = np.where(np.isfinite(original), np.where(original == 0, 1e6, -3 * original), 12345.)
    changed[:, ::2] = np.where(np.isfinite(original[:, ::2]), np.nan, 54321.)
    altered[holdout, targets, :] = changed
    mask_changes = int(np.sum(np.isfinite(original) != np.isfinite(changed)))
    assert mask_changes > 0
    refit, _ = select_and_fit(altered[train_indices], genes, probes, targets,
                            contexts[train_indices], tune_enabled)
    for name in ("gene_indices", "mu", "A", "grams", "eigenvalues"):
        np.testing.assert_array_equal(getattr(model, name), getattr(refit, name))
    assert model.tau2 == refit.tau2 and model.requested_rank == refit.requested_rank
    original_probes = data[holdout][probes][:, model.gene_indices]
    altered_probes = altered[holdout][probes][:, refit.gene_indices]
    stats1 = probe_statistics(model, probes, original_probes)
    stats2 = probe_statistics(refit, probes, altered_probes)
    for policy in (*DETERMINISTIC, "random"):
        seed = 0 if policy == "random" else -1
        count = len(deterministic_orders[DETERMINISTIC[0]])
        o1, c1, _ = plan_order(model, probes, targets, policy, count, seed, reveal=stats1.gram)
        o2, c2, _ = plan_order(refit, probes, targets, policy, count, seed, reveal=stats2.gram)
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(c1, c2)
        m1 = posterior_means(model, o1, c1, stats1, range(count + 1))
        m2 = posterior_means(refit, o2, c2, stats2, range(count + 1))
        for budget in range(count + 1):
            np.testing.assert_array_equal(m1[budget], m2[budget])
        np.testing.assert_array_equal(predict(model, targets, m1[5]),
                                      predict(refit, targets, m2[5]))
    return {"target_values_changed": int(original.size), "target_mask_entries_changed": mask_changes,
            "fit_unchanged": True,
            "all_deterministic_and_random0_orders_unchanged": True,
            "posterior_means_all_steps_unchanged": True, "k5_predictions_unchanged": True}


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")


def aggregate(rows, contexts, budgets):
    per_context, auc = [], []
    grouped = {}
    for row in rows:
        grouped.setdefault((row["context"], row["policy"], row["budget"]), []).append(row)
    for context in contexts:
        for policy in POLICIES:
            curve = []
            for budget in budgets:
                group = grouped[(str(context), policy, budget)]
                result = {"context": str(context), "policy": policy, "budget": budget,
                          "n_seeds": len(group)}
                for metric in ("mse", "baseline_mse", "normalized_mse", "pearson_mean"):
                    result[metric] = finite_mean([x[metric] for x in group])
                per_context.append(result)
                curve.append(result["normalized_mse"])
            trapezoid = sum((curve[i] + curve[i + 1]) * (budgets[i + 1] - budgets[i]) / 2
                            for i in range(len(budgets) - 1))
            auc.append({"context": str(context), "policy": policy,
                        "normalized_auc": float(trapezoid / budgets[-1])})
    return per_context, auc


def summarize(per_context, auc, contexts, budgets):
    count = len(contexts)
    bootstrap = (np.random.default_rng(77).integers(0, count, size=(2000, count))
                 if count >= 8 else None)

    def describe(values):
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values)
        result = {"mean": finite_mean(values), "n_contexts": int(valid.sum())}
        if valid.any():
            result.update({"context_min": float(values[valid].min()),
                           "context_max": float(values[valid].max())})
        if bootstrap is not None and valid.all():
            limits = np.quantile(values[bootstrap].mean(axis=1), [0.025, 0.975])
            result["paired_context_bootstrap_95pct"] = limits.tolist()
        return result

    lookup = {(x["context"], x["policy"], x["budget"]): x for x in per_context}
    scores, comparisons = [], []
    for budget in budgets:
        for policy in POLICIES:
            group = [lookup[(str(c), policy, budget)] for c in contexts]
            scores.append({"policy": policy, "budget": budget,
                           "normalized_mse": describe([x["normalized_mse"] for x in group]),
                           "mse": describe([x["mse"] for x in group]),
                           "pearson_mean": describe([x["pearson_mean"] for x in group])})
            for reference in ("random", "unadapted"):
                if reference != policy:
                    gain = [lookup[(str(c), reference, budget)]["normalized_mse"] -
                            lookup[(str(c), policy, budget)]["normalized_mse"] for c in contexts]
                    comparisons.append({"policy": policy, "reference": reference,
                                        "budget": budget, "gain_reference_minus_policy": describe(gain)})
    auc_lookup = {(x["context"], x["policy"]): x["normalized_auc"] for x in auc}
    auc_scores, auc_comparisons = [], []
    for policy in POLICIES:
        auc_scores.append({"policy": policy, "normalized_auc":
                           describe([auc_lookup[(str(c), policy)] for c in contexts])})
        for reference in ("random", "unadapted"):
            if policy != reference:
                auc_comparisons.append({"policy": policy, "reference": reference,
                    "gain_reference_minus_policy": describe([
                        auc_lookup[(str(c), reference)] - auc_lookup[(str(c), policy)]
                        for c in contexts])})
    return {"unit_of_analysis": "held-out context; random seeds averaged within context first",
            "primary_metric": "observed target-gene MSE / same-mask same-context unadapted MSE",
            "secondary_metric": "mean per-target Pearson across observed training-selected genes",
            "uncertainty": ("2000 paired context bootstrap resamples, seed 77, percentile 95%"
                            if count >= 8 else "context min/max only; fewer than eight contexts"),
            "interval_scope": "descriptive across evaluated contexts; outer models share training contexts",
            "hypothesis_tests": "none; no significance claims",
            "scores": scores, "paired_comparisons": comparisons,
            "auc": auc_scores, "auc_paired_comparisons": auc_comparisons,
            "auc_definition": "trapezoidal normalized MSE integral / maximum recorded budget",
            "limitations": ["Author log2FC signature completion, plate 1 / 0.05 uM, known drugs only",
                            "Exposure duration and author DE methods unverified; no control encoder",
                            "Covariance designs depend on acquired missingness, not response values; not nonlinear PRISM",
                            "Effective model noise is not calibrated biological uncertainty",
                            "No clinical, unseen-drug, primary-tumour, dose/time extrapolation claims"]}


def run(input_path, output):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    if not input_path.is_file():
        raise FileNotFoundError(f"input unavailable: {input_path}; no results created")
    digest = sha256(input_path)
    data, contexts, perturbations, genes = load_input(input_path)
    if sha256(input_path) != digest:
        raise RuntimeError("input changed while being loaded")
    probes, targets = split_perturbations(perturbations)
    budgets = [b for b in BUDGETS if b <= len(probes)]
    context_order = np.argsort(contexts, kind="stable")
    source_files = (Path(__file__), ROOT / "scripts/check_pilot.py",
                    ROOT / "notes/pilot_protocol.md", ROOT / "notes/implementation.md",
                    ROOT / "notes/data_provenance.md")
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path.resolve()), "input_sha256": digest,
        "input_shape": list(data.shape), "contexts": contexts.tolist(),
        "perturbations": perturbations.tolist(), "genes_count": len(genes),
        "evaluated_context_order": contexts[context_order].tolist(),
        "probe_indices": probes.tolist(), "target_indices": targets.tolist(),
        "probe_ids": perturbations[probes].tolist(), "target_ids": perturbations[targets].tolist(),
        "software": {"python": sys.version, "executable": sys.executable,
                     "numpy": np.__version__, "platform": platform.platform(),
                     "thread_environment": {k: os.environ[k] for k in os.environ if k in (
                         "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files if p.is_file()},
        "config": {"budgets": budgets, "split_seed": SPLIT_SEED,
                   "policies": POLICIES, "random_seeds": RANDOM_SEEDS,
                   "inner_seeds": INNER_SEEDS, "rank_grid": [2, 4, 8],
                   "noise_multipliers": NOISE_GRID, "default_noise_multiplier": 1.0,
                   "gene_cap": 1000, "rank_cap": 8,
                   "inner_context_validation": len(contexts) >= 8,
                   "bootstrap_resamples": 2000 if len(contexts) >= 8 else 0,
                   "bootstrap_seed": 77, "checkpoint_budget": 5,
                   "tie_break": "lexical identifier; 1e-12 relative objective near-ties",
                   "normalization": "author log2fc; no transform or imputation applied",
                   "input_schema": "log2fc,observed,lfc_se,contexts,drugs,gene_names",
                   "gene_eligibility": "finite in every training context x drug profile only",
                   "scoring_mask": "held-out finite target genes; identical across policies/budgets",
                   "acquisition": "nominal full-gene candidates; observed-row update only after acquisition",
                   "scope": "published-signature completion, plate1, 0.05uM; methods/time unknown",
                   "variance_definition": "median selected-gene mean-perturbation context sample variance"}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "predictions").mkdir()
    write_json(output / "manifest.json", manifest)
    metrics, selections, checks, coverage = [], [], [], []
    pearson_path = output / "per_target_pearson.csv"
    try:
        with open(pearson_path, "w", newline="", encoding="utf-8") as pearson_file:
            pearson_writer = csv.writer(pearson_file)
            pearson_writer.writerow(("context", "policy", "seed", "budget", "target", "pearson"))
            for number, holdout in enumerate(context_order):
                context = str(contexts[holdout])
                train_indices = context_order[context_order != holdout]
                train_contexts = contexts[train_indices]
                model, tuning = select_and_fit(data[train_indices], genes, probes, targets,
                                              train_contexts, len(contexts) >= 8)
                observed = data[holdout][:, model.gene_indices]
                truth = observed[targets]
                scoring = TargetMetrics(model, targets, truth)
                for target, count in zip(targets, scoring.counts):
                    coverage.append({"context": context, "target": str(perturbations[target]),
                                     "observed_genes": int(count), "selected_genes": len(model.gene_indices),
                                     "coverage": float(count / len(model.gene_indices))})
                if scoring.baseline_mse == 0:
                    raise ValueError(f"zero unadapted target MSE in context {context}; normalized MSE undefined")
                stats = probe_statistics(model, probes, observed[probes])
                stem = f"context_{number:03d}"
                np.savez_compressed(output / "models" / f"{stem}.npz",
                    context=np.asarray(context), training_contexts=train_contexts,
                    gene_indices=model.gene_indices, gene_ids=genes[model.gene_indices],
                    perturbations=perturbations, mu=model.mu, A=model.A, grams=model.grams,
                    tau2=np.asarray(model.tau2), eigenvalues=model.eigenvalues)
                write_json(output / "models" / f"{stem}.json", {
                    "context": context, "training_contexts": train_contexts,
                    "rank": model.rank, "requested_rank": model.requested_rank,
                    "training_eligible_gene_count": model.eligible_count,
                    "noise_multiplier": model.noise_multiplier, "tau2": model.tau2,
                    "residual_variance_median": model.residual_variance_median,
                    "variance_floor": model.variance_floor,
                    "variance_floor_active": model.tau2 == model.variance_floor,
                    "selected_gene_ids": genes[model.gene_indices], "tuning": tuning})
                checkpoint = {}
                deterministic_orders = {}
                context_metrics = []
                min_reduction = 0.0
                baseline = scoring.evaluate(np.zeros(model.rank))
                for policy in POLICIES:
                    seeds = RANDOM_SEEDS if policy == "random" else (-1,)
                    for seed in seeds:
                        if policy == "unadapted":
                            means = {b: np.zeros(model.rank) for b in budgets}
                            missing_by_budget = {b: 0 for b in budgets}
                        else:
                            order, covariances, reduction = plan_order(
                                model, probes, targets, policy, budgets[-1], seed, reveal=stats.gram)
                            min_reduction = min(min_reduction, reduction)
                            means = posterior_means(model, order, covariances, stats, budgets)
                            missing_by_budget = {b: sum(int((~stats.mask(p)).sum()) for p in order[:b])
                                                 for b in budgets}
                            if policy in DETERMINISTIC:
                                deterministic_orders[policy] = order
                            for step, p in enumerate(order, start=1):
                                selections.append({"context": context, "policy": policy,
                                    "seed": seed, "step": step, "perturbation": str(perturbations[p]),
                                    "observed_genes": int(stats.mask(p).sum()),
                                    "missing_genes": int((~stats.mask(p)).sum())})
                            if policy in DETERMINISTIC or (policy == "random" and seed == 0):
                                checkpoint[policy] = (predict(model, targets, means[5]),
                                                     order[:5], means[5], covariances[5])
                        for budget in budgets:
                            mse, normalized, correlation, valid, by_target = scoring.evaluate(means[budget])
                            if budget == 0:
                                assert mse == baseline[0] and normalized == 1.0
                            row = {"context": context, "policy": policy, "seed": seed,
                                   "budget": budget, "mse": mse, "baseline_mse": scoring.baseline_mse,
                                   "normalized_mse": normalized, "pearson_mean": correlation,
                                    "pearson_valid_targets": valid, "target_count": len(targets),
                                    "gene_count": len(model.gene_indices),
                                    "target_observed_entries": scoring.n, "target_possible_entries": truth.size,
                                    "target_coverage": scoring.n / truth.size,
                                    "acquired_missing_entries": missing_by_budget[budget]}
                            context_metrics.append(row)
                            for t, value in zip(targets, by_target):
                                pearson_writer.writerow((context, policy, seed, budget,
                                    str(perturbations[t]), float(value) if np.isfinite(value) else ""))
                metrics.extend(context_metrics)
                np.savez_compressed(output / "predictions" / f"{stem}_k5.npz",
                    context=np.asarray(context), policies=np.asarray(CHECKPOINT_POLICIES),
                    seeds=np.asarray([-1, -1, -1, 0]), budget=np.asarray(5),
                    gene_ids=genes[model.gene_indices], target_ids=perturbations[targets],
                    target_indices=targets, truth=truth, baseline=model.mu[targets],
                    target_observed=np.isfinite(truth),
                    selected_probe_observed=np.stack([np.stack([stats.mask(q) for q in checkpoint[p][1]])
                                                      for p in CHECKPOINT_POLICIES]),
                    predictions=np.stack([checkpoint[p][0] for p in CHECKPOINT_POLICIES]),
                    selected_probe_indices=np.stack([checkpoint[p][1] for p in CHECKPOINT_POLICIES]),
                    posterior_means=np.stack([checkpoint[p][2] for p in CHECKPOINT_POLICIES]),
                    posterior_covariances=np.stack([checkpoint[p][3] for p in CHECKPOINT_POLICIES]))
                sentinel = sentinel_check(data, holdout, train_indices, genes, probes, targets,
                                         contexts, model, len(contexts) >= 8, deterministic_orders)
                checks.append({"context": context, "probe_target_disjoint": True,
                    "acquisition_counts_distinct": True, "k0_identical": True,
                    "covariance_psd_and_monotone": True,
                    "minimum_covariance_reduction_eigenvalue": min_reduction, "sentinel": sentinel})
                # Flush useful partial real outputs after each completed context.
                write_csv(output / "metrics.csv", metrics, METRIC_FIELDS)
                write_csv(output / "selections.csv", selections,
                          ("context", "policy", "seed", "step", "perturbation", "observed_genes", "missing_genes"))
                write_csv(output / "coverage.csv", coverage,
                          ("context", "target", "observed_genes", "selected_genes", "coverage"))
                write_json(output / "checks.json", {"contexts": checks})
                pearson_file.flush()
                print(f"completed {number + 1}/{len(contexts)}: {context}, rank={model.rank}, "
                      f"noise_multiplier={model.noise_multiplier:g}", flush=True)
        per_context, auc = aggregate(metrics, contexts[context_order], budgets)
        write_csv(output / "per_context.csv", per_context, CONTEXT_FIELDS)
        write_csv(output / "context_auc.csv", auc, ("context", "policy", "normalized_auc"))
        write_json(output / "summary.json", summarize(per_context, auc, contexts[context_order], budgets))
        manifest.update({"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                         "metrics_rows": len(metrics), "selection_rows": len(selections),
                         "output_sha256": {str(p.relative_to(output)): sha256(p)
                            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}})
        write_json(output / "manifest.json", manifest)
        print(f"complete: {output}", flush=True)
    except BaseException as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                         "traceback": traceback.format_exc()})
        write_json(output / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed_v1/responses.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pilot/main",
                        help="fresh result directory; existing directories are never overwritten")
    args = parser.parse_args(argv)
    try:
        run(args.input.resolve(), args.output.resolve())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
