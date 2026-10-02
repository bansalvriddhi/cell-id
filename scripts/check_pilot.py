#!/usr/bin/env python3
"""Independent CELL-ID result reconstruction, or in-memory algebra unit checks.

--self-test creates no input datasets or empirical output tables.
Default verification reads results/pilot/main and its recorded input file.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import builtins
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, label, rtol=2e-8, atol=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                               equal_nan=True, err_msg=label)


def number(value):
    return float(value) if value not in (None, "") else float("nan")


def direct_pearson(predictions, truth):
    """Independent full-vector Pearson; no model-specific sufficient statistics."""
    result = []
    for p, y in zip(predictions, truth):
        observed = np.isfinite(y)
        p, y = p[observed], y[observed]
        if len(y) < 2:
            result.append(float("nan"))
            continue
        p, y = p - np.mean(p), y - np.mean(y)
        denom = np.linalg.norm(p) * np.linalg.norm(y)
        result.append(float(p @ y / denom) if denom > 0 else float("nan"))
    return np.asarray(result)


def mean_finite(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def verify_result(directory, input_path=None):
    """Read-only: do not import the producer or reuse its metric/summary code."""
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete", "run is incomplete"
    for relative, digest in manifest["output_sha256"].items():
        assert checksum(directory / relative) == digest, f"output checksum changed: {relative}"
    input_path = input_path or Path(manifest["input_path"])
    assert checksum(input_path) == manifest["input_sha256"], "input SHA-256 mismatch"
    with np.load(input_path, allow_pickle=False) as source:
        responses = source["log2fc"].astype(float)
        contexts = source["contexts"].astype(str)
        perturbations = source["drugs"].astype(str)
        genes = source["gene_names"].astype(str)
        np.testing.assert_array_equal(source["observed"], np.isfinite(responses))
        assert source["lfc_se"].shape == responses.shape
    assert list(responses.shape) == manifest["input_shape"]
    assert contexts.tolist() == manifest["contexts"]
    assert perturbations.tolist() == manifest["perturbations"]
    probes = np.asarray(manifest["probe_indices"])
    targets = np.asarray(manifest["target_indices"])
    assert min(len(probes), len(targets)) >= 10
    assert set(probes).isdisjoint(targets)
    assert sorted([*probes, *targets]) == list(range(len(perturbations)))
    independent_perm = np.random.default_rng(20261002).permutation(np.argsort(perturbations))
    assert set(probes) == set(independent_perm[:len(perturbations) // 2])
    assert perturbations[probes].tolist() == sorted(manifest["probe_ids"])
    assert perturbations[targets].tolist() == sorted(manifest["target_ids"])
    budgets = manifest["config"]["budgets"]
    assert budgets == [k for k in (0, 1, 2, 3, 5, 10, 20, 50) if k <= len(probes)]
    policies = ("unadapted", "random", "trace-uncertainty", "D-optimal", "V-optimal")
    context_order = sorted(contexts)
    rows = read_csv(directory / "metrics.csv")
    metrics = {(x["context"], x["policy"], int(x["seed"]), int(x["budget"])): x for x in rows}
    assert len(metrics) == len(rows) == len(contexts) * 24 * len(budgets)
    assert len(rows) == manifest["metrics_rows"]
    expected_keys = {(str(c), p, s, k) for c in context_order for p in policies
                     for s in (range(20) if p == "random" else (-1,)) for k in budgets}
    assert set(metrics) == expected_keys
    selections = read_csv(directory / "selections.csv")
    assert len(selections) == manifest["selection_rows"]
    expected_groups = {(str(c), p, s) for c in context_order for p in policies[1:]
                       for s in (range(20) if p == "random" else (-1,))}
    selection_groups = {}
    for row in selections:
        key = (row["context"], row["policy"], int(row["seed"]))
        selection_groups.setdefault(key, []).append(row)
    assert set(selection_groups) == expected_groups
    for key, group in selection_groups.items():
        assert [int(x["step"]) for x in group] == list(range(1, budgets[-1] + 1))
        ids = [x["perturbation"] for x in group]
        assert len(ids) == len(set(ids)) == budgets[-1]
        assert set(ids) <= set(manifest["probe_ids"])
        assert set(ids).isdisjoint(manifest["target_ids"])
        if key[1] == "random":
            expected = perturbations[np.random.default_rng(key[2]).permutation(probes)[:budgets[-1]]]
            assert ids == expected.tolist(), "random acquisition order is not reproducible"

    pearson_rows = read_csv(directory / "per_target_pearson.csv")
    correlations = {}
    for row in pearson_rows:
        key = (row["context"], row["policy"], int(row["seed"]), int(row["budget"]))
        correlations.setdefault(key, {})[row["target"]] = number(row["pearson"])
    assert len(pearson_rows) == len(rows) * len(targets)
    assert set(correlations) == set(metrics)
    for key, recorded in metrics.items():
        by_target = correlations[key]
        assert set(by_target) == set(manifest["target_ids"])
        assert int(recorded["target_count"]) == len(targets)
        assert 1 <= int(recorded["gene_count"]) <= min(1000, len(genes))
        assert int(recorded["pearson_valid_targets"]) == int(np.isfinite(list(by_target.values())).sum())
        close(number(recorded["pearson_mean"]), mean_finite(list(by_target.values())), "Pearson mean")
        close(number(recorded["normalized_mse"]), number(recorded["mse"]) /
              number(recorded["baseline_mse"]), "within-context normalization")
        if key[3] == 0 or key[1] == "unadapted":
            close(number(recorded["normalized_mse"]), 1.0, "k0/unadapted equality", atol=0)

    full_prediction_checks = 0
    max_mse_discrepancy = 0.0
    coverage_rows = read_csv(directory / "coverage.csv")
    coverage_lookup = {(x["context"], x["target"]): x for x in coverage_rows}
    assert len(coverage_lookup) == len(coverage_rows) == len(contexts) * len(targets)
    mask_summary, fitted_hyperparameters = [], []
    acquisition_replays = 0
    for index, context in enumerate(context_order):
        stem = f"context_{index:03d}"
        metadata = json.loads((directory / "models" / f"{stem}.json").read_text())
        with np.load(directory / "models" / f"{stem}.npz", allow_pickle=False) as m:
            gene_indices, mu, A, tau2 = m["gene_indices"], m["mu"], m["A"], float(m["tau2"])
            assert str(m["context"]) == context
            assert m["gene_ids"].tolist() == genes[gene_indices].tolist()
            train_ids = [i for i in np.argsort(contexts) if contexts[i] != context]
            assert m["training_contexts"].tolist() == contexts[train_ids].tolist()
            train = responses[train_ids]
            eligible = np.flatnonzero(np.isfinite(train).all(axis=(0, 1)))
            variance = np.full(len(genes), np.nan)
            variance[eligible] = (np.var(train[:, :, eligible], axis=0, ddof=1).mean(axis=0)
                                  if len(train) > 1 else 0.)
            chosen = eligible[np.lexsort((genes[eligible], -variance[eligible]))[:1000]]
            assert metadata["training_eligible_gene_count"] == len(eligible)
            np.testing.assert_array_equal(gene_indices, chosen)
            direct_mu = train[:, :, chosen].mean(axis=0)
            close(mu, direct_mu, "training-only perturbation mean")
            close(m["grams"], np.stack([p.T @ p for p in A]), "per-drug Gram matrices")
            close(metadata["residual_variance_median"], np.median(variance[chosen]), "noise scale")
            close(tau2, max(metadata["noise_multiplier"] * np.median(variance[chosen]),
                            metadata["variance_floor"]), "noise multiplier")
            assert A.shape == (*mu.shape, metadata["rank"])
            assert A.shape[-1] <= min(len(train) - 1, 8)
            # Direct SVD cross-check avoids the producer's context-Gram implementation.
            if A.shape[-1]:
                centered = (train[:, :, chosen] - direct_mu).reshape(len(train), -1)
                _, singular, _ = np.linalg.svd(centered, full_matrices=False)
                r = A.shape[-1]
                flattened = A.reshape(-1, r)
                close(np.linalg.svd(flattened, compute_uv=False), singular[:r] /
                      np.sqrt(len(train) - 1), "empirical covariance eigenvalues", rtol=1e-7)
                # The eigen-equation remains valid when a truncation crosses a tied eigenspace.
                covariance_times_factor = centered.T @ (centered @ flattened) / (len(train) - 1)
                eigen_scaled_factor = flattened * (singular[:r] ** 2 / (len(train) - 1))
                relative_error = np.linalg.norm(covariance_times_factor - eigen_scaled_factor)
                assert relative_error <= 1e-7 * max(np.linalg.norm(eigen_scaled_factor), 1e-20)
        holdout = int(np.flatnonzero(contexts == context)[0])
        fitted_hyperparameters.append({"context": context, "rank": metadata["rank"],
            "noise_multiplier": metadata["noise_multiplier"], "eligible_genes": len(eligible),
            "selected_genes": len(gene_indices)})
        # Replay every acquired covariance from the observed rows. For deterministic
        # designs, independently check that each choice maximizes the NOMINAL objective.
        nominal_grams = np.stack([a.T @ a for a in A])
        nominal_Q = nominal_grams[targets].sum(axis=0)
        for (c, policy, seed), group in selection_groups.items():
            if c != context:
                continue
            precision = np.eye(A.shape[-1])
            covariance = precision.copy()
            remaining = list(probes)
            missing_count = 0
            for row in group:
                chosen_probe = int(np.flatnonzero(perturbations == row["perturbation"])[0])
                if policy != "random":
                    objectives = []
                    for candidate in remaining:
                        gram = nominal_grams[candidate]
                        if policy == "trace-uncertainty":
                            objective = float(np.trace(covariance @ gram))
                        elif policy == "D-optimal":
                            # Independent equivalent precision determinant ratio.
                            objective = (np.linalg.slogdet(precision + gram / tau2)[1]
                                         - np.linalg.slogdet(precision)[1])
                        else:
                            new_covariance = np.linalg.inv(precision + gram / tau2)
                            objective = float(np.trace(nominal_Q @ (covariance - new_covariance)))
                        objectives.append(objective)
                    selected_value = objectives[remaining.index(chosen_probe)]
                    best = max(objectives)
                    assert selected_value >= best - 1e-8 * max(abs(best), 1e-20), "nonmaximal design selection"
                remaining.remove(chosen_probe)
                y = responses[holdout, chosen_probe, gene_indices]
                observed_mask = np.isfinite(y)
                assert int(row["observed_genes"]) == int(observed_mask.sum())
                assert int(row["missing_genes"]) == int((~observed_mask).sum())
                missing_count += int((~observed_mask).sum())
                ap = A[chosen_probe, observed_mask]
                precision += ap.T @ ap / tau2
                new_covariance = np.linalg.inv(precision)
                if len(covariance):
                    assert np.linalg.eigvalsh(new_covariance).min() >= -1e-9
                    assert np.linalg.eigvalsh(covariance - new_covariance).min() >= -1e-9
                covariance = new_covariance
                step = int(row["step"])
                if step in budgets:
                    assert int(metrics[(context, policy, seed, step)]["acquired_missing_entries"]) == missing_count
            acquisition_replays += 1
        with np.load(directory / "predictions" / f"{stem}_k5.npz", allow_pickle=False) as p:
            assert str(p["context"]) == context and int(p["budget"]) == 5
            assert p["policies"].tolist() == ["trace-uncertainty", "D-optimal", "V-optimal", "random"]
            assert p["seeds"].tolist() == [-1, -1, -1, 0]
            assert p["target_ids"].tolist() == perturbations[targets].tolist()
            assert p["gene_ids"].tolist() == genes[gene_indices].tolist()
            truth, baseline = p["truth"], p["baseline"]
            np.testing.assert_array_equal(truth, responses[holdout][targets][:, gene_indices])
            mask = np.isfinite(truth)
            np.testing.assert_array_equal(p["target_observed"], mask)
            close(baseline, mu[targets], "saved baseline")
            baseline_mse = float(np.mean((baseline[mask] - truth[mask]) ** 2))
            assert baseline_mse > 0
            mask_summary.append({"context": context, "observed_target_entries": int(mask.sum()),
                                 "possible_target_entries": int(mask.size),
                                 "coverage": float(mask.mean())})
            for t, m in zip(perturbations[targets], mask):
                recorded_coverage = coverage_lookup[(context, str(t))]
                assert int(recorded_coverage["observed_genes"]) == int(m.sum())
                assert int(recorded_coverage["selected_genes"]) == len(gene_indices)
                close(number(recorded_coverage["coverage"]), m.mean(), "target coverage")
            for row in rows:
                if row["context"] == context:
                    close(number(row["baseline_mse"]), baseline_mse, "common context baseline")
                    assert int(row["gene_count"]) == len(gene_indices)
                    assert int(row["target_observed_entries"]) == int(mask.sum())
                    assert int(row["target_possible_entries"]) == mask.size
                    close(number(row["target_coverage"]), mask.mean(), "policy-invariant coverage")
            assert p["predictions"].shape == (4, len(targets), len(gene_indices))
            for j, policy in enumerate(p["policies"]):
                seed = int(p["seeds"][j])
                prediction = p["predictions"][j]
                recorded = metrics[(context, str(policy), seed, 5)]
                direct_mse = float(np.mean((prediction[mask] - truth[mask]) ** 2))
                discrepancy = abs(number(recorded["mse"]) - direct_mse)
                max_mse_discrepancy = max(max_mse_discrepancy, discrepancy)
                close(number(recorded["mse"]), direct_mse, "full prediction MSE")
                close(number(recorded["normalized_mse"]), direct_mse / baseline_mse,
                      "full prediction normalized MSE")
                direct_corr = direct_pearson(prediction, truth)
                saved_corr = [correlations[(context, str(policy), seed, 5)][str(t)]
                              for t in perturbations[targets]]
                close(saved_corr, direct_corr, "full prediction per-target Pearson", rtol=1e-7)
                selected = p["selected_probe_indices"][j]
                ids = [x["perturbation"] for x in selection_groups[(context, str(policy), seed)][:5]]
                assert perturbations[selected].tolist() == ids
                precision = np.eye(A.shape[-1])
                natural = np.zeros(A.shape[-1])
                previous_covariance = np.eye(A.shape[-1])
                for position, selected_probe in enumerate(selected):
                    observed = responses[holdout, selected_probe, gene_indices]
                    observed_mask = np.isfinite(observed)
                    np.testing.assert_array_equal(p["selected_probe_observed"][j, position], observed_mask)
                    Ap = A[selected_probe, observed_mask]
                    precision += Ap.T @ Ap / tau2
                    natural += Ap.T @ (observed[observed_mask] - mu[selected_probe, observed_mask]) / tau2
                    covariance = np.linalg.inv(precision)
                    if A.shape[-1]:
                        assert np.linalg.eigvalsh(covariance).min() >= -1e-9
                        assert np.linalg.eigvalsh(previous_covariance - covariance).min() >= -1e-9
                    previous_covariance = covariance
                independent_mean = np.linalg.solve(precision, natural)
                close(p["posterior_means"][j], independent_mean, "posterior sufficient statistics")
                close(p["posterior_covariances"][j], previous_covariance, "posterior covariance")
                independent_prediction = np.stack([mu[t] + A[t] @ independent_mean for t in targets])
                close(prediction, independent_prediction, "full posterior prediction")
                full_prediction_checks += 1

    # Independent context aggregation, paired bootstrap and normalized AUC.
    per_context_rows = read_csv(directory / "per_context.csv")
    per_context = {(x["context"], x["policy"], int(x["budget"])): x for x in per_context_rows}
    assert len(per_context) == len(per_context_rows) == len(contexts) * 5 * len(budgets)
    for key, row in per_context.items():
        group = [x for k, x in metrics.items() if (k[0], k[1], k[3]) == key]
        assert int(row["n_seeds"]) == len(group)
        for field in ("mse", "baseline_mse", "normalized_mse", "pearson_mean"):
            close(number(row[field]), mean_finite([number(x[field]) for x in group]),
                  f"seed-averaged {field}")
    auc_rows = read_csv(directory / "context_auc.csv")
    auc = {(x["context"], x["policy"]): number(x["normalized_auc"]) for x in auc_rows}
    assert len(auc) == len(auc_rows) == len(contexts) * 5
    for key, value in auc.items():
        curve = np.asarray([number(per_context[(*key, k)]["normalized_mse"]) for k in budgets])
        direct_auc = float(np.dot(np.diff(budgets), (curve[1:] + curve[:-1]) / 2) / budgets[-1])
        close(value, direct_auc, "normalized AUC")
    summary = json.loads((directory / "summary.json").read_text())
    bootstrap = (np.random.default_rng(77).integers(0, len(contexts), (2000, len(contexts)))
                 if len(contexts) >= 8 else None)

    def verify_description(description, values):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        assert description["n_contexts"] == len(finite)
        close(number(description["mean"]), mean_finite(values), "summary mean")
        if len(finite):
            close(description["context_min"], np.min(finite), "summary context minimum")
            close(description["context_max"], np.max(finite), "summary context maximum")
        if bootstrap is not None and len(finite) == len(values):
            interval = np.percentile(np.mean(values[bootstrap], axis=1), [2.5, 97.5])
            close(description["paired_context_bootstrap_95pct"], interval, "paired bootstrap")
        else:
            assert "paired_context_bootstrap_95pct" not in description

    for score in summary["scores"]:
        for metric in ("normalized_mse", "mse", "pearson_mean"):
            values = [number(per_context[(str(c), score["policy"], score["budget"])][metric])
                      for c in context_order]
            verify_description(score[metric], values)
    for comparison in summary["paired_comparisons"]:
        values = [number(per_context[(str(c), comparison["reference"], comparison["budget"])]["normalized_mse"])
                  - number(per_context[(str(c), comparison["policy"], comparison["budget"])]["normalized_mse"])
                  for c in context_order]
        verify_description(comparison["gain_reference_minus_policy"], values)
    for score in summary["auc"]:
        verify_description(score["normalized_auc"], [auc[(str(c), score["policy"])] for c in context_order])
    for comparison in summary["auc_paired_comparisons"]:
        verify_description(comparison["gain_reference_minus_policy"], [
            auc[(str(c), comparison["reference"])] - auc[(str(c), comparison["policy"])]
            for c in context_order])
    checks = json.loads((directory / "checks.json").read_text())["contexts"]
    assert len(checks) == len(contexts)
    for check in checks:
        assert all(check[k] is True for k in ("probe_target_disjoint", "acquisition_counts_distinct",
                                              "k0_identical", "covariance_psd_and_monotone"))
        assert check["sentinel"]["fit_unchanged"] is True
        assert check["sentinel"]["target_mask_entries_changed"] > 0
        assert check["sentinel"]["all_deterministic_and_random0_orders_unchanged"] is True
        assert check["sentinel"]["posterior_means_all_steps_unchanged"] is True
        assert check["sentinel"]["k5_predictions_unchanged"] is True
    return {"status": "pass", "contexts": len(contexts), "metrics_rows": len(rows),
            "input_sha256": manifest["input_sha256"],
            "full_prediction_policy_context_checks": full_prediction_checks,
            "mask_aware_acquisition_replays": acquisition_replays,
            "maximum_absolute_mse_reconstruction_discrepancy": max_mse_discrepancy,
            "target_coverage": mask_summary, "fitted_hyperparameters": fitted_hyperparameters,
            "key_scores": [x for x in summary["scores"] if x["budget"] in (5, budgets[-1])],
            "key_paired_comparisons": [x for x in summary["paired_comparisons"] if
                                        x["budget"] == 5 and x["reference"] == "random"],
            "auc": summary["auc"],
            "checks": ["input/output hashes", "fixed metadata-only split", "all acquisition counts",
                       "training-only finite-gene selection/means", "direct SVD covariance cross-check",
                       "common k0", "masked full k5 MSE/Pearson/posteriors", "covariance monotonicity",
                       "acquired-mask-only covariance replay and nominal candidate design scores",
                       "target coverage/missing probe counts", "numeric-and-missingness sentinel recorded",
                       "random-seed/context aggregation", "paired bootstrap or min/max", "AUC"]}


class AlgebraChecks(unittest.TestCase):
    """Deliberately perturbed examples, not research data or accuracy evidence."""
    @classmethod
    def setUpClass(cls):
        import run_pilot
        cls.pilot = run_pilot

    def example(self, c=5, p=26, g=19, seed=184):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(c, p, g))
        # Nonlinear and heteroscedastic components avoid just testing a factor teacher.
        x += 0.3 * np.sin(2 * x) + rng.normal(size=(c, 1, 1)) * rng.uniform(0.1, 4, (1, p, 1))
        return (x, np.asarray([f"c{i:03}" for i in range(c)]),
                np.asarray([f"p{i:03}" for i in range(p)]),
                np.asarray([f"g{i:04}" for i in range(g)]))

    def test_input_contract_and_metadata_only_split(self):
        p = self.pilot
        data, contexts, drugs, genes = self.example()
        p.validate_arrays(data, contexts, drugs, genes)
        probes, targets = p.split_perturbations(drugs)
        reverse_probes, reverse_targets = p.split_perturbations(drugs[::-1])
        self.assertEqual(drugs[probes].tolist(), drugs[::-1][reverse_probes].tolist())
        self.assertEqual(drugs[targets].tolist(), drugs[::-1][reverse_targets].tolist())
        self.assertFalse(set(probes) & set(targets))
        for invalid in (data.astype(int), data[:, :19], np.full_like(data, np.inf)):
            with self.assertRaises(ValueError):
                p.validate_arrays(invalid, contexts, drugs, genes)
        with self.assertRaises(ValueError):
            p.validate_arrays(data, np.repeat("same", len(contexts)), drugs, genes)

    def test_covariance_matches_empirical_full_rank(self):
        p = self.pilot
        data, _, _, genes = self.example(g=7)
        model = p.fit_model(data, genes)
        selected = data[:, :, model.gene_indices]
        centered = (selected - selected.mean(axis=0)).reshape(len(data), -1)
        a = model.A.reshape(-1, model.rank)
        close(a @ a.T, centered.T @ centered / (len(data) - 1), "empirical covariance", rtol=1e-9)
        for drug in range(data.shape[1]):
            close(model.grams[drug], model.A[drug].T @ model.A[drug], "drug Gram")

    def test_all_policies_quadratic_mse_and_correlations(self):
        p = self.pilot
        for scale, seed in ((1.0, 184), (1e-5, 913), (1e5, 46)):
            data, _, drugs, genes = self.example(seed=seed)
            data = data * scale
            probes, targets = p.split_perturbations(drugs)
            model = p.fit_model(data[:-1], genes)
            observed = data[-1][:, model.gene_indices]
            scoring = p.TargetMetrics(model, targets, observed[targets])
            stats = p.probe_statistics(model, probes, observed[probes])
            for policy in (*p.DETERMINISTIC, "random"):
                order, covariances, _ = p.plan_order(model, probes, targets, policy, len(probes), 0)
                means = p.posterior_means(model, order, covariances, stats, range(len(probes) + 1))
                for k, mean in means.items():
                    full = p.predict(model, targets, mean)
                    mse, normalized, correlation, _, by_target = scoring.evaluate(mean)
                    direct = float(np.mean((full - observed[targets]) ** 2))
                    close(mse, direct, "quadratic MSE", rtol=1e-9, atol=1e-20 * scale ** 2)
                    close(normalized, direct / scoring.baseline_mse, "normalized MSE")
                    close(by_target, direct_pearson(full, observed[targets]), "quadratic Pearson")
                    close(correlation, np.mean(direct_pearson(full, observed[targets])), "mean Pearson")
                    if k:
                        p.check_covariance(covariances[k - 1], covariances[k])
                self.assertEqual(scoring.evaluate(means[0])[1], 1.0)

    def test_design_objectives_against_full_gene_space(self):
        p = self.pilot
        data, _, drugs, genes = self.example(g=5)
        probes, targets = p.split_perturbations(drugs)
        model = p.fit_model(data[:-1], genes)
        full_target_a = model.A[targets].reshape(-1, model.rank)
        for policy in p.DETERMINISTIC:
            order, covariances, _ = p.plan_order(model, probes, targets, policy, 3)
            remaining = list(probes)
            covariance = np.eye(model.rank)
            for step, choice in enumerate(order):
                values = []
                for candidate in remaining:
                    a = model.A[candidate]
                    observation_covariance = model.tau2 * np.eye(len(a)) + a @ covariance @ a.T
                    conditional = covariance - covariance @ a.T @ np.linalg.solve(
                        observation_covariance, a @ covariance)
                    if policy == "trace-uncertainty":
                        objective = np.trace(a @ covariance @ a.T)
                    elif policy == "D-optimal":
                        objective = np.linalg.slogdet(np.eye(len(a)) + a @ covariance @ a.T / model.tau2)[1]
                    else:
                        objective = np.trace(full_target_a @ (covariance - conditional) @ full_target_a.T)
                    values.append(objective)
                self.assertEqual(choice, remaining[int(np.argmax(values))])
                remaining.remove(choice)
                covariance = covariances[step + 1]

    def test_masked_acquisition_and_target_quadratic_algebra(self):
        p = self.pilot
        data, contexts, drugs, genes = self.example(c=8, g=31, seed=622)
        probes, targets = p.split_perturbations(drugs)
        # These two training-missing genes must be excluded without consulting the test context.
        data[0, 0, -1] = np.nan
        data[2, 1, -2] = np.nan
        model = p.fit_model(data[:-1], genes)
        self.assertEqual(model.eligible_count, len(genes) - 2)
        self.assertNotIn(len(genes) - 1, model.gene_indices)
        self.assertNotIn(len(genes) - 2, model.gene_indices)
        self.assertTrue(np.isfinite(data[:-1, :, model.gene_indices]).all())
        observed = data[-1][:, model.gene_indices].copy()
        observed[::2, ::3] = np.nan
        observed[probes[0], :] = np.nan  # Entirely missing probe supplies zero information.
        observed[targets[0], :] = np.nan
        observed[targets[1], 1:] = np.nan  # Pearson needs at least two observations.
        scoring = p.TargetMetrics(model, targets, observed[targets])
        mask = np.isfinite(observed[targets])
        self.assertEqual(scoring.n, mask.sum())
        stats = p.probe_statistics(model, probes, observed[probes])
        close(stats.gram(probes[0]), np.zeros((model.rank, model.rank)), "all-missing probe Gram")
        close(stats[probes[0]], np.zeros(model.rank), "all-missing probe natural parameter")
        for policy in (*p.DETERMINISTIC, "random"):
            queried = []

            def reveal(chosen):
                self.assertNotIn(chosen, queried)
                queried.append(chosen)
                return stats.gram(chosen)

            order, covariances, _ = p.plan_order(model, probes, targets, policy, 10, 1, reveal=reveal)
            self.assertEqual(queried, list(order))
            means = p.posterior_means(model, order, covariances, stats, range(11))
            for k, mean in means.items():
                prediction = p.predict(model, targets, mean)
                direct_mse = np.mean((prediction[mask] - observed[targets][mask]) ** 2)
                values = scoring.evaluate(mean)
                close(values[0], direct_mse, "masked quadratic MSE")
                close(values[4], direct_pearson(prediction, observed[targets]), "masked Pearson")
                precision, natural = np.eye(model.rank), np.zeros(model.rank)
                for q in order[:k]:
                    good = np.isfinite(observed[q])
                    a = model.A[q, good]
                    precision += a.T @ a / model.tau2
                    natural += a.T @ (observed[q, good] - model.mu[q, good]) / model.tau2
                close(mean, np.linalg.solve(precision, natural), "masked direct posterior")
                close(covariances[k], np.linalg.inv(precision), "masked direct covariance")
            # Numerical responses cannot affect design when missingness is unchanged.
            changed = observed[probes] * -173.4 + 91
            changed_stats = p.probe_statistics(model, probes, changed)
            changed_order, changed_cov, _ = p.plan_order(model, probes, targets, policy, 10, 1,
                                                        reveal=changed_stats.gram)
            np.testing.assert_array_equal(order, changed_order)
            np.testing.assert_array_equal(covariances, changed_cov)
            # No unacquired mask can affect the first choice; callback is consulted afterwards.
            all_missing = p.probe_statistics(model, probes, np.full_like(observed[probes], np.nan))
            missing_order, _, _ = p.plan_order(model, probes, targets, policy, 1, 1, reveal=all_missing.gram)
            self.assertEqual(order[0], missing_order[0])
        with self.assertRaises(ValueError):
            p.fit_model(np.full_like(data[:-1], np.nan), genes)
        with self.assertRaises(ValueError):
            p.TargetMetrics(model, targets, np.full_like(observed[targets], np.nan))

    def test_hidden_target_masks_do_not_change_training_or_design(self):
        p = self.pilot
        data, contexts, drugs, genes = self.example(c=8, g=17, seed=139)
        probes, targets = p.split_perturbations(drugs)
        data[-1, :, ::4] = np.nan
        data[0, 0, -1] = np.nan
        indices = np.arange(7)
        model, _ = p.select_and_fit(data[indices], genes, probes, targets, contexts[indices], True)
        self.assertIn(0, model.gene_indices)  # Held-out missingness did not remove this gene.
        observed = data[-1][:, model.gene_indices]
        stats = p.probe_statistics(model, probes, observed[probes])
        orders = {policy: p.plan_order(model, probes, targets, policy, 10, reveal=stats.gram)[0]
                  for policy in p.DETERMINISTIC}
        sentinel = p.sentinel_check(data, 7, indices, genes, probes, targets, contexts, model, True, orders)
        self.assertGreater(sentinel["target_mask_entries_changed"], 0)
        self.assertTrue(sentinel["fit_unchanged"])

    def test_rank_eight_fifty_probes_and_ill_conditioning(self):
        p = self.pilot
        data, _, drugs, genes = self.example(c=12, p=102, g=9, seed=9301)
        # Distinct eigenvalue scales stress covariance conditioning beyond the small examples.
        data *= np.geomspace(1e-4, 1e4, len(genes))[None, None, :]
        probes, targets = p.split_perturbations(drugs)
        model = p.fit_model(data[:-1], genes)
        self.assertEqual(model.rank, 8)
        observed = data[-1][:, model.gene_indices]
        stats = p.probe_statistics(model, probes, observed[probes])
        scoring = p.TargetMetrics(model, targets, observed[targets])
        for policy in (*p.DETERMINISTIC, "random"):
            order, covariances, _ = p.plan_order(model, probes, targets, policy, 50, 19)
            means = p.posterior_means(model, order, covariances, stats, p.BUDGETS)
            self.assertEqual(set(means), set(p.BUDGETS))
            self.assertEqual(len(set(order)), 50)
            previous_target_trace = float(np.trace(scoring.Q))
            for covariance in covariances:
                target_trace = float(np.trace(scoring.Q @ covariance))
                self.assertLessEqual(target_trace, previous_target_trace + 1e-8 * previous_target_trace)
                previous_target_trace = target_trace
            for budget, mean in means.items():
                full = p.predict(model, targets, mean)
                direct = np.mean((full - observed[targets]) ** 2)
                close(scoring.evaluate(mean)[0], direct, f"stress MSE k={budget}", rtol=1e-8)
                selected = order[:budget]
                direct_precision = np.eye(model.rank) + model.grams[selected].sum(axis=0) / model.tau2
                direct_natural = (np.sum([stats[int(q)] for q in selected], axis=0)
                                  if budget else np.zeros(model.rank))
                close(mean, np.linalg.solve(direct_precision, direct_natural), "stress posterior", rtol=1e-7)

    def test_gene_cap_inner_tuning_and_target_sentinel(self):
        p = self.pilot
        data, contexts, drugs, genes = self.example(c=8, g=11)
        probes, targets = p.split_perturbations(drugs)
        indices = np.arange(7)
        model, tuning = p.select_and_fit(data[indices], genes, probes, targets, contexts[indices], True)
        self.assertEqual(tuning["ranks"], [2, 4, 5])
        self.assertEqual(len(tuning["candidates"]), 9)
        self.assertTrue(all(len(x["folds"]) == 7 for x in tuning["candidates"]))
        orders = {policy: p.plan_order(model, probes, targets, policy, 10)[0] for policy in p.DETERMINISTIC}
        result = p.sentinel_check(data, 7, indices, genes, probes, targets, contexts, model, True, orders)
        self.assertTrue(result["fit_unchanged"])
        wide, _, _, wide_genes = self.example(g=1017)
        fit = p.fit_model(wide[:-1], wide_genes)
        self.assertEqual(len(fit.gene_indices), 1000)
        variance = np.var(wide[:-1], axis=0, ddof=1).mean(axis=0)
        np.testing.assert_array_equal(fit.gene_indices, np.lexsort((wide_genes, -variance))[:1000])

    def test_degenerate_rank_constant_genes_and_ties(self):
        p = self.pilot
        data, contexts, drugs, genes = self.example(g=1)
        probes, targets = p.split_perturbations(drugs)
        for train in (np.zeros_like(data[:-1]), data[:1]):
            model = p.fit_model(train, genes)
            self.assertEqual(model.rank, 0)
            self.assertGreater(model.tau2, 0)
            for policy in p.DETERMINISTIC:
                order, covariance, _ = p.plan_order(model, probes, targets, policy, 10)
                np.testing.assert_array_equal(order, probes[:10])
                observed = data[-1][:, model.gene_indices]
                stats = p.probe_statistics(model, probes, observed[probes])
                means = p.posterior_means(model, order, covariance, stats, (0, 5, 10))
                scoring = p.TargetMetrics(model, targets, observed[targets])
                for mean in means.values():
                    self.assertEqual(scoring.evaluate(mean)[1], 1.0)
                    self.assertEqual(scoring.evaluate(mean)[3], 0)
        identical = np.repeat(data[:1], 5, axis=0)
        self.assertEqual(p.fit_model(identical, genes).rank, 0)

    def test_summary_pairing_auc_and_small_context_rule(self):
        p = self.pilot
        for count in (3, 8):
            contexts = [f"c{i}" for i in range(count)]
            rows = []
            for i, context in enumerate(contexts):
                for policy in p.POLICIES:
                    for seed in (range(20) if policy == "random" else (-1,)):
                        for budget in (0, 5, 10):
                            value = 1 + budget * (i - 2) * 0.001 + (seed + 1) * budget * 0.0001
                            rows.append({"context": context, "policy": policy, "seed": seed,
                                "budget": budget, "mse": 2 * value, "baseline_mse": 2,
                                "normalized_mse": value, "pearson_mean": 0.2})
            per_context, auc = p.aggregate(rows, contexts, [0, 5, 10])
            summary = p.summarize(per_context, auc, contexts, [0, 5, 10])
            self.assertEqual(len(auc), count * 5)
            value = next(x for x in per_context if x["context"] == "c0" and
                         x["policy"] == "random" and x["budget"] == 10)
            close(value["normalized_mse"], np.mean([1 - .02 + (s + 1) * .001 for s in range(20)]),
                  "within-context seed mean")
            for score in summary["scores"]:
                self.assertEqual("paired_context_bootstrap_95pct" in score["normalized_mse"], count >= 8)

    def test_refuse_existing_outputs_before_input_access(self):
        with self.assertRaises(FileExistsError):
            self.pilot.run(Path("input_that_does_not_exist.npz"), ROOT / "scripts")
        with self.assertRaises(FileNotFoundError):
            self.pilot.run(ROOT / "input_that_does_not_exist.npz", ROOT / "results/pilot/unit_missing_no_write")
        self.assertFalse((ROOT / "results/pilot/unit_missing_no_write").exists())

    def test_full_serialization_and_independent_checker_in_memory(self):
        """Exercise production run/check without writing any synthetic file to disk."""
        root = ROOT / "results/pilot/__in_memory_unit_only__"
        real_open, real_load, real_save = builtins.open, np.load, np.savez_compressed
        real_exists, real_isfile = Path.exists, Path.is_file
        real_mkdir, real_glob, real_text = Path.mkdir, Path.rglob, Path.read_text
        files, directories = {}, set()

        def virtual(path):
            return isinstance(path, (str, os.PathLike)) and Path(path).is_relative_to(root)

        class Writer(io.StringIO):
            def __init__(self, path):
                super().__init__()
                self.path = Path(path)

            def close(self):
                if not self.closed:
                    files[self.path] = self.getvalue().encode("utf-8")
                super().close()

        def memory_open(path, mode="r", *args, **kwargs):
            if not virtual(path):
                return real_open(path, mode, *args, **kwargs)
            path = Path(path)
            if "w" in mode:
                self.assertNotIn("b", mode)
                return Writer(path)
            return io.BytesIO(files[path]) if "b" in mode else io.StringIO(files[path].decode("utf-8"))

        def memory_load(path, *args, **kwargs):
            return real_load(io.BytesIO(files[Path(path)]) if virtual(path) else path, *args, **kwargs)

        def memory_save(path, *args, **kwargs):
            if not virtual(path):
                return real_save(path, *args, **kwargs)
            buffer = io.BytesIO()
            real_save(buffer, *args, **kwargs)
            files[Path(path)] = buffer.getvalue()

        def memory_mkdir(path, *args, **kwargs):
            if not virtual(path):
                return real_mkdir(path, *args, **kwargs)
            if path in directories and not kwargs.get("exist_ok", False):
                raise FileExistsError(str(path))
            directories.add(path)

        patches = (
            mock.patch("builtins.open", memory_open),
            mock.patch.object(np, "load", memory_load),
            mock.patch.object(np, "savez_compressed", memory_save),
            mock.patch.object(Path, "mkdir", memory_mkdir),
            mock.patch.object(Path, "exists", lambda p: (p in files or p in directories)
                              if virtual(p) else real_exists(p)),
            mock.patch.object(Path, "is_file", lambda p: p in files if virtual(p) else real_isfile(p)),
            mock.patch.object(Path, "rglob", lambda p, pattern: iter(k for k in files if k.is_relative_to(p))
                              if virtual(p) else real_glob(p, pattern)),
            mock.patch.object(Path, "read_text", lambda p, *a, **k: files[p].decode("utf-8")
                              if virtual(p) else real_text(p, *a, **k)))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            # Small-context, nested-validation/bootstrap, and rank-zero/undefined-Pearson cases.
            for count, gene_count in ((3, 9), (8, 7), (2, 1)):
                data, contexts, drugs, genes = self.example(c=count, p=20, g=gene_count)
                input_path, output = root / f"input{count}.npz", root / f"run{count}"
                if gene_count > 1:
                    data[-1, ::3, 0] = np.nan
                    data[0, 1, -1] = np.nan
                memory_save(input_path, log2fc=data, observed=np.isfinite(data), lfc_se=np.ones_like(data),
                            contexts=contexts, drugs=drugs, gene_names=genes)
                self.pilot.run(input_path, output)
                verification = verify_result(output, input_path)
                self.assertEqual(verification["status"], "pass")
                self.assertEqual(verification["full_prediction_policy_context_checks"], count * 4)
                with self.assertRaises(FileExistsError):
                    self.pilot.run(input_path, output)
        self.assertFalse(real_exists(root), "in-memory check unexpectedly created files")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="in-memory algorithm unit checks, not empirical evidence")
    parser.add_argument("--run", type=Path, default=ROOT / "results/pilot/main")
    parser.add_argument("--input", type=Path, help="optional relocated original input (SHA-256 must match)")
    parser.add_argument("--save-verification", type=Path, help="new JSON file; refuses overwrite")
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AlgebraChecks)
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if outcome.wasSuccessful() else 1
    result = verify_result(args.run.resolve(), args.input.resolve() if args.input else None)
    if args.save_verification:
        with open(args.save_verification, "x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
