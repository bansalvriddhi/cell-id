#!/usr/bin/env python3
"""Exploratory observed-gene context offset, with independent saved-file checks."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "2"

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [0, 1, 2, 3, 5, 10, 20]
POLICY = "context-shift-random"
FIELDS = ("context", "policy", "seed", "budget", "mse", "baseline_mse", "normalized_mse",
          "target_observed_entries", "target_possible_entries", "corrected_genes")
CONTEXT_FIELDS = ("context", "policy", "budget", "n_seeds", "mse", "baseline_mse", "normalized_mse")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path):
    return json.loads(path.read_text())


def load(primary, source):
    manifest = read_json(primary / "manifest.json")
    assert manifest["status"] == "complete"
    assert digest(source) == manifest["input_sha256"]
    assert manifest["config"]["budgets"] == BUDGETS
    assert manifest["config"]["random_seeds"] == list(range(20))
    for name, hash_value in manifest["output_sha256"].items():
        assert digest(primary / name) == hash_value, f"primary changed: {name}"
    with np.load(source, allow_pickle=False) as d:
        data = d["log2fc"].astype(float)
        contexts, drugs, genes = (d[k].astype(str) for k in ("contexts", "drugs", "gene_names"))
        np.testing.assert_array_equal(d["observed"], np.isfinite(data))
    return manifest, data, contexts, drugs, genes


def run(source, primary, output):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    pm, data, contexts, drugs, genes = load(primary, source)
    probes, targets = np.asarray(pm["probe_indices"]), np.asarray(pm["target_indices"])
    assert set(probes).isdisjoint(targets)
    original_metrics = {(r["context"], r["policy"], int(r["seed"]), int(r["budget"])): r
                        for r in read_csv(primary / "metrics.csv")}
    orders = {}
    for r in read_csv(primary / "selections.csv"):
        if r["policy"] == "random":
            orders.setdefault((r["context"], int(r["seed"])), []).append(r["perturbation"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    rows, context_rows = [], []
    for index, context in enumerate(pm["evaluated_context_order"]):
        holdout = int(np.flatnonzero(contexts == context)[0])
        with np.load(primary / "models" / f"context_{index:03d}.npz", allow_pickle=False) as m:
            chosen, mu = m["gene_indices"], m["mu"]
            np.testing.assert_array_equal(m["gene_ids"], genes[chosen])
            assert str(m["context"]) == context
        observed = data[holdout][:, chosen]
        truth, target_mu = observed[targets], mu[targets]
        mask = np.isfinite(truth)
        denominator = float(original_metrics[(context, "unadapted", -1, 0)]["mse"])
        np.testing.assert_allclose(np.mean((target_mu[mask] - truth[mask]) ** 2), denominator,
                                   rtol=1e-12, atol=1e-12)
        # Independent of seed: masked quadratic target SSE for a common gene correction.
        residual = np.zeros_like(truth)
        residual[mask] = truth[mask] - target_mu[mask]
        sse0, gene_sum, counts = np.sum(residual ** 2), residual.sum(axis=0), mask.sum(axis=0)
        for seed in range(20):
            order = np.asarray([int(np.flatnonzero(drugs == drug)[0]) for drug in orders[(context, seed)]])
            np.testing.assert_array_equal(order, np.random.default_rng(seed).permutation(probes)[:20])
            total, number = np.zeros(len(chosen)), np.zeros(len(chosen), dtype=int)
            checkpoint_corrections, checkpoint_predictions = [], []
            for budget in range(21):
                if budget:
                    probe = order[budget - 1]
                    good = np.isfinite(observed[probe])
                    total[good] += observed[probe, good] - mu[probe, good]
                    number[good] += 1
                if budget not in BUDGETS:
                    continue
                correction = np.divide(total, number, out=np.zeros_like(total), where=number > 0)
                sse = float(sse0 - 2 * gene_sum @ correction + counts @ (correction ** 2))
                assert sse >= -1e-10 * max(float(sse0), 1.)
                mse = max(sse, 0.) / int(mask.sum())
                if budget == 0:
                    np.testing.assert_allclose(mse / denominator, 1., rtol=1e-12)
                if budget == 1:
                    first = order[0]
                    expected = np.zeros(len(chosen))
                    good = np.isfinite(observed[first])
                    expected[good] = observed[first, good] - mu[first, good]
                    np.testing.assert_array_equal(correction, expected)
                rows.append(dict(zip(FIELDS, (context, POLICY, seed, budget, mse, denominator,
                    mse / denominator, int(mask.sum()), mask.size, int((number > 0).sum())))))
                if seed == 0 and budget in (0, 1, 5):
                    checkpoint_corrections.append(correction.copy())
                    checkpoint_predictions.append(target_mu + correction[None, :])
            if seed == 0:
                np.savez_compressed(output / "checkpoints" / f"context_{index:03d}_seed0.npz",
                    context=np.asarray(context), budgets=np.asarray([0, 1, 5]), gene_indices=chosen,
                    gene_ids=genes[chosen], target_indices=targets, probe_order=order, truth=truth,
                    target_observed=mask, baseline=target_mu, corrections=np.stack(checkpoint_corrections),
                    predictions=np.stack(checkpoint_predictions))
        for budget in BUDGETS:
            group = [r for r in rows if r["context"] == context and r["budget"] == budget]
            context_rows.append({"context": context, "policy": POLICY, "budget": budget, "n_seeds": 20,
                **{k: float(np.mean([r[k] for r in group])) for k in
                   ("mse", "baseline_mse", "normalized_mse")}})
    context_order = pm["evaluated_context_order"]
    draws = np.random.default_rng(77).integers(0, len(context_order), (2000, len(context_order)))

    def describe(values):
        values = np.asarray(values)
        return {"mean": float(values.mean()), "context_min": float(values.min()),
                "context_max": float(values.max()),
                "descriptive_paired_context_95pct": np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()}

    scores = [{"budget": b, "normalized_mse": describe([r["normalized_mse"] for r in context_rows
                                                        if r["budget"] == b])} for b in BUDGETS]
    primary_context = {(r["context"], r["policy"], int(r["budget"])): float(r["normalized_mse"])
                       for r in read_csv(primary / "per_context.csv")}
    k5 = {r["context"]: r["normalized_mse"] for r in context_rows if r["budget"] == 5}
    comparisons = [{"primary_policy": policy, "budget": 5,
                    "primary_normalized_mse": describe([primary_context[(c, policy, 5)] for c in context_order]),
                    "gain_primary_minus_context_shift": describe([
                        primary_context[(c, policy, 5)] - k5[c] for c in context_order])}
                   for policy in ("random", "V-optimal")]
    summary = {"status": "exploratory; no tuning", "policy": POLICY, "n_contexts": len(context_order),
        "n_random_seeds": 20, "scores": scores, "main_k5_comparisons": comparisons,
        "uncertainty": "2000 paired context resamples, seed77; shared-training-fold intervals are descriptive",
        "scope": "author log2FC signature completion at plate1/0.05uM; methods/time unknown",
        "comparison_note": "primary random uses matched probe orders; V-optimal changes predictor and order",
        "hypothesis_tests": "none"}
    write_csv(output / "metrics.csv", rows, FIELDS)
    write_csv(output / "per_context.csv", context_rows, CONTEXT_FIELDS)
    write_json(output / "summary.json", summary)
    manifest = {"status": "complete", "input": str(source), "input_sha256": digest(source),
        "primary": str(primary), "primary_manifest_sha256": digest(primary / "manifest.json"),
        "primary_output_sha256": pm["output_sha256"], "script_sha256": digest(Path(__file__)),
        "software": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "threads": 2},
        "budgets": BUDGETS, "seeds": list(range(20)), "metric_rows": len(rows),
        "output_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    verification = verify(output)
    write_json(output / "verification.json", verification)
    print(json.dumps({"summary": summary, "verification": verification}, indent=2))


def verify(output):
    """Separate full-vector reconstruction; never calls the streaming/quadratic scorer."""
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "complete"
    primary, source = Path(manifest["primary"]), Path(manifest["input"])
    assert digest(source) == manifest["input_sha256"]
    assert digest(primary / "manifest.json") == manifest["primary_manifest_sha256"]
    for name, hash_value in manifest["primary_output_sha256"].items():
        assert digest(primary / name) == hash_value
    for name, hash_value in manifest["output_sha256"].items():
        assert digest(output / name) == hash_value
    with np.load(source, allow_pickle=False) as d:
        data, contexts = d["log2fc"].astype(float), d["contexts"].astype(str)
    rows = read_csv(output / "metrics.csv")
    lookup = {(r["context"], int(r["seed"]), int(r["budget"])): r for r in rows}
    assert len(rows) == len(lookup) == len(contexts) * 20 * len(BUDGETS)
    maximum, checks = 0., 0
    for path in sorted((output / "checkpoints").glob("*.npz")):
        index = int(path.name.split("_")[1])
        with np.load(primary / "models" / f"context_{index:03d}.npz", allow_pickle=False) as m:
            mu, gene_indices = m["mu"], m["gene_indices"]
        with np.load(path, allow_pickle=False) as p:
            context = str(p["context"])
            holdout = int(np.flatnonzero(contexts == context)[0])
            observed = data[holdout][:, gene_indices]
            np.testing.assert_array_equal(p["gene_indices"], gene_indices)
            np.testing.assert_array_equal(p["truth"], observed[p["target_indices"]])
            mask = np.isfinite(p["truth"])
            np.testing.assert_array_equal(p["target_observed"], mask)
            baseline = mu[p["target_indices"]]
            for j, budget in enumerate(p["budgets"]):
                residuals = observed[p["probe_order"][:budget]] - mu[p["probe_order"][:budget]]
                correction = np.zeros(len(gene_indices))
                for gene in range(len(gene_indices)):
                    values = residuals[:, gene]
                    finite = values[np.isfinite(values)]
                    if len(finite):
                        correction[gene] = sum(float(x) for x in finite) / len(finite)
                np.testing.assert_allclose(p["corrections"][j], correction, rtol=1e-12, atol=1e-12)
                prediction = baseline + correction
                np.testing.assert_allclose(p["predictions"][j], prediction, rtol=1e-12, atol=1e-12)
                direct_mse = float(np.mean((prediction[mask] - p["truth"][mask]) ** 2))
                recorded = lookup[(context, 0, int(budget))]
                maximum = max(maximum, abs(direct_mse - float(recorded["mse"])))
                np.testing.assert_allclose(direct_mse, float(recorded["mse"]), rtol=1e-12, atol=1e-12)
                direct_baseline = float(np.mean((baseline[mask] - p["truth"][mask]) ** 2))
                np.testing.assert_allclose(direct_baseline, float(recorded["baseline_mse"]), rtol=1e-12)
                if budget == 0:
                    np.testing.assert_array_equal(prediction, baseline)
                if budget == 1:
                    values = residuals[0]
                    good = np.isfinite(values)
                    np.testing.assert_array_equal(correction[good], values[good])
                    np.testing.assert_array_equal(correction[~good], 0.)
                checks += 1
    for row in rows:
        np.testing.assert_allclose(float(row["normalized_mse"]), float(row["mse"]) / float(row["baseline_mse"]), rtol=1e-12)
        if int(row["budget"]) == 0:
            np.testing.assert_allclose(float(row["normalized_mse"]), 1., rtol=1e-12)
    per_context = read_csv(output / "per_context.csv")
    assert len(per_context) == len(contexts) * len(BUDGETS)
    for row in per_context:
        group = [lookup[(row["context"], seed, int(row["budget"]))] for seed in range(20)]
        for metric in ("mse", "baseline_mse", "normalized_mse"):
            np.testing.assert_allclose(float(row[metric]), np.mean([float(r[metric]) for r in group]), rtol=1e-12)
    return {"status": "pass", "metric_rows": len(rows), "full_checkpoint_checks": checks,
            "k0_all_seeds_equal": True, "k1_direct_observed_residual": True,
            "maximum_absolute_mse_discrepancy": maximum, "primary_hashes_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed_v1/responses.npz")
    parser.add_argument("--primary", type=Path, default=ROOT / "results/pilot/main")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pilot/context_shift")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.output.resolve()), indent=2))
    else:
        run(args.input.resolve(), args.primary.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
