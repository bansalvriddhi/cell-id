#!/usr/bin/env python3
"""Prepare and verify an offline, observed Tahoe signature-completion pilot.

No network calls, model fitting, imputation, significance filtering or synthetic data.
Default output is new-only; use --out-dir for a rerun beside an earlier result.
Dependencies: numpy, pandas, pyarrow. All processing uses one Arrow thread.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
ROOT = Path(__file__).resolve().parents[1]
REV = "2dc57900b7981cfcf5e211527169a0b006546a95"
BASE = f"https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/{REV}/"
SHARDS = [0,20,21,41,42,61,62,82,83,102,103,123,124,143,144,164,165,184,185]
KEY = ["Cell_ID_Cellosaur", "plate", "drug", "concentration", "concentration_unit"]
META = KEY + ["Cell_ID_DepMap", "Cell_Name_Vevo", "n_cells_trt", "n_cells_ctrl"]
COLS = META + ["gene_name", "log2FoldChange", "lfcSE"]
EXTRA = {
    "sample_metadata.parquet": ("metadata/sample_metadata.parquet", "33167f0ce28cff8357c503cda67d1c7fea200bff6918ee1de3c4f3549175b9d3"),
    "cell_line_metadata.parquet": ("metadata/cell_line_metadata.parquet", "67641f5bdd3fb077978ff1fec4d0d617490674bcd42850515ae3997194c29d4d"),
    "gene_metadata.parquet": ("metadata/gene_metadata.parquet", "d6104d1ca570d94832be0d27ffe8b7e4e54cfd366c9080239219f9c406eb751c"),
    "drug_metadata.parquet": ("metadata/drug_metadata.parquet", "7a04c7a6611e74c92253163b994ff0a47191928ee7511f1c2ca508cee0e79f28"),
    "Tahoe_LICENSE.md": ("LICENSE.md", "82fe759ab83e7619dfe956bb8b295b0ce640ed3706b68b871c15645a0f695a6b"),
    "Tahoe_README.md": ("README.md", "257d17d255800088f2f100805dc4a15dfe03ca1bcc88591b8282d5c4711ee934"),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def source_paths(source):
    return [source / f"train-{i:05d}-of-01026.parquet" for i in SHARDS]


def manifest(source):
    listing = json.loads((source / "Tahoe_shards_00000_00199.json").read_text())
    listed = {Path(r["path"]).name: r for r in listing}
    rows = []
    for p in sorted(source.iterdir()):
        if not p.is_file():
            continue
        h = sha256(p)
        expected = None
        if p.name in listed:
            r = listed[p.name]
            url, expected = BASE + r["path"], r["lfs"]["oid"]
            assert p.stat().st_size == r["size"], p
        elif p.name in EXTRA:
            remote, expected = EXTRA[p.name]
            url = BASE + remote
        elif p.name == "Tahoe_methods_discussion_7.json":
            url = "https://huggingface.co/api/datasets/tahoebio/Tahoe-100M/discussions/7"
        elif p.name == "Tahoe_shards_00000_00199.json":
            url = f"https://huggingface.co/api/datasets/tahoebio/Tahoe-100M/tree/{REV}/metadata/pseudobulk_differential_expression?limit=200"
        else:
            raise ValueError(f"Unregistered source: {p}")
        if expected:
            assert h == expected, f"Source checksum mismatch: {p}"
        rows.append(dict(path=str(p.resolve()), filename=p.name, url=url,
                         bytes=p.stat().st_size, sha256=h,
                         expected_sha256=expected or "", revision=REV if "discussions" not in url else "live"))
    result = pd.DataFrame(rows)
    assert result.bytes.sum() < 2_000_000_000
    assert len(source_paths(source)) == 19
    assert all(p.exists() for p in source_paths(source))
    return result


def batches(source, batch_size=262144):
    for p in source_paths(source):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=batch_size, columns=COLS, use_threads=False):
            d = b.to_pandas()
            d = d[(d.plate == "1") & np.isclose(d.concentration, 0.05, atol=1e-7, rtol=0)
                  & (d.concentration_unit == "uM")]
            if len(d):
                yield p.name, d


def prepare(source, out):
    assert not out.exists(), f"Output already exists; select a new --out-dir: {out}"
    out.mkdir(parents=True)
    m = manifest(source)
    manifest_path = ROOT / "data/source_manifest.tsv"
    if manifest_path.exists():
        old = pd.read_csv(manifest_path, sep="\t", keep_default_na=False)
        assert old.fillna("").equals(m.fillna("")), "Source manifest changed"
    else:
        m.to_csv(manifest_path, sep="\t", index=False)
    m.to_csv(out / "source_manifest.tsv", sep="\t", index=False)

    first = pq.read_table(source_paths(source)[0], columns=["gene_name"], use_threads=False)
    genes = first.slice(0, 62710).column(0).to_pylist()
    assert len(set(genes)) == 62710
    gindex = pd.Index(genes)
    profiles = {}
    for filename, d in batches(source):
        for key, x in d.groupby(KEY, sort=False, dropna=False):
            if key not in profiles:
                profiles[key] = dict(meta=x.iloc[0][META].to_dict(),
                                     lfc=np.full(len(genes), np.nan, np.float32),
                                     se=np.full(len(genes), np.nan, np.float32),
                                     seen=np.zeros(len(genes), bool), files=set())
            p = profiles[key]
            assert x[META].drop_duplicates().shape[0] == 1
            assert p["meta"] == x.iloc[0][META].to_dict()
            idx = gindex.get_indexer(x.gene_name)
            assert (idx >= 0).all() and len(set(idx)) == len(idx)
            assert not p["seen"][idx].any(), f"Duplicate gene/profile: {key}"
            p["seen"][idx] = True
            p["lfc"][idx] = x.log2FoldChange.to_numpy(dtype=np.float32)
            p["se"][idx] = x.lfcSE.to_numpy(dtype=np.float32)
            p["files"].add(filename)
    rows = []
    for key, p in profiles.items():
        row = dict(p["meta"])
        row.update(drug_clean=row["drug"].strip(), n_gene_rows=int(p["seen"].sum()),
                   n_finite_lfc=int(np.isfinite(p["lfc"]).sum()), source_files=";".join(sorted(p["files"])))
        row["complete"] = row["n_gene_rows"] == len(genes)
        row["qc_pass"] = row["complete"] and row["n_cells_trt"] >= 50 and row["n_cells_ctrl"] >= 50
        rows.append(row)
    candidate = pd.DataFrame(rows).sort_values(["Cell_ID_Cellosaur", "drug_clean"])
    names = candidate[["drug", "drug_clean"]].drop_duplicates()
    assert not names.drug_clean.duplicated().any(), "Whitespace cleanup causes collision"
    contexts = sorted(candidate.Cell_ID_Cellosaur.unique())
    assert len(contexts) == 10, contexts
    eligible = candidate[candidate.qc_pass]
    common = set.intersection(*(set(eligible.loc[eligible.Cell_ID_Cellosaur == c, "drug_clean"]) for c in contexts))
    drugs = sorted(common)
    assert len(drugs) >= 20, len(drugs)
    candidate["retained"] = candidate.qc_pass & candidate.drug_clean.isin(common)
    candidate.to_csv(out / "candidate_profiles.tsv", sep="\t", index=False)
    chosen = candidate[candidate.retained].copy()
    assert len(chosen) == len(contexts) * len(drugs)
    assert not chosen.duplicated(["Cell_ID_Cellosaur", "drug_clean"]).any()
    shape = (len(contexts), len(drugs), len(genes))
    lfc, se = np.full(shape, np.nan, np.float32), np.full(shape, np.nan, np.float32)
    chosen["context_index"] = chosen.Cell_ID_Cellosaur.map({c:i for i,c in enumerate(contexts)})
    chosen["drug_index"] = chosen.drug_clean.map({d:i for i,d in enumerate(drugs)})
    for _, row in chosen.iterrows():
        p = profiles[tuple(row[k] for k in KEY)]
        lfc[row.context_index, row.drug_index] = p["lfc"]
        se[row.context_index, row.drug_index] = p["se"]
    assert not np.isinf(lfc).any()
    assert (se[np.isfinite(se)] >= 0).all()
    observed = np.isfinite(lfc)
    np.savez_compressed(out / "responses.npz", log2fc=lfc, lfc_se=se, observed=observed,
                        contexts=np.array(contexts), drugs=np.array(drugs), gene_names=np.array(genes))
    chosen.sort_values(["context_index", "drug_index"]).to_csv(out / "profiles.tsv", sep="\t", index=False)
    cmeta = pq.read_table(source / "cell_line_metadata.parquet").to_pandas()
    cmeta = cmeta[["Cell_ID_Cellosaur", "cell_name", "Organ"]].drop_duplicates()
    assert not cmeta.Cell_ID_Cellosaur.duplicated().any()
    ct = pd.DataFrame({"context_index":range(len(contexts)), "Cell_ID_Cellosaur":contexts}).merge(cmeta, how="left", validate="one_to_one")
    ct.to_csv(out / "contexts.tsv", sep="\t", index=False)
    pd.DataFrame({"drug_index":range(len(drugs)), "drug":drugs}).to_csv(out / "drugs.tsv", sep="\t", index=False)
    gm = pq.read_table(source / "gene_metadata.parquet").to_pandas()
    gm = gm.rename(columns={"gene_symbol":"gene_name"})
    unique = gm[~gm.gene_name.duplicated(keep=False)][["gene_name", "ensembl_id", "token_id"]]
    gene_table = pd.DataFrame({"gene_index":range(len(genes)), "gene_name":genes}).merge(unique, how="left", validate="one_to_one")
    # For Ensembl fallback names, use the existing exact accession, not a guessed alias.
    fallback = gene_table.ensembl_id.isna() & gene_table.gene_name.isin(set(gm.ensembl_id))
    gene_table.loc[fallback, "ensembl_id"] = gene_table.loc[fallback, "gene_name"]
    gene_table.to_csv(out / "genes.tsv", sep="\t", index=False)
    summary = dict(source_revision=REV, source_bytes=int(m.bytes.sum()), source_files=len(m),
                   source_shards=len(SHARDS), source_rows=sum(pq.ParquetFile(p).metadata.num_rows for p in source_paths(source)),
                   shape=list(shape), dtype=str(lfc.dtype), gene_names_unique=len(set(genes)),
                   contexts=ct.to_dict("records"), drugs=drugs, plate="1", concentration=0.05, concentration_unit="uM",
                   exposure_time_hours=None, time_status="Not specified in acquired source metadata; not inferred",
                   candidate_profiles=len(candidate), complete_candidate_profiles=int(candidate.complete.sum()),
                   qc_pass_candidate_profiles=int(candidate.qc_pass.sum()), retained_profiles=len(chosen),
                   low_cell_candidate_profiles=int(((candidate.n_cells_trt < 50) | (candidate.n_cells_ctrl < 50)).sum()),
                   incomplete_candidate_profiles=int((~candidate.complete).sum()),
                   finite_lfc_values=int(observed.sum()), missing_lfc_values=int((~observed).sum()),
                   missing_lfc_fraction=float((~observed).mean()),
                   genes_finite_in_all_profiles=int(observed.all(axis=(0,1)).sum()),
                   genes_finite_in_at_least_one_profile=int(observed.any(axis=(0,1)).sum()),
                   treatment_cells_min=int(chosen.n_cells_trt.min()), treatment_cells_max=int(chosen.n_cells_trt.max()),
                   control_cells_min=int(chosen.n_cells_ctrl.min()), control_cells_max=int(chosen.n_cells_ctrl.max()),
                   gene_mapping_missing=int(gene_table.ensembl_id.isna().sum()),
                   metadata_shapes={f:list(pq.read_table(source / f).shape) for f in EXTRA if f.endswith(".parquet")},
                   numpy_version=np.__version__, pandas_version=pd.__version__, pyarrow_version=pa.__version__,
                   method="Source log2FoldChange unchanged; plate1, 0.05uM, >=50 cells per arm, complete profiles and common drugs; no gene filtering")
    write_json(out / "summary.json", summary)
    with (out / "checksums.sha256").open("x") as f:
        for p in sorted(out.iterdir()):
            if p.name != "checksums.sha256":
                f.write(f"{sha256(p)}  {p.name}\n")
    return summary


def verify(source, out, name):
    # Verify final serialized artifacts, then independently re-index every retained
    # source value using a different batch size. Checks include original nulls.
    for line in (out / "checksums.sha256").read_text().splitlines():
        h, filename = line.split("  ", 1)
        assert sha256(out / filename) == h, filename
    m = manifest(source)
    s = json.loads((out / "summary.json").read_text())
    z = np.load(out / "responses.npz", allow_pickle=False)
    lfc, se, observed = z["log2fc"], z["lfc_se"], z["observed"]
    assert lfc.dtype == np.float32 and observed.dtype == bool
    assert list(lfc.shape) == s["shape"] and observed.shape == se.shape == lfc.shape
    assert np.array_equal(observed, np.isfinite(lfc))
    assert int(observed.sum()) == s["finite_lfc_values"]
    ci, di, gi = ({str(v):i for i,v in enumerate(z[k])} for k in ["contexts", "drugs", "gene_names"])
    assert len(ci) >= 3 and len(di) >= 20 and len(gi) == 62710
    rows = pd.read_csv(out / "profiles.tsv", sep="\t", keep_default_na=False)
    keys = {(r.Cell_ID_Cellosaur, str(r.plate), r.drug, float(np.float32(r.concentration)), r.concentration_unit): r
            for r in rows.itertuples()}
    assert len(keys) == len(ci) * len(di)
    checked = np.zeros(lfc.shape, bool)
    total = 0
    for _, d in batches(source, batch_size=131072):
        for key, x in d.groupby(KEY, sort=False, dropna=False):
            if key not in keys:
                continue
            r = keys[key]
            assert (x.n_cells_trt == r.n_cells_trt).all() and r.n_cells_trt >= 50
            assert (x.n_cells_ctrl == r.n_cells_ctrl).all() and r.n_cells_ctrl >= 50
            c, drug = ci[key[0]], di[key[2].strip()]
            g = np.fromiter((gi[v] for v in x.gene_name), dtype=np.int64, count=len(x))
            assert not checked[c, drug, g].any()
            assert np.array_equal(lfc[c,drug,g], x.log2FoldChange.to_numpy(np.float32), equal_nan=True)
            assert np.array_equal(se[c,drug,g], x.lfcSE.to_numpy(np.float32), equal_nan=True)
            checked[c,drug,g] = True
            total += len(x)
    assert checked.all() and total == lfc.size
    sample_meta = pq.read_table(source / "sample_metadata.parquet").to_pandas()
    controls = sample_meta[(sample_meta.plate == "plate1") & (sample_meta.drug == "DMSO_TF")]
    assert len(controls) > 0
    source_conditions = set()
    for row in sample_meta.itertuples():
        for drug, dose, unit in ast.literal_eval(row.drugname_drugconc):
            source_conditions.add((row.plate.removeprefix("plate"), drug.strip(), float(np.float32(dose)), unit))
    assert all((str(r.plate), r.drug.strip(), float(np.float32(r.concentration)), r.concentration_unit)
               in source_conditions for r in rows.itertuples())
    control_records = controls[["sample", "plate", "drug", "drugname_drugconc"]].to_dict("records")
    raw = pq.ParquetFile(source_paths(source)[0]).read_row_group(0).to_pandas().iloc[0]
    assert raw.gene_name == "TSPAN6" and raw.drug == "4EGI-1" and raw.Cell_ID_Cellosaur == "CVCL_0023"
    assert lfc[ci["CVCL_0023"],di["4EGI-1"],gi["TSPAN6"]] == np.float32(-0.27955949306488037)
    # Exact value above independently agrees with the public dataset-viewer record.
    report = dict(status="PASS", exit_status=0, command=shlex.join([sys.executable]+sys.argv),
                  shape=list(lfc.shape), source_bytes=int(m.bytes.sum()), source_checksums_verified=len(m),
                  independently_rechecked_gene_profile_rows=total,
                  compared_columns=["log2FoldChange", "lfcSE"], finite_lfc_values=int(observed.sum()),
                  missing_lfc_values=int((~observed).sum()), no_imputation=True,
                  complete_context_drug_panel=True, all_retained_profiles_min_50_cells_per_arm=True,
                  plate1_vehicle_samples_in_sample_metadata=control_records,
                  sample_metadata_agrees_with_every_retained_drug_dose=True,
                  public_viewer_TSPAN6_4EGI1_A549_log2fc=float(raw.log2FoldChange),
                  responses_npz_bytes=(out / "responses.npz").stat().st_size,
                  per_context_missing_lfc_fraction={c:float((~observed[i]).mean()) for c,i in ci.items()})
    report_path = out / f"{name}.json"
    log_path = out / f"{name}.log"
    if report_path.exists():
        assert json.loads(report_path.read_text()) == report, "Verification changed; choose a new --verification-name"
        assert log_path.read_text() == json.dumps(report, indent=2) + "\n"
    else:
        write_json(report_path, report)
        with log_path.open("x") as f:
            f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/source")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/processed_v1")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verification-name", default="verification")
    a = parser.parse_args()
    assert Path(a.verification_name).name == a.verification_name
    if not a.verify_only:
        prepare(a.source_dir, a.out_dir)
    verify(a.source_dir, a.out_dir, a.verification_name)


if __name__ == "__main__":
    main()
