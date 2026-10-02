# CELL-ID data acquisition trace

## Scope and delivery checklist (recorded before extraction)

- [x] Preserve only real public source files under `data/`, with exact URLs, pinned source revision when available, byte counts, and SHA-256 checksums.
- [x] Produce a CPU-readable response matrix plus row/column metadata under `data/`; response entries must be observed published pseudobulk log2 fold changes, never generated values or zero-filled missing responses.
- [x] Retain gene identifiers as supplied; retain context, drug, dose, unit, plate and cell-count QC metadata. Missing entries stay missing with an explicit mask.
- [x] Verify at least three measured cell contexts and tens of common drugs within the download budget: 10 contexts and 58 drugs at plate 1 / 0.05 uM. Exposure duration remains unverified; this is an explicitly bounded limitation, not an invented time annotation.
- [x] Save reproducible extraction and verification in `scripts/prepare_data.py`; no experiments or manuscript.
- [x] Document available source normalization/control definitions, unresolved method details, batch limitations, selection exclusions, and valid leave-one-cell-context-out active-probing splits.
- [x] Record the exact verification command, output location and exit status.

Validation domain: the acquired subset of Tahoe-100M cancer cell lines at a matched plate/dose regime, not arbitrary cell types, patients, doses, times or unseen drugs. Dataset coverage/QC can be checked globally as design information; expression-derived feature selection and model fitting must be confined to training contexts in each outer split.

## Acquisition decisions

The working folder contained only `.DS_Store` at inspection. Tahoe Bio's official Hugging Face repository supplies a `pseudobulk_differential_expression` table with measured treatment-versus-control log2 fold changes and cell counts, avoiding raw single-cell downloads. The official table is much larger than the budget, but individually downloadable Parquet shards are approximately 80–100 MB. We will inspect bounded shards and retain a matched-dose multi-context subset if supported; the full table is not being downloaded.

Official repository: https://huggingface.co/datasets/tahoebio/Tahoe-100M

Pinned revision: `2dc57900b7981cfcf5e211527169a0b006546a95` (API `lastModified`: `2025-07-23T14:30:24.000Z`). Dataset card declares CC0-1.0. The card explicitly identifies `DMSO_TF` as vehicle controls and requires plate-matched controls.

Paper DOI: https://doi.org/10.1101/2025.02.20.639398. Literature retrieval initially returned abstract only (OpenAlex lists no open-access copy); methods not available through that attempt and not inferred from the abstract.

Alternative Arc hosting is not preferred: its current README documents migration to a Google Marketplace Requester Pays bucket and deprecation of its old bucket on 2026-03-31. No billed requests will be made.

## Frozen extraction contract

Before extraction, select 19 shards: `00000`, `00020`, `00021`, `00041`, `00042`, `00061`, `00062`, `00082`, `00083`, `00102`, `00103`, `00123`, `00124`, `00143`, `00144`, `00164`, `00165`, `00184`, `00185` of `01026`. Their declared total is 1,722,093,601 bytes. Small metadata/license/manifests are additional and remain within budget. Shard choice follows source order and cell boundaries, not expression magnitude or differential significance. The first ten contexts in source order are targeted; no claim of representative cell-line sampling.

Outputs planned under `data/processed_v1/`:

- `responses.npz`: `log2fc` float32 and `observed` Boolean arrays, both shape `(C,D,62710)`; `lfc_se` float32 of the same shape; `contexts`, `drugs`, and `gene_names` Unicode axes. Missing published estimates remain NaN, not zero. No clipping, averaging, shrinkage or normalization is added.
- `contexts.tsv`: context index, Cellosaurus ID, published cell name and tissue label.
- `drugs.tsv`: drug index and whitespace-trimmed drug name. Trimming is rejected if it collapses distinct source names.
- `genes.tsv`: gene index and exact published gene name, with unambiguous Ensembl mappings where available. Mixed gene symbols/Ensembl fallback names must not be renamed silently.
- `profiles.tsv`: one row per retained context–drug pair with source plate, exact float32 dose, dose unit, original drug name, source cell counts, completeness, and finite-LFC count.
- `candidate_profiles.tsv`: all acquired plate-1 profiles and QC/completeness status, including exclusions.
- `summary.json`, `verification.json`, `verification.log`, and `checksums.sha256` for measured shapes, missingness and reproducibility.
- `data/source_manifest.tsv`: exact pinned URLs, local source paths, observed bytes, observed SHA-256 and independently published LFS SHA-256 where provided.

Regime/QC: retain only plate `1`, concentration approximately 0.05 uM (absolute tolerance 1e-7 for float32 encoding), at least 50 treatment cells and 50 control cells, exactly one complete 62,710-gene profile per context/drug. Keep only drugs satisfying these rules in every acquired eligible context. Context/drug axes are sorted by exact identifier/name; genes retain the first complete source profile's order. No expression-based gene filtering is performed. The common-panel requirement conditions the pilot on assay availability and survival/QC and is not a claim about all administered drugs.

Important newly verified limitation: https://huggingface.co/datasets/tahoebio/Tahoe-100M/discussions/7 explicitly asks about the released differential-expression table's fit design, observation/replicate unit, control pooling and filtering. No maintainer answer was visible at retrieval. Its schema is DESeq2-like but the exact implementation, normalization, shrinkage and biological replication are not established. Preserve estimates for an exploratory *published-signature completion* pilot, not formal DE inference or absolute-expression reconstruction. `lfcSE` and `padj` must not be interpreted as validated biological-replicate uncertainty.

## Outcome and measured dataset

**Acquisition succeeded, with a methods-qualified use recommendation.** All data are real public Tahoe-100M author-released differential-expression estimates. No raw single-cell matrices were downloaded, no model was run, and no generated/placeholder expression was introduced. No compute jobs, paid data APIs, uploads or experiments were used. All data retrieval was through `webfetch`.

Working root: `/Users/aayambansal/Desktop/bullshit-goofy/iris-vriddhi`.

Primary readable dataset: `/Users/aayambansal/Desktop/bullshit-goofy/iris-vriddhi/data/processed_v1/responses.npz` (126,307,181 bytes). Load with `numpy.load(path, allow_pickle=False)`; it needs only NumPy for downstream loading. The preserved Parquet sources require PyArrow. Local preparation versions: NumPy 2.4.6, pandas 2.3.3, PyArrow 25.0.1. PyArrow was installed for this work; processing and verification were local CPU-only.

Measured by `scripts/prepare_data.py`, saved in `data/processed_v1/summary.json`:

| Quantity | Result |
|---|---:|
| Retained source files, including metadata/legal/manifest records | 27 |
| Retained downloaded source bytes | 1,723,649,452 (<2 GB decimal) |
| Differential-expression Parquet shards | 19 |
| Rows in downloaded DE shards | 75,737,429 |
| Tensor shape, context × drug × gene | 10 × 58 × 62,710 |
| Retained complete context–drug profiles | 580 |
| Acquired plate-1 candidate profiles | 819 |
| Complete candidate profiles | 814 |
| Candidates passing completeness and ≥50 cells/arm | 750 |
| Low-cell candidate profiles | 64 |
| Incomplete shard-boundary candidate profiles | 5 |
| Finite LFC entries | 16,447,466 |
| Missing LFC entries | 19,924,334 (54.7796%) |
| Genes finite in at least one retained profile | 48,791 |
| Genes finite in every retained profile | 14,894 |
| Treatment cells per retained profile, range | 295–10,286 |
| Control cells per retained profile, range | 1,805–10,609 |

The 14,894 globally complete genes are a **descriptive QC count only**, not a supplied train/test gene filter. Selecting genes based on held-out-context availability would leak information about the held-out expression distribution.

Contexts in serialized axis order:

| Cellosaurus | Cell name | Source tissue label |
|---|---|---|
| CVCL_0023 | A549 | Lung |
| CVCL_0218 | COLO 205 | Bowel |
| CVCL_0320 | HT-29 | Bowel |
| CVCL_0332 | HS-578T | Breast |
| CVCL_0399 | LoVo | Bowel |
| CVCL_0504 | RKO | Bowel |
| CVCL_0546 | SW480 | Bowel |
| CVCL_1693 | SHP-77 | Lung |
| CVCL_1717 | SW1417 | Bowel |
| CVCL_1731 | SW 900 | Lung |

All 58 named drugs are listed in `data/processed_v1/drugs.tsv`; examples include Bortezomib, Gemcitabine, Irinotecan, Oxaliplatin, Afatinib, Adagrasib, and Everolimus. Each is measured in all ten contexts at 0.05 uM (50 nM), plate 1. No drug-dose pooling is performed. The common-panel restriction is conservative and excludes profiles failing availability/QC, but is not evidence those excluded drugs are biologically inactive. The retained drugs are an alphabetical/source-order prefix imposed by bounded downloads, not a balanced sample of mechanisms of action.

## Source URLs, versions, licensing and scientific interpretation

Authoritative dataset identifier: **`tahoebio/Tahoe-100M`**, configuration **`pseudobulk_differential_expression`**, split **`train`**. Pin revision **`2dc57900b7981cfcf5e211527169a0b006546a95`**.

- Dataset card: https://huggingface.co/datasets/tahoebio/Tahoe-100M
- Pinned DE-file inventory: https://huggingface.co/api/datasets/tahoebio/Tahoe-100M/tree/2dc57900b7981cfcf5e211527169a0b006546a95/metadata/pseudobulk_differential_expression?limit=200
- Exact first shard URL: https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/2dc57900b7981cfcf5e211527169a0b006546a95/metadata/pseudobulk_differential_expression/train-00000-of-01026.parquet
- Remaining exact URLs, local absolute paths, observed bytes and SHA-256: **`data/source_manifest.tsv`**. This is the authoritative per-file inventory, not an instruction to download all shards.
- Pinned metadata URLs use the same `resolve/<revision>/metadata/` prefix with `sample_metadata.parquet`, `gene_metadata.parquet`, `cell_line_metadata.parquet`, and `drug_metadata.parquet`.
- Pinned legal terms: https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/2dc57900b7981cfcf5e211527169a0b006546a95/LICENSE.md ; preserved verbatim as `data/source/Tahoe_LICENSE.md`. **CC0 1.0 Universal** applies to the dataset. Article copyright is separate from dataset licensing.
- Official methods-discussion API: https://huggingface.co/api/datasets/tahoebio/Tahoe-100M/discussions/7 ; preserved as `data/source/Tahoe_methods_discussion_7.json`. Its questions are useful evidence of missing documentation, not independent authoritative methods or validated statistics.
- Independent public first-row check: https://datasets-server.huggingface.co/first-rows?dataset=tahoebio%2FTahoe-100M&config=pseudobulk_differential_expression&split=train ; A549 / 4EGI-1 / TSPAN6 has published `log2FoldChange = -0.27955949306488037`, exactly reproduced in the local tensor.

**Updated publication verified after initial discovery:** Zhang et al., *Tahoe-100M: Mapping drug-induced molecular phenotypes at single-cell resolution*, Cell 189(19):5945–5961.e8, 2026-09-17, DOI https://doi.org/10.1016/j.cell.2026.08.035, PMID **42753697**. PubMed verified this record and abstract. The abstract correctly describes **1,100 drug-dose conditions**, not 1,100 distinct compounds. The literature tool retrieved only a 417-character bibliographic landing-page text for the published article; it did not provide methods. The earlier bioRxiv DOI is https://doi.org/10.1101/2025.02.20.639398. Neither unread full methods nor inferred details are treated as verified evidence here.

**Gene representation.** The source DE table has 62,710 distinct `gene_name` strings, comprising symbols and Ensembl fallback names. The official gene metadata says GRCh38 / Ensembl release 109. `genes.tsv` preserves source ordering and names, and all entries have a mapping to an Ensembl gene ID. Do not apply the raw single-cell token/CLS processing rules to these already processed DE tables.

**Response units and normalization.** `log2FoldChange` is the publisher's processed signed log2 fold-change estimate, not raw counts, CPM, log1p expression or a newly computed pseudobulk mean. The table also provides `baseMean`, `lfcSE`, `stat`, `pvalue`, `padj`, plate and treatment/control cell counts. The precise count aggregation, size factors, DE fit, shrinkage, dispersion, replicate definition and contrast-control pool are **not documented by the acquired official materials**. The DESeq2-like column names alone do not establish these settings. `baseMean` is not a pure untreated baseline and is deliberately not placed in the pilot tensor. Raw source tables retain all published columns for audit.

**Controls.** The official card identifies `DMSO_TF` as vehicle controls and specifies plate matching. Local `sample_metadata.parquet` documents two vehicle samples on plate 1: **smp_1589** and **smp_1590**. Their encoded treatment is `[('DMSO_TF', 0.0, 'uM')]`; this is a metadata convention, **not a measured zero-DMSO concentration**. Every retained drug/dose was checked against sample metadata. The DE tables supply control cell counts, but exactly how the two vehicle wells contributed to each released fit remains unverified. This delivery contains precomputed response estimates, **not untreated expression vectors**.

**Time.** No exposure-duration column exists in the acquired sample or DE metadata. The primary full methods were not retrievable in this session. `exposure_time_hours` is therefore JSON null with an explanatory status string. The pilot is restricted to a single recorded plate/protocol; do not label it 24 h, 48 h, or claim time transfer until the primary protocol is verified.

**Batches and replication.** The raw dataset card distinguishes 14 plates and sample IDs, and Arc's metadata documentation additionally identifies sublibraries. This extraction fixes plate 1, avoiding confounding different doses with different plates inside the pilot. The same plate/vehicle preparation is nevertheless shared across cell contexts. Holding out a cell line is not an independent-plate or independent-experiment test. Source files can cross cell/plate boundaries; the script filters row metadata rather than assuming an entire shard belongs to one context. There are no replicate-level estimates in the tensor; no replicate noise ceiling can be estimated from it.

**Missingness.** Finite/NaN source LFCs are preserved exactly. In the public first rows, a zero-`baseMean` gene has null LFC; this illustrates unestimable-expression missingness, but does not establish that every missing value has this cause. Source boundary truncation is separate: five incomplete acquired profiles were excluded. Low cell counts occur even in the official processed source (including conditions with one or a few treatment cells); the ≥50-cell rule was therefore applied explicitly rather than assuming source QC. No p-value/padj threshold is used: filtering for significance would cause selection bias and leak test outcomes.

## Recommended valid active-probing task

Frame the initial result as **offline recovery of author-released drug-response signatures in a held-out cancer cell context**, at the recorded plate-1 / 50-nM regime. It is not an absolute-expression virtual cell, not a new-drug task, not prospective experimental validation, and not a claim about independent biological replication.

1. Use ten outer leave-one-Cellosaurus-context-out folds: fit on nine complete contexts, then allow only explicitly purchased drug profiles from the tenth. A context includes all of its genes, drug rows and uncertainty fields.
2. Freeze a hidden evaluation-drug subset and a disjoint probe-candidate subset before seeing held-out-context responses. For example, 16 of the 58 drugs can be evaluation-only and 42 probe-eligible, supporting budgets 0, 1, 2, 4, 8, 16, 32. Keep the evaluation set identical across strategies so choosing probes does not change the scored population. Drug identities can be known; training contexts may contain all 58 drugs. Repeat deterministic evaluation partitions if desired.
3. Tune hyperparameters, stopping rules and probe strategies using inner held-out **training** contexts only. Fit all centering, scaling, PCA, gene selection, representations and response means inside the outer training fold. A CPU pilot can select a few thousand sufficiently observed/variable genes using only those nine training contexts. Break feature-selection ties by immutable gene ID.
4. Treat the NPZ as an oracle owned by the evaluator, not as a table the acquisition policy can inspect. Before a drug is queried, do not reveal the held-out LFC vector, its missingness pattern, `lfcSE`, `baseMean`, p-values, or gene-specific QC. The global common-panel membership defines the conditional assay-availability domain; it does not authorize access to hidden response contents.
5. A probe reveals the **whole measured context–drug signature** at this dose, not one gene at a time. Predict the fixed evaluation drugs. Score only observed target entries using a predeclared mask-aware metric and report target-gene coverage. Do not silently fill missing targets with zero or drop hard test contexts. If imputation is needed internally, learn it only from training data and keep it distinct from observed ground truth.
6. Report results per held-out context and aggregate across contexts. Do not count thousands of genes or the cells inside a pseudobulk as independent test replicates. This dataset supports algorithmic comparisons conditional on the published signatures; uncertain source DE fitting and shared controls bound biological interpretation.

## Alternative routes considered

Tahoe succeeded within budget, so LINCS/L1000 and SciPlex3 were **not downloaded**. A secondary community Tahoe pseudobulk project was inspected (https://github.com/agpranay/tahoe_100M-Pseudobulk) but was not used: it points to a separate Google bucket and does not resolve the official DE-table methodology. Arc's current authoritative hosting notes are at https://raw.githubusercontent.com/ArcInstitute/arc-virtual-cell-atlas/main/tahoe-100M/README.md ; they document Requester Pays Marketplace migration. No bucket subscription, billing or raw 100M-cell download was attempted.

## Verification and reproducibility

Successful initial preparation was run in a new Python subprocess, using the default script arguments, and independently re-read every retained gene/profile record. The final read-only verification, after the script's additional control/metadata checks, is:

```sh
/Users/aayambansal/.config/openscience/data-root/conda/envs/python/bin/python /Users/aayambansal/Desktop/bullshit-goofy/iris-vriddhi/scripts/prepare_data.py --verify-only --verification-name verification_final
```

**Exit status 0; PASS.** The log is `data/processed_v1/verification_final.log` and structured output is `data/processed_v1/verification_final.json`. It verifies source hashes, output checksums, shapes/dtypes, finite-value mask, all retained dose/sample metadata, existence of plate-matched vehicle samples, and independently compares **36,371,800 source gene-profile rows** for both LFC and SE using a second batch size and fresh indexing. Original NaNs are included in the equality check. The public A549/4EGI-1/TSPAN6 record is also matched exactly.

`data/processed_v1/checksums.sha256` covers the serialized data/metadata/summary as saved before verification logs. `data/source_manifest.tsv` covers the acquired sources; LFS checksums are independently checked where published. Repeating the final verification command is read-only/idempotent if unchanged. Re-extraction never overwrites an existing processed directory: pass a new `--out-dir data/processed_v2` (or another new path).

Acquisition/debug trace: an initial relative move failed (`mv: ... No such file or directory`, exit 1) because broker downloads land in session workspace rather than the connected folder; all were subsequently moved explicitly to `data/source/`. The first Parquet inspection reported `ModuleNotFoundError: No module named 'pyarrow'`; installing PyArrow resolved it. A bioRxiv full-page retrieval returned HTTP 403; a guessed GitHub repository endpoint returned 404 and was not retried; neither supplied scientific evidence. The final dataset and checks are unaffected. No failed/intermediate generated expression artifacts remain.

## Remaining blocker for stronger biological claims

Obtain the primary exposure-duration protocol and the exact implementation of the official DE table (normalization, observation unit, control pool, replication/dispersion/shrinkage and contrast-level exclusions). Until that is resolved, this is a valid **methods-qualified exploratory signature-completion dataset**, not a fully reconstructed or independently normalized pseudobulk experiment. If strict source-method reproducibility is mandatory rather than pilot feasibility, this documentation gap is a blocker and a fully documented SciPlex/L1000 alternative or regenerated pseudobulk from a bounded raw subset should be considered separately.
