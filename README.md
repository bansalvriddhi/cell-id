# CELL-ID — IRIS National Fair project

## Submission PDFs

1. **[Synopsis (PDF)](report/Synopsis.pdf)** — two-page summary using IRIS's six prescribed sections and word limits.
2. **[Project paper (PDF)](report/Project_Paper.pdf)** — ten-page executed-study paper with title page, contents, methods, four colored figures, discussion, conclusion, acknowledgements and references.

Both documents use anonymous black text and may use colored scientific figures. The [IRIS rules](https://irisnationalfair.org/guidelines.html) prohibit naming the school, city or state in either upload; personal details belong in the registration form. The portal also requires a video link and a project guide, which are outside these two PDFs. [Submission requirements and source links](notes/iris_submission_requirements.md).

This project reports a **completed retrospective computational experiment** using public Tahoe-100M drug-response signatures, not a new laboratory experiment or a clinical test. In the primary ten-context held-out pilot, optimized probe selection did **not** beat random probing. A separately identified exploratory context-offset comparator improved predictions at five and twenty probes but performed worse at one to three. Exact data, algorithms, uncertainties and limitations are in the full paper.

## Data, code and audit trail

- `data/source_manifest_public.tsv`: pinned public Tahoe-100M source URLs and hashes. Download the listed files to their stated `relative_path` if reproducing the source extraction; the original 1.72 GB source shards are retained locally but excluded from Git. The 126 MB `data/processed_v1/responses.npz` is also excluded because ordinary GitHub commits reject files over 100 MB. `data/processed_v1/` retains small processed metadata and verification logs; the analysis scripts reconstruct the tensor from the pinned source files.
- `results/pilot/main/`: primary leave-one-context-out models, acquisition traces, measured comparisons and independent verification. `results/pilot/context_shift/`: exploratory baseline.
- `scripts/prepare_data.py`, `scripts/run_pilot.py`, `scripts/check_pilot.py`, `scripts/context_shift_baseline.py`: preparation, experiment, independent checks and exploratory comparator.
- `report/content.json` and `report/Project_Paper.md`: editable synopsis and full-paper sources. `scripts/render_iris.py` produces both PDFs. `scripts/check_iris.py` checks the **final files** and writes `report/submission_validation.json`.
- `figures/plot_paper_figures.py`: measured colored charts from saved real-result tables; `figures/concept.png` and `figures/method_corrected.png` are labeled conceptual illustrations, not evidence of measured performance.
- `notes/`: data provenance, frozen protocol, model derivation, literature evidence and IRIS-specific restrictions.

On a fresh clone, download all files in `data/source_manifest_public.tsv` using their exact pinned `url`, verify the listed `bytes` and `sha256`, and place each at `relative_path`. Then run `python scripts/prepare_data.py` to build the response tensor locally and check its independently saved results. Public Tahoe data are CC0; no project data have been fabricated or hidden inside the PDFs. A fresh clone cannot run the full source-row recheck or retrain until these upstream files have been fetched.

Use the scientific Python environment documented in `report/environment.json`. From the repository root, read-only checks are:

```bash
python scripts/check_pilot.py --run results/pilot/main
python scripts/context_shift_baseline.py --verify-only
python figures/plot_paper_figures.py --verify-only
python scripts/check_iris.py
```

To rebuild the PDFs after editing their sources, run `python scripts/render_iris.py` and then `python scripts/check_iris.py`. For an independent repeat of the experiment, write to a *new* results directory; never overwrite `results/pilot/main/`. The source fold-change tensor contains real missing values that must remain masked. No credentials are needed for checking the saved work.
