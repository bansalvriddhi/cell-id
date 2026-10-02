# Literature evidence and positioning

Search conducted 2026-10-02. Default topical-fit priority, no exclusion by year. Question: what prior work constrains claims of novelty and evaluation validity for few-probe identification of a held-out cell context? Difficulty 6/10; initial targeted searches followed by a gap-focused active-learning search. Search provider fallbacks returned several irrelevant general reviews; these were rejected rather than treated as pertinent evidence.

## Read and retained

### Zhang et al. (2026)

**Tahoe-100M: Mapping drug-induced molecular phenotypes at single-cell resolution.** Cell 189(19), 5945–5961.e8. DOI: https://doi.org/10.1016/j.cell.2026.08.035. PMID 42753697.

Published metadata independently verified against Crossref. Abstract read via Europe PMC, exact query `EXT_ID:42753697 AND SRC:MED`, after the PubMed webpage returned a cookie screen. The abstract reports 100 million transcriptomes, 50 cancer cell lines and 1,100 drug-dose conditions. These atlas-level counts are not the size of our extracted pilot. It establishes the atlas's scope and public availability but does not verify exposure duration or the normalization of the author-released differential-expression tables. Dataset-specific methodological evidence and unresolved gaps are in `notes/data_provenance.md`.

### Ahlmann-Eltze, Huber and Anders (2025)

**Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines.** Nature Methods 22, 1657–1661. DOI: https://doi.org/10.1038/s41592-025-02772-6

Metadata verified against Crossref. Full text read from the author-hosted PDF: https://www.huber.embl.de/pub/pdf/s41592-025-02772-6.pdf, pages 1–4. The paper compares five foundation models and two other deep-learning models against simple baselines for genetic perturbation prediction. Its result motivates including simple baselines and delta-expression metrics. It does not demonstrate that simple baselines always win for chemical perturbations or for our held-out-context calibration task.

### Scholkemper and Mukherjee (2026)

**Causal Cellular Context Transfer Learning (C3TL): An Efficient Architecture for Prediction of Unseen Perturbation Effects.** Preprint, arXiv:2603.13051. https://arxiv.org/abs/2603.13051

Read pages 1–3 of the full preprint. C3TL explicitly studies perturbations already observed in other contexts, prediction in new contexts, compute-efficient bulk-expression models, and few-shot/limited-data settings. Therefore few-shot prediction of unseen contexts is not a defensible standalone novelty claim for CELL-ID. This report instead studies acquisition policies for a fixed reconstruction task. A direct C3TL comparison would remain necessary for a broader method paper; the present pilot does not implement C3TL or claim superiority over it.

### Qin, Wessels, Fernandez-Granda and Hao (2024)

**Active learning for efficient discovery of optimal gene combinations in the combinatorial perturbation space.** NAIAD, arXiv:2411.12010v2, AI for New Drug Modalities at NeurIPS 2024 workshop. https://arxiv.org/abs/2411.12010

Read pages 1–3 of the full text. The framework uses an AI–experiment loop to prioritize gene pairs for desired phenotypic effects and also discusses RECOVER. It establishes that active learning for biological perturbation experiments is prior art. CELL-ID's question is reconstruction of responses across an unseen context, rather than discovery of maximally effective combinations. This distinction is an experimental focus, not evidence of a first-ever method.

## Excluded from claim support

- DOI 10.1038/s41592-025-02980-0 resolved to a title but no accessible abstract/full text through the literature resolver. No substantive finding is attributed to it.
- DOI 10.1038/s42256-026-01286-w resolved to bibliographic metadata only. The user supplied a proposed mechanism/structure claim, but it has not been verified from readable primary text and is not used as evidence.
- Promotional claims about competition relevance, novelty, publication prospects and hypothetical five-versus-fifty performance are not empirical results and are omitted.

## Mathematical scope

Greedy D-optimal design is established experimental-design machinery, not introduced by this report. The log-determinant posterior-information objective differs from reduction of total predictive variance. Linear-Gaussian covariance updates do not depend on observed response values, so such a selector is an open-loop design; only its response-conditioned predictor adapts. A small single-time-point pseudobulk pilot is static response-operator identification, not validation of dynamical control or clinical response prediction.

With missing gene estimates, covariance updates may depend on the availability mask revealed by an acquired probe, but not its response magnitudes. This distinction is retained in the final pilot description. Crossref title searches for published versions of C3TL and NAIAD returned unrelated records, so those references remain identified by their verified arXiv versions rather than an invented publication DOI.
