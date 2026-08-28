# Data provenance — RAER

This file is the audit trail for which copy of RAER produced which numbers.
Keep it current; the thesis cites it.

## The dataset

**RAER** (Real-world Academic Emotion Recognition) — Zhao, Xuan, Lou, Yu, Yang,
*Context-Aware Academic Emotion Dataset and Benchmark*, ICCV 2025.
[arXiv:2507.00586](https://arxiv.org/abs/2507.00586) ·
project page <https://zgsfer.github.io/RAER/>

2,649 clips, 136 university students (91 M / 45 F, ages 18–25), recorded in
classrooms, libraries, laboratories and dormitories. Each clip is the middle
5–15 s of a segment. Two label sets: coarse (engaged / distracted) and fine
(enjoyment, neutrality, confusion, fatigue, distraction), ~5 annotators each,
majority vote, Fleiss' κ = 0.832. Clips whose coarse and fine labels disagreed
(6.5%) were discarded by the authors.

**Not to be confused with CAER** (Lee et al., ICCV 2019), a basic-emotion dataset
of TV-show clips. The RAER project page lives under a `/CAER` path, which is the
source of the confusion. Only RAER is used here.

## Terms of use

The official terms are **non-commercial research only**, and explicitly prohibit
redistribution:

> Publish, distribute, or share the dataset or any substantial portion of it with
> third parties outside your organization.

Access is granted by email to the corresponding author, and is restricted to
**full-time professors or non-student research staff** at recognised
universities; files are password-protected.

### Consequences for this repo

1. **Nothing derived from RAER media is published.** The LLMSTU pipeline ends by
   pushing crops to a public Hugging Face repo; that stage is **disabled** here.
   Pseudo-labels, metrics and reports may be shared. Frames, clips and crops may
   not — not to Hugging Face, not via a shared Drive link, not as an appendix.
2. **Storage stays private.** Working copies live on the Colab runtime disk and,
   for derived artifacts only, a private Drive folder. No link sharing.
3. **The source is a config switch**, not a hardcoded path — see `source:` in
   `config.yaml` — so results can be regenerated from the official copy without
   touching pipeline code.

## Source of record

| | |
|---|---|
| **Official access** | Requested by supervisor — awaiting approval (as of 2026-08-28) |
| **Interim source** | Kaggle mirror `bearmn/raer-video-emotion-dataset`, version 1, uploaded 2025-11-24 |

### Interim source — known issues

The Kaggle mirror is a third-party re-upload with **no data card, no description,
no citation, and no provenance statement**, and it is labelled **CC0: Public
Domain**. That label is not valid: an uploader cannot place a gated,
non-commercial, consent-limited dataset into the public domain. Kaggle usability
score 2.94.

Contents (100.35 GB, 87,809 files, from Kaggle's file manifest):

| type | count | size | note |
|---|---|---|---|
| `.mp4` | 2,669 | 44.4 GB | the clips — 20 more than the paper's 2,649 |
| `.jpg` | 82,488 | 55.9 GB | pre-extracted frames, ~31 per clip |
| `.pt` | 2,640 | 10.5 MB | `cached_features/` — someone's precomputed CLIP features, unusable for us |
| `.txt` | 9 | 8.3 MB | possibly split lists — unverified |
| `.json` | 2 | 13.2 MB | columns `caption`, `image_path` |
| `.csv` | 1 | **20 bytes** | header only, empty |

Naming convention observed in `cached_features/`:

```
RAER_images_SelfStudy_Laboratory_P040_012
RAER_images_Class_Classroom_Class002_P001_021
```

→ modality, session type, environment, class id (classroom only), person index,
clip index. **`P###` is presumed session-local, not a global subject id** — see
`raer/audit.py`, which tests this rather than assuming it.

**Every one of these points must be re-verified against the official copy once
access is granted.** Any number reported from the interim source is provisional
and must be labelled as such.

## Change log

| date | event |
|---|---|
| 2026-08-28 | Access requested from authors; repo created; audit stage written |
