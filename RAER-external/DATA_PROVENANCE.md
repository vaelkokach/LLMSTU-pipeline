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

### Actual layout (audited 2026-08-28, all 87,809 files)

```
RAER/clips /<Session>/<Location>[/<ClassNNN>]/P<NNN>/<NNN>.mp4     2,669
RAER/images/<Session>/<Location>[/<ClassNNN>]/P<NNN>/<NNN>/<k>.jpg 82,488
RAER/cached_features/RAER_images_<Session>_..._P<NNN>_<NNN>.pt      2,640
RAER/annotation/{train,test,train_subset}.txt                           3
RAER/bounding_box/{body,face}.json                                      2
RAER/output_annotation/*.txt|csv                                        7
```

Only `cached_features/` uses the flat underscore naming; the media is a directory
hierarchy. Frame files are `1.jpg, 2.jpg, … 10.jpg` — **numeric, so they must be
sorted numerically**, or temporal order is scrambled.

## Audit findings (stage 00, Kaggle copy, 2026-08-28)

### Annotation format — recovered

`RAER/annotation/{train,test}.txt`, one clip per line:

```
RAER/images/Class/Classroom/Class009/P001/032 32 5
<clip directory>  <n_frames>  <label index 1-5>
```

**train 2,121 + test 528 = 2,649** — exactly the paper's count. The mirror ships
**2,669 mp4**, so **20 clips are unannotated extras**; they must be dropped, not
labeled. 2,640 `cached_features` means 29 clips also lack a cached feature.

### Label index mapping — recovered, not assumed

Undocumented in the copy. Recovered by rank-matching observed class frequencies
against the published percentages; **every class lands within 0.00pp**, so the
mapping is unambiguous:

| idx | class | n | observed | published |
|---|---|---|---|---|
| 1 | neutrality | 1,728 | 65.23% | 65.23% |
| 2 | enjoyment | 161 | 6.08% | 6.08% |
| 3 | confusion | 37 | 1.40% | 1.40% |
| 4 | fatigue | 196 | 7.40% | 7.40% |
| 5 | distraction | 527 | 19.89% | 19.89% |

This also confirms the Kaggle copy is the **full RAER set, not a subset**.
Confusion is 37 clips total — any per-class metric on it is near-anecdotal.

### Subject identity — `P###` is session-local

43 distinct `P###` values, but **25 of them appear under more than one
session/location**, so the index resets per recording. Folding session + location
+ class into it gives **141 identity keys against the paper's 136 subjects** —
close enough to be the right unit, with the small excess most likely a student
recorded in both a class and a self-study session (which this key over-splits).
**Never split on `person`; split on `identity_key`.**

### ⚠ The shipped split is NOT subject-disjoint

The split is 2,121 / 528 (79.9 / 20.1) and class-stratified to within 0.1pp — but:

- clip overlap train ∩ test: **0**
- identity overlap train ∩ test: **132 of 141 — 93.6%**

The paper says it "strives to exclude overlapping individuals"; in this copy that
plainly did not happen. Almost every subject appears on both sides, so a model can
memorise a face in training and be scored on it at test.

**Consequence:** numbers produced on the shipped split are not comparable to a
subject-disjoint benchmark, including the paper's own reported UAR. We build our
own split on `identity_key`, and any comparison against published numbers must
state this. A naive clip-level random split is worse still — 18.9 clips per
identity means it contaminates ~100% of the test set.

### Not the labels

`RAER/output_annotation/*.txt` are a **third party's VLM-generated captions**
("The image features a young man wearing glasses…"), not RAER annotations, and
`test_descriptions.csv` is a 20-byte header with no rows. Ignore all of it.

**Every one of these points must be re-verified against the official copy once
access is granted.** Any number reported from the interim source is provisional
and must be labelled as such.

## Change log

| date | event |
|---|---|
| 2026-08-28 | Access requested from authors; repo created; audit stage written |
| 2026-08-28 | Stage 00 run on the Kaggle manifest: layout, annotation format and label map recovered; **shipped split found to leak 93.6% of subjects** |
