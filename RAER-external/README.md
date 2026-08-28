# RAER external task

Pseudo-label the **RAER** academic-emotion dataset with the *same* labeler used for
LLMSTU, then compare three views of the same clips:

```
RAER's own label   vs   our Qwen3.5 visible-cue labels   vs   a human adjudication
```

This is deliberately **not** a label mapping. RAER's classes (enjoyment, neutrality,
confusion, fatigue, distraction) are inferred affective states; ours are observable
cues (`gaze_direction`, `hand_state`, `posture`, `phone_visible`, …). Forcing one onto
the other would bury the interesting result. Treating RAER as a separate external task
and measuring where the two taxonomies agree is the interesting result.

Read **`DATA_PROVENANCE.md` first** — the dataset is licence-gated and nothing derived
from its media may be published.

## Why there is no labeling code in this repo

There isn't any, on purpose. `raer.labeler` imports `llmstu.caption`, `llmstu.schema`,
`llmstu.prompts` and `llmstu.sampling` from the LLMSTU-pipeline checkout. Same model,
same prompt, same field set, same JSON coercion, same resumable JSONL, same golden-set
sampler — because it is literally the same code, not a copy that drifts. "Identical
pseudo-labeling" is enforced by construction.

Colab clones both repos side by side; `config.yaml` pins the LLMSTU-pipeline commit
that produced each run.

## Stages

| stage | what | new code? |
|---|---|---|
| **00 audit** | inventory, subject identity, class balance, split, leakage | yes — `raer/audit.py` |
| **06 split** | subject-disjoint train/test — the shipped one leaks | yes — `raer/split.py` |
| **01 frames** | clip → 8 sampled frames, as an LLMSTU-format manifest | yes — `raer/frames.py` |
| **02 label** | pseudo-label the frames | no — `llmstu.caption.run()` |
| **03 aggregate** | frame cue labels → one verdict per clip | yes — `raer/aggregate.py` |
| **04 golden** | 200 clips for human audit, stratified over RAER's label | `llmstu.sampling` + a 3-way UI |
| **05 compare** | RAER vs Qwen vs human — κ, cue lift, adjudication | yes — `raer/compare.py` |

All implemented. Notebooks: `RAER_00_audit_colab.ipynb`, then
`RAER_01_pipeline_colab.ipynb`.

## What stage 00 found

Run on all 87,809 files via Kaggle's manifest — no download needed. Full detail in
`DATA_PROVENANCE.md`; the load-bearing results:

- **2,649 annotated clips**, exactly matching the paper. The mirror ships 2,669 mp4, so
  20 are unannotated extras to drop.
- **Label indices decoded**: `1 neutrality, 2 enjoyment, 3 confusion, 4 fatigue,
  5 distraction`, recovered by matching class frequencies to the published
  distribution — agreeing to **0.00pp**, which also proves this is the full set.
- **`P###` is session-local.** Split on `identity_key`, never on `person`.
- **The shipped split is not subject-disjoint**: zero clip overlap, but **132 of 141
  subjects (93.6%) appear in both train and test**. Numbers from it are not comparable
  to a subject-disjoint benchmark. `scripts/06_split.py` builds a replacement that is
  disjoint, same 79.9/20.1 ratio, class balance within 0.29pp.
- `output_annotation/*` is a third party's VLM captioning, not RAER labels.

## Run it

```bash
python scripts/00_audit.py --kaggle bearmn/raer-video-emotion-dataset --out ./work/audit
```

```bash
python scripts/06_split.py --annotations /content/raer --out ./work/split
```

```bash
python scripts/01_frames.py --root /content/raer/RAER --out ./work
```

```bash
python scripts/02_label.py --llmstu ../LLMSTU-pipeline --out ./work --limit 96
```

```bash
python scripts/03_aggregate.py --llmstu ../LLMSTU-pipeline --out ./work
```

```bash
python scripts/04_golden.py --llmstu ../LLMSTU-pipeline --out ./work
```

```bash
python scripts/05_compare.py --llmstu ../LLMSTU-pipeline --golden labeling/golden.json
```

### What the audit will not do

It will not guess. The Kaggle mirror ships no data card, so anything the audit cannot
establish from the files themselves is reported as `UNVERIFIED` rather than filled in
from the paper. In particular it **tests** whether `P###` is a global subject id instead
of assuming it, and falls back to a conservative `session/location/class/person`
identity key that over-splits rather than risking a subject leaking across the
train/test boundary.

## Reference

Zhao, Xuan, Lou, Yu, Yang. *Context-Aware Academic Emotion Dataset and Benchmark.*
ICCV 2025. [arXiv:2507.00586](https://arxiv.org/abs/2507.00586)
