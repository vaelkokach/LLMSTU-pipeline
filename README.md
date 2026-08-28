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
| **01 frames** | clip → N sampled frames, written as an LLMSTU-format manifest | yes — `raer/frames.py` |
| **02 label** | pseudo-label the frames | no — `llmstu.caption.run()` |
| **03 aggregate** | frame cue labels → one verdict per clip | yes — `raer/aggregate.py` |
| **04 golden** | sample 100–200 clips for human audit, stratified over RAER's label | `llmstu.sampling` + a 3-way UI |
| **05 compare** | RAER vs Qwen vs human — cross-tab, per-class, κ | yes — `raer/compare.py` |

Only 00 is implemented so far. The rest wait on the audit gate.

## Stage 00 — run it

Manifest-only (no download, ~1 minute):

```bash
python scripts/00_audit.py --kaggle bearmn/raer-video-emotion-dataset --out ./work/audit
```

Full copy, once downloaded:

```bash
python scripts/00_audit.py --root /content/raer/RAER --out ./work/audit
```

In Colab: `notebooks/RAER_00_audit_colab.ipynb`.

The report flags each finding as OK / UNVERIFIED / LEAKY. Resolve every non-OK line and
record the outcome in `DATA_PROVENANCE.md` before moving to stage 01.

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
