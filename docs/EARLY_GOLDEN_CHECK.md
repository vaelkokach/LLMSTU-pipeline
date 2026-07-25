# Early golden check — validate quality on 1–2 shards before the full run

Captioning all 104k frames (~15–20 GPU-hours) is expensive. Before committing to it,
run this mini-loop on a couple of shards, hand-label a ~200-crop golden set, and get
real accuracy numbers. If they're good, launch the full run; if not, fix the
prompt/schema and repeat — cheap insurance.

Do this **after** you've locked crop/prompt settings in `config.yaml` from tuning.

## 1. Crop + caption just 2 shards

```python
from pathlib import Path
from llmstu import crop as crop_mod, caption as caption_mod, dataset_io as io
from llmstu.config import load
cfg = load('config.yaml')

# pull two full shards (~10k frames)
frames = io.download_frames(cfg.data.source_repo, cfg.data.source_subdir,
                            Path('work/frames_dl'), token=None,
                            shards=['shard_000', 'shard_001'])

crop_mod.run(io.iter_frames(frames, cfg.data.frames_glob), Path('work/crops'),
             cfg.crop, manifest_path=Path('work/crops_manifest.jsonl'))
# Cap captioning for the early check — a wide classroom frame yields ~10 crops, so
# 2 shards is ~100k crops. You only need a few hundred labeled for a 200-crop golden
# set. limit=800 keeps this fast; drop the limit only for the real full run.
caption_mod.run(Path('work/crops_manifest.jsonl'), Path('work/crops'),
                Path('work/pseudo_labels.jsonl'), cfg.caption, limit=800)
```

## 2. Build a ~200-crop golden sample

```bash
python scripts/03_sample_eval.py --labels work/pseudo_labels.jsonl \
    --crops work/crops --n 200 --embed
```

`--embed` makes `labeling/data.json` self-contained. Download the `labeling/` folder.

## 3. Hand-label

Open `labeling/index.html` (double-click works with `--embed`). For each crop the
Qwen prediction is pre-filled: press **v** if it's already correct, or fix the fields
(changed ones highlight amber). Focus your attention on the fields the tuning report
showed as shaky (e.g. `gaze_direction`, `attention_target`). Click **Export
golden.json** when done, and drop it back into `labeling/`.

## 4. Score pseudo-labels vs golden

```bash
python scripts/07_eval_golden.py --golden labeling/golden.json
```

Read `work/eval/golden_report.html`: per-field accuracy + example disagreements.

## 5. Decide

- **Accuracy good enough** (e.g. the fields you care about are >~85%, shaky fields
  acceptable) → launch the full run (`LLMSTU_pipeline_colab.ipynb` / `docs/GPU_95GB.md`).
- **Some fields weak** → tighten them in `llmstu/prompts.py` (clearer definitions,
  add an example) or simplify categories in `llmstu/schema.py`, re-caption these 2
  shards, and re-score. Cheap to iterate here; expensive after the full run.

Keep this 200-crop golden set — you can fold it into the final golden set (or expand
it to ~500) after the full run, and it uploads with `scripts/04_upload.py --golden`.
