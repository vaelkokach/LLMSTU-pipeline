#!/usr/bin/env python
"""Build the golden-eval sample and the payload for the manual labeling tool.

Takes a stratified sample of pseudo-labeled crops, copies the images next to
the labeling tool, and writes labeling/data.json (image + pre-filled Qwen3-VL
prediction for each item). You then open labeling/index.html to correct them.

  python scripts/03_sample_eval.py --labels ./work/pseudo_labels.jsonl \
      --crops ./work/crops --n 500
"""
import argparse
import base64
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", default="./work/pseudo_labels.jsonl")
    ap.add_argument("--crops", default="./work/crops")
    ap.add_argument("--out", default="./labeling")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--embed", action="store_true",
                    help="base64-embed images into data.json (single-file, no server)")
    args = ap.parse_args()

    cfg = load(args.config)
    n = args.n or cfg.eval.n_samples
    field = cfg.eval.stratify_by
    random.seed(cfg.eval.seed)

    rows = [json.loads(l) for l in Path(args.labels).open() if l.strip()]
    # stratified sample across the chosen field so the golden set is balanced
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get(field, "unknown")].append(r)
    per = max(1, n // max(1, len(buckets)))
    sample = []
    for _, items in buckets.items():
        random.shuffle(items)
        sample.extend(items[:per])
    random.shuffle(sample)
    sample = sample[:n]

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    crops = Path(args.crops)

    data = []
    for i, r in enumerate(sample):
        src = crops / Path(r["crop_path"]).name
        if not src.exists():
            src = crops.parent / r["crop_path"]
        if not src.exists():
            continue
        item = {"id": i, "crop_path": r["crop_path"],
                "pseudo": {k: r.get(k) for k in schema.FIELDS},
                "model_confidence": r.get(schema.MODEL_CONFIDENCE_FIELD)}
        if args.embed:
            b = base64.b64encode(src.read_bytes()).decode()
            item["image"] = f"data:image/jpeg;base64,{b}"
        else:
            shutil.copy(src, img_dir / src.name)
            item["image"] = f"images/{src.name}"
        data.append(item)

    (out / "data.json").write_text(json.dumps({
        "fields": schema.FIELDS, "items": data}, indent=1))
    print(f"[eval] {len(data)} items -> {out/'data.json'}"
          f"  (open {out/'index.html'} to label)")


if __name__ == "__main__":
    main()
