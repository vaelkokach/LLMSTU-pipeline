#!/usr/bin/env python
"""Step 7, LOCAL: build the golden-eval sample from extracted-on-disk shards.

Reads all labels/*.jsonl and the local crops (crops/shard_XXX/part_XXX/*.jpg), samples
(stratified representative + hard subset), and writes the self-contained labeling
payload. No HF access.

  python scripts/19_golden_local.py --labels-dir ./data/labels --crops-root ./data/crops \
      --n 1500 --stratify-fields engagement_level,activity,gaze_direction \
      --min-per-class 40 --hard-frac 0.2
"""
import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import schema, sampling


def _resolve(crops_root: Path, row):
    fn = row.get("file_name")
    if fn and (crops_root / fn).exists():
        return crops_root / fn
    name = Path(row.get("crop_path", fn or "")).name
    hits = list(crops_root.glob(f"**/{name}"))          # fallback: find by basename
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels-dir", required=True,
                    help="dir containing shard_*.jsonl (or a single .jsonl file)")
    ap.add_argument("--crops-root", required=True,
                    help="local crops root (contains shard_XXX/part_XXX/*.jpg)")
    ap.add_argument("--out", default="./labeling")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--stratify-fields", default="engagement_level,activity,gaze_direction")
    ap.add_argument("--min-per-class", type=int, default=40)
    ap.add_argument("--hard-frac", type=float, default=0.2)
    args = ap.parse_args()

    cfg = load(args.config)
    n_total = args.n or cfg.eval.n_samples
    fields = [f.strip() for f in args.stratify_fields.split(",") if f.strip()]
    crops_root = Path(args.crops_root)

    lp = Path(args.labels_dir)
    files = [lp] if lp.is_file() else sorted(lp.glob("**/*.jsonl"))
    rows = []
    for f in files:
        rows += [json.loads(l) for l in f.open() if l.strip()]
    print(f"[golden] {len(rows)} labels from {len(files)} file(s)")

    picks = sampling.select(rows, n_total, fields, args.min_per_class,
                            args.hard_frac, seed=cfg.eval.seed)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data, missing = [], 0
    for k, (i, group) in enumerate(picks):
        r = rows[i]
        p = _resolve(crops_root, r)
        if p is None:
            missing += 1
            continue
        b = base64.b64encode(p.read_bytes()).decode()
        data.append({"id": k, "crop_path": r.get("crop_path", r.get("file_name")),
                     "group": group, "pseudo": {kk: r.get(kk) for kk in schema.FIELDS},
                     "model_confidence": r.get(schema.MODEL_CONFIDENCE_FIELD),
                     "image": f"data:image/jpeg;base64,{b}"})
    (out / "data.json").write_text(json.dumps({"fields": schema.FIELDS, "items": data}, indent=1))
    nr = sum(1 for d in data if d["group"] == "representative")
    nh = sum(1 for d in data if d["group"] == "hard")
    print(f"[golden] {len(data)} items ({nr} representative + {nh} hard) -> {out/'data.json'}"
          + (f"  ({missing} crops not found)" if missing else ""))
    print(f"[golden] open {out}/index.html to label")


if __name__ == "__main__":
    main()
