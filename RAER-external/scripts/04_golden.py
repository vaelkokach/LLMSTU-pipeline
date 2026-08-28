#!/usr/bin/env python
"""Stage 04 — build the three-way human-review payload.

  python scripts/04_golden.py --llmstu ../LLMSTU-pipeline --out ./work

Writes labeling/data.json. Open labeling/index.html (double-click works; the
payload is self-contained), review, then Export golden.json for stage 05.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmstu", default="../LLMSTU-pipeline")
    ap.add_argument("--out", default="./work")
    ap.add_argument("--labeling", default="./labeling")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.llmstu).resolve()))
    from llmstu import schema, sampling
    from raer import golden, labeler

    cfg = labeler.load_config(args.config)
    schema.set_active(cfg.get("caption", {}).get("schema_name", "full"))
    g = cfg.get("golden", {})

    out = Path(args.out)
    golden.build(out / "clip_labels.jsonl", out / "frames", Path(args.labeling),
                 schema, sampling,
                 n_clips=args.n or g.get("n_clips", 200),
                 min_per_class=g.get("min_per_class", 25),
                 hard_frac=g.get("hard_frac", 0.2),
                 stratify_by=g.get("stratify_by", "raer_label"),
                 seed=cfg.get("raer", {}).get("seed", 13))
    print(f"[04] open {Path(args.labeling)/'index.html'} to review")


if __name__ == "__main__":
    main()
