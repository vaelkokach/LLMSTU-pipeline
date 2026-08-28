#!/usr/bin/env python
"""Stage 01 — sample N frames per clip into a caption-ready manifest.

  python scripts/01_frames.py --root /content/raer/RAER --out ./work

Reads the RAER copy, samples `raer.frames_per_clip` frames uniformly from each
clip, materialises them under work/frames with unique basenames, and writes
work/manifest.jsonl for stage 02.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raer import audit, frames, labeler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted RAER directory")
    ap.add_argument("--out", default="./work")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=None, help="override frames per clip")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N clips in sorted order (debug only - biased)")
    ap.add_argument("--sample", type=int, default=None,
                    help="stratified pilot subset of N clips (use this, not --limit)")
    args = ap.parse_args()

    cfg = labeler.load_config(args.config)
    rc = cfg.get("raer", {})
    n = args.n or rc.get("frames_per_clip", 8)
    out = Path(args.out)

    root = Path(args.root)
    clips = audit.index_clips(audit.collect(root))["clips"]
    labels = audit.find_labels(root)
    print(f"[01] {len(clips):,} clips, {len(labels):,} with a RAER label")
    if not labels:
        print("[01] WARNING: no labels recovered — stages 04/05 need them. "
              "Resolve this in the stage 00 report before a full run.")

    if args.sample:
        clips = frames.pilot_subset(clips, labels, args.sample,
                                    seed=rc.get("seed", 13))

    mf = frames.build(clips, root, out / "frames", out / "manifest.jsonl",
                      n_per_clip=n,
                      use_existing_frames=rc.get("use_existing_frames", True),
                      labels=labels, limit=args.limit)
    print(json.dumps(frames.summarise(mf), indent=1))


if __name__ == "__main__":
    main()
