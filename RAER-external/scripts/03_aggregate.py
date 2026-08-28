#!/usr/bin/env python
"""Stage 03 — frame cue labels -> one verdict per clip.

  python scripts/03_aggregate.py --llmstu .. --out ./work
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmstu", default="..")
    ap.add_argument("--out", default="./work")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--min-frames", type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.llmstu).resolve()))
    from llmstu import schema
    from raer import aggregate, labeler

    cfg = labeler.load_config(args.config)
    schema.set_active(cfg.get("caption", {}).get("schema_name", "full"))

    out = Path(args.out)
    cl = aggregate.run(out / "frame_labels.jsonl", out / "clip_labels.jsonl",
                       schema, min_frames=args.min_frames)
    print(json.dumps(aggregate.summarise(cl, schema), indent=1))


if __name__ == "__main__":
    main()
