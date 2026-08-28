#!/usr/bin/env python
"""Stage 02 — pseudo-label the sampled frames with the LLMSTU labeler.

  python scripts/02_label.py --llmstu .. --out ./work

Refuses to run if this repo's caption settings have drifted from the ones locked
in LLMSTU-pipeline/config.yaml. Resumable: re-running continues where it stopped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmstu", default="..",
                    help="LLMSTU-pipeline checkout — the labeler comes from here")
    ap.add_argument("--out", default="./work")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="frames to label (smoke test)")
    args = ap.parse_args()

    llmstu_root = Path(args.llmstu).resolve()
    sys.path.insert(0, str(llmstu_root))

    from raer import labeler

    out = Path(args.out)
    labeler.run(manifest=out / "manifest.jsonl",
                frames_dir=out / "frames",
                out_labels=out / "frame_labels.jsonl",
                our_cfg=labeler.load_config(args.config),
                llmstu_root=llmstu_root,
                limit=args.limit)


if __name__ == "__main__":
    main()
