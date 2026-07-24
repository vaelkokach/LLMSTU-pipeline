#!/usr/bin/env python
"""Package crops + labels and push to a new HF dataset repo.

  python scripts/04_upload.py --crops ./work/crops \
      --labels ./work/pseudo_labels.jsonl --repo WaelK/LLMSTU-crops
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--crops", default="./work/crops")
    ap.add_argument("--labels", default="./work/pseudo_labels.jsonl")
    ap.add_argument("--golden", default="./labeling/golden.json",
                    help="exported golden labels from the labeling tool (optional)")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    cfg = load(args.config)
    repo = args.repo or cfg.data.crops_repo
    stage = Path(cfg.data.workdir) / "dataset"
    (stage / "crops").mkdir(parents=True, exist_ok=True)

    # copy crops + build parquet of pseudo-labels
    for p in Path(args.crops).glob("*.jpg"):
        shutil.copy(p, stage / "crops" / p.name)
    io.build_parquet(Path(args.labels), stage / "pseudo_labels.parquet")
    if Path(args.golden).exists():
        shutil.copy(args.golden, stage / "golden_eval.json")

    io.upload_dataset(stage, repo, token=args.token, private=not args.public)


if __name__ == "__main__":
    main()
