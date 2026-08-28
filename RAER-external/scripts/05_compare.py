#!/usr/bin/env python
"""Stage 05 — RAER label vs our cue labels vs human adjudication.

  python scripts/05_compare.py --llmstu .. --golden labeling/golden.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmstu", default="..")
    ap.add_argument("--out", default="./work")
    ap.add_argument("--golden", default="labeling/golden.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.llmstu).resolve()))
    from llmstu import schema
    from raer import compare, labeler

    cfg = labeler.load_config(args.config)
    schema.set_active(cfg.get("caption", {}).get("schema_name", "full"))

    out = Path(args.out)
    r = compare.run(out / "clip_labels.jsonl", out / "compare", schema,
                    golden_path=Path(args.golden))
    co = r["coarse"]
    if co.get("n"):
        print(f"\ncoarse agreement {co['raw_agreement']*100:.1f}%  kappa {co['cohens_kappa']}")
        print(f"per-class recall {co['per_class_recall']}")


if __name__ == "__main__":
    main()
