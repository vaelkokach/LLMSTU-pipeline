#!/usr/bin/env python
"""Build a subject-disjoint train/test split.

  python scripts/06_split.py --annotations ./work/meta --out ./work/split

The shipped RAER split shares 93.6% of subjects between train and test, so it
cannot be used to report generalisation. This replaces it.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raer import audit, naming, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True,
                    help="directory containing RAER/annotation/*.txt")
    ap.add_argument("--out", default="./work/split")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    labels = audit.find_labels(Path(args.annotations))
    if not labels:
        raise SystemExit(f"no RAER annotations found under {args.annotations}")

    rows = []
    for stem, cls in labels.items():
        f = naming.parse(stem)
        if f:
            rows.append({"clip_stem": stem, "raer_label": cls,
                         "identity_key": f["identity_key"]})

    r = split.build(rows, test_frac=args.test_frac, seed=args.seed)
    ev = r["evaluation"]
    print(f"\nsubject-disjoint: {ev['subject_disjoint']} "
          f"({ev['shared_identities']} shared identities)")
    print(f"max class delta : {ev['max_class_delta_pp']}pp")
    for x in ev["per_class"]:
        print(f"  {x['class']:<12} train {x['train_n']:>5} ({x['train_pct']:5.2f}%)  "
              f"test {x['test_n']:>4} ({x['test_pct']:5.2f}%)  delta {x['delta_pp']:+.2f}")
    split.save(r, Path(args.out))


if __name__ == "__main__":
    main()
