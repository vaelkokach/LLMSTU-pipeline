#!/usr/bin/env python
"""Audit a RAER copy before labeling anything.

  python scripts/00_audit.py --root /content/raer/RAER --out ./work/audit

Writes work/audit/audit.json + audit_report.html. Read the report and resolve
every UNVERIFIED / LEAKY flag before running stage 01.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raer import audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="extracted RAER directory (full audit)")
    ap.add_argument("--kaggle", help="owner/slug — audit from the file manifest, no download")
    ap.add_argument("--listing", help="path to a saved manifest json")
    ap.add_argument("--out", default="./work/audit")
    args = ap.parse_args()

    if args.root:
        r = audit.run(root=Path(args.root), out_dir=Path(args.out))
    elif args.kaggle or args.listing:
        from raer import kaggle_listing
        if args.listing:
            listing = kaggle_listing.load(Path(args.listing))
            src = args.listing
        else:
            listing = kaggle_listing.fetch(args.kaggle)
            kaggle_listing.save(listing, Path(args.out) / "listing.json")
            src = f"kaggle:{args.kaggle}"
        print(f"[audit] {len(listing):,} files in manifest")
        r = audit.run(listing=listing, out_dir=Path(args.out), source=src)
    else:
        ap.error("pass --root, --kaggle owner/slug, or --listing path.json")

    print("\n=== gate checks ===")
    print(f"clips           {r['clips']['n_clips']:,} "
          f"(paper {r['clips']['paper_n_clips']:,}, delta {r['clips']['delta_vs_paper']:+d})")
    print(f"identity        {r['identity']['verdict']}")
    print(f"distribution    {r['distribution']['status']}")
    print(f"official split  {r['splits'].get('status')}")
    print(f"leakage         {r['leakage'].get('verdict', r['leakage'].get('status'))}")


if __name__ == "__main__":
    main()
