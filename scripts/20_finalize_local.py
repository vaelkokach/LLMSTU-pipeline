#!/usr/bin/env python
"""Step 9, LOCAL: build the HF-ready metadata + parquet from extracted-on-disk shards.

Writes crops/metadata.jsonl (HF imagefolder: each crop image paired with its labels)
next to your local crops, plus pseudo_labels.parquet and (optionally) copies the
golden set. Then you can either load it locally with
`datasets.load_dataset("imagefolder", data_dir=<crops-root>)` or upload it to HF with
--repo (runs from your PC, off GPU).

  python scripts/20_finalize_local.py --labels-dir ./data/labels --crops-root ./data/crops \
      --golden labeling/golden.json
  # optional, to publish:  --repo WaelK/LLMSTU-crops
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import schema, dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--crops-root", required=True,
                    help="local crops root; metadata.jsonl is written here")
    ap.add_argument("--golden", default="labeling/golden.json")
    ap.add_argument("--repo", default=None, help="if set, upload crops-root to this HF repo")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    load(args.config)
    crops_root = Path(args.crops_root)

    lp = Path(args.labels_dir)
    files = [lp] if lp.is_file() else sorted(lp.glob("**/*.jsonl"))
    rows = []
    for f in files:
        rows += [json.loads(l) for l in f.open() if l.strip()]

    # 1. metadata.jsonl next to the crops (HF imagefolder convention)
    keep = list(schema.FIELDS) + [schema.MODEL_CONFIDENCE_FIELD, "src_frame"]
    meta_path = crops_root / "metadata.jsonl"
    n = 0
    with meta_path.open("w") as mf:
        for r in rows:
            fn = r.get("file_name")
            if not fn:
                continue
            rec = {"file_name": fn}
            for k in keep:
                if k in r:
                    rec[k] = r[k]
            mf.write(json.dumps(rec) + "\n")
            n += 1
    print(f"[finalize] wrote {n} rows -> {meta_path}")

    # 2. full parquet + golden copy (next to crops)
    tmp = crops_root.parent / "pseudo_labels_all.jsonl"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    io.build_parquet(tmp, crops_root.parent / "pseudo_labels.parquet")
    if Path(args.golden).exists():
        shutil.copy(args.golden, crops_root.parent / "golden_eval.json")

    # 3. optional upload (crops + metadata) to HF, from your PC
    if args.repo:
        token = args.token or os.environ.get("HF_TOKEN")
        io.ensure_repo(args.repo, token=token)
        io.upload_dir(crops_root.parent, args.repo, token=token)
        print(f"[finalize] uploaded -> https://huggingface.co/datasets/{args.repo}")
    else:
        print("[finalize] local dataset ready. Load with:\n"
              f'  from datasets import load_dataset\n'
              f'  ds = load_dataset("imagefolder", data_dir="{crops_root}")')


if __name__ == "__main__":
    main()
