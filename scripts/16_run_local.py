#!/usr/bin/env python
"""Full run from a SINGLE zip in an HF Storage Bucket (best for rate-limit issues).

Downloads the one zip (one request, no per-file throttling), extracts it, then
processes shard-by-shard FROM LOCAL DISK: crop -> dedup -> caption -> upload
crops+labels to the Hub -> mark done. No per-shard frame downloads at all.
Resumable via the same _done markers, so a disconnect just resumes remaining shards.

  # find the zip's path inside the bucket:
  python scripts/16_run_local.py --bucket WaelK/LLMSTU --list

  # then run it:
  python scripts/16_run_local.py --bucket WaelK/LLMSTU --zip all_frames.zip \
      --repo WaelK/LLMSTU-crops

After all shards finish, use 14_golden_from_hf.py / 07_eval_golden.py / 13_finalize.py
exactly as with the sharded run.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import caption as caption_mod, crop as crop_mod, dedup, dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--bucket", default="WaelK/LLMSTU")
    ap.add_argument("--zip", default=None, help="zip path inside the bucket")
    ap.add_argument("--repo", default=None, help="crops repo (default: config data.crops_repo)")
    ap.add_argument("--shard-size", type=int, default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--list", action="store_true", help="list bucket contents and exit")
    args = ap.parse_args()

    cfg = load(args.config)
    token = args.token or os.environ.get("HF_TOKEN")

    if args.list:
        print(f"[bucket {args.bucket}] contents:")
        io.list_bucket(args.bucket, token=token)
        return

    assert args.zip, "pass --zip <path in bucket> (use --list to find it)"
    repo = args.repo or cfg.data.crops_repo
    shard_size = args.shard_size or cfg.data.upload_shard_size
    work = Path(cfg.data.workdir)
    frames_dir = work / "frames_all"

    # 1. download + extract the zip ONCE (skip if frames already extracted)
    if not any(frames_dir.glob("**/*.jpg")):
        extracted = io.download_bucket_zip(args.bucket, args.zip, work / "bucket_dl", token=token)
        frames_dir = extracted
    all_frames = sorted(frames_dir.glob("**/*.jpg"))
    print(f"[local] {len(all_frames)} frames on disk")

    # 2. deterministic shard partition (sorted -> chunks of shard_size)
    shards = {}
    for i, f in enumerate(all_frames):
        shards.setdefault("shard_%03d" % (i // shard_size), []).append(f)
    names = sorted(shards)
    todo = [s for s in names if not io.shard_done(repo, s, token)]
    print(f"[local] {len(names)} shards, {len(todo)} remaining: {todo or 'none'}")
    if not todo:
        print("[local] all shards done. Next: scripts/14_golden_from_hf.py")
        return

    cap = caption_mod.load_captioner(cfg.caption)      # load model once
    for shard in names:
        if io.shard_done(repo, shard, token):
            continue
        crops = work / "crops"
        stage = work / "stage"
        manifest = work / "manifest.jsonl"
        manifest_dedup = work / "manifest_dedup.jsonl"
        labels = work / "labels.jsonl"
        shutil.rmtree(crops, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
        for fp in (manifest, manifest_dedup, labels):
            fp.unlink(missing_ok=True)

        crop_mod.run(iter(shards[shard]), crops, cfg.crop, manifest_path=manifest)
        dedup.dedup_manifest(manifest, crops, manifest_dedup,
                             hamming_threshold=cfg.dedup.hamming_threshold,
                             cell_px=cfg.dedup.cell_px, hash_size=cfg.dedup.hash_size)
        caption_mod.run(manifest_dedup, crops, labels, cfg.caption, cap=cap)
        n = io.stage_shard(crops, labels, stage, shard, shard_size=shard_size)
        io.ensure_repo(repo, token=token)
        io.upload_dir(stage, repo, token=token)
        io.mark_shard_done(repo, shard, token=token)
        print(f"[shard {shard}] uploaded {n} crops + labels -> {repo}")
        shutil.rmtree(crops, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)

    print("[local] all shards processed. Next: scripts/14_golden_from_hf.py")


if __name__ == "__main__":
    main()
