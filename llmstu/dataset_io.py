"""Hugging Face Hub helpers: pull source frames, push enhanced crops+labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional


def download_frames(repo_id: str, subdir: str, dest: Path,
                    token: Optional[str] = None,
                    shards: Optional[List[str]] = None) -> Path:
    """Snapshot the frames subfolder of a dataset repo to `dest`.

    shards: optional list of shard folder names (e.g. ["shard_000"]) to fetch
            ONLY those subfolders instead of the whole dataset. Great for tuning
            — pull ~1 shard (~1.7GB) instead of all 21 (~35GB). Names without a
            path are matched under subdir; full patterns are used as-is.
    """
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if shards:
        patterns = [s if "/" in s else f"{subdir}/{s}/**" for s in shards]
    else:
        patterns = [f"{subdir}/**"]
    local = snapshot_download(
        repo_id=repo_id, repo_type="dataset", token=token,
        allow_patterns=patterns, local_dir=str(dest),
    )
    return Path(local) / subdir


def download_frame_sample(repo_id: str, subdir: str, dest: Path, n: int = 40,
                          token: Optional[str] = None,
                          shard: Optional[str] = None,
                          seed: Optional[int] = 0) -> Path:
    """Download `n` frames sampled at RANDOM (seconds, not minutes).

    Ideal for tuning: pulls ~n individual files via hf_hub_download instead of the
    whole 5000-file shard.
      shard: a shard name (e.g. "shard_000") to sample within it, or None to sample
             across ALL shards (more variety — recommended for tuning).
      seed:  int for a reproducible random sample; None = take the first n (no shuffle).
    Returns the frames root (dest/subdir) for iter_frames.
    """
    import random
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset")
    prefix = f"{subdir}/{shard}/" if shard else f"{subdir}/"
    pool = [f for f in files
            if f.startswith(prefix) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if seed is not None:
        random.seed(seed)
        sel = random.sample(pool, min(n, len(pool)))
    else:
        sel = pool[:n]
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    for f in sel:
        hf_hub_download(repo_id, filename=f, repo_type="dataset",
                        token=token, local_dir=str(dest))
    where = shard or "all shards"
    print(f"[io] sampled {len(sel)} frames from {where} (seed={seed}) -> {dest/subdir}")
    return dest / subdir


def iter_frames(frames_dir: Path, pattern: str = "**/*.jpg") -> Iterator[Path]:
    yield from sorted(Path(frames_dir).glob(pattern))


def build_parquet(labels_jsonl: Path, out_parquet: Path) -> Path:
    """Convert the JSONL labels into a single parquet for HF datasets."""
    import pandas as pd

    rows = [json.loads(l) for l in Path(labels_jsonl).open() if l.strip()]
    df = pd.DataFrame(rows)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"[io] {len(df)} rows -> {out_parquet}")
    return Path(out_parquet)


def upload_dataset(local_dir: Path, repo_id: str, token: Optional[str] = None,
                   private: bool = True) -> str:
    """Create (if needed) and upload the enhanced dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="dataset")
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[io] uploaded -> {url}")
    return url
