"""Fetch a Kaggle dataset's file manifest without downloading the data.

100 GB is an expensive way to find out a dataset is not what you expected. The
Kaggle API paginates a full file listing (name + size), which is enough to run
the clip-count, subject-identity and leakage checks in `raer.audit`.

Needs Kaggle credentials — put KAGGLE_USERNAME / KAGGLE_KEY in Colab Secrets, or
drop kaggle.json at ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def fetch(dataset: str, max_pages: int = 500) -> List[Tuple[str, int]]:
    """Return [(relative_path, size_bytes)] for every file in `owner/slug`."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    out: List[Tuple[str, int]] = []
    token = None
    for _ in range(max_pages):
        resp = api.dataset_list_files(dataset, page_token=token)
        files = getattr(resp, "files", None) or []
        if not files:
            break
        for f in files:
            size = getattr(f, "total_bytes", None)
            if size is None:
                size = getattr(f, "size", 0) or 0
            out.append((str(f.name), int(size)))
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    return out


def save(listing: List[Tuple[str, int]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(listing), encoding="utf-8")
    return path


def load(path: Path) -> List[Tuple[str, int]]:
    return [tuple(x) for x in json.loads(Path(path).read_text(encoding="utf-8"))]


def download_files(dataset: str, paths: List[str], out_dir: Path) -> List[Path]:
    """Download only the named files (use for the small label/split/json files)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for rel in paths:
        api.dataset_download_file(dataset, rel, path=str(out_dir), force=False, quiet=False)
        # Kaggle writes <name> or <name>.zip depending on size
        for cand in (out_dir / Path(rel).name, out_dir / (Path(rel).name + ".zip")):
            if cand.exists():
                got.append(cand)
                break
    return got
