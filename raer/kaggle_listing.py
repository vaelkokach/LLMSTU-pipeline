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


def fetch(dataset: str, max_pages: int = 2000, page_size: int = 200,
          progress_every: int = 25) -> List[Tuple[str, int]]:
    """Return [(relative_path, size_bytes)] for every file in `owner/slug`.

    The API's default page_size is 20, which needs thousands of round trips on a
    dataset this size; the server caps a page at 200, so that is what we ask for.
    Expect roughly one request per 200 files.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    out: List[Tuple[str, int]] = []
    token = None
    for page in range(max_pages):
        resp = api.dataset_list_files(dataset, page_token=token, page_size=page_size)
        files = getattr(resp, "files", None) or []
        if not files:
            break
        for f in files:
            size = getattr(f, "total_bytes", None)
            if size is None:
                size = getattr(f, "size", 0) or 0
            out.append((str(f.name), int(size)))
        token = getattr(resp, "next_page_token", None)
        if progress_every and (page + 1) % progress_every == 0:
            print(f"[listing] {len(out):,} files...", flush=True)
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
    """Download only the named files (use for the small label/split/json files).

    Each file lands under its full relative path. Kaggle writes to the basename,
    and RAER ships both `annotation/train.txt` and `output_annotation/train.txt`
    — flattening would silently leave one overwriting the other.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out_dir = Path(out_dir)
    got = []
    for rel in paths:
        dest_dir = out_dir / Path(rel).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_file(dataset, rel, path=str(dest_dir), force=False, quiet=True)
        for cand in (dest_dir / Path(rel).name, dest_dir / (Path(rel).name + ".zip")):
            if cand.exists():
                if cand.suffix == ".zip":
                    import zipfile
                    with zipfile.ZipFile(cand) as z:
                        z.extractall(dest_dir)
                    cand.unlink()
                    cand = dest_dir / Path(rel).name
                got.append(cand)
                break
    return got
