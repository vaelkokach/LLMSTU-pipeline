"""Stage 02 — pseudo-label RAER frames with the LLMSTU labeler.

This module contains no labeling logic and must not grow any. It builds an
`llmstu.config.CaptionConfig` from our `config.yaml` and hands off to
`llmstu.caption.run()`. Same model, same prompt, same schema, same JSON
coercion, same resumable JSONL — because it is the same function.

`assert_identical()` compares our `caption:` block against the values locked in
LLMSTU-pipeline's own config.yaml and refuses to run on a mismatch, so the two
tasks cannot silently drift apart.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Every caption field that must match LLMSTU exactly for results to be comparable.
LOCKED_FIELDS = (
    "model_id", "backend", "dtype", "load_in_4bit", "attn_implementation",
    "prompt_name", "schema_name", "enable_thinking", "max_new_tokens",
    "max_pixels", "temperature",
)
# batch_size affects throughput, not output, so it is allowed to differ.


def load_config(path: Path = Path("config.yaml")) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _effective(caption_block: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a caption: yaml block against CaptionConfig's defaults.

    Compare effective values, not raw yaml — LLMSTU's config.yaml omits fields
    that fall back to a dataclass default (schema_name among them), and a raw
    dict comparison would report those omissions as differences.
    """
    try:
        from llmstu.config import CaptionConfig
    except ImportError as e:
        # Degrading to a raw-dict comparison here would report every field the
        # yaml omits as a difference — a spurious "configs differ" that sends you
        # hunting a problem that does not exist. Fail on the real cause instead.
        raise SystemExit(
            f"cannot import llmstu ({e}). The drift check compares effective config "
            f"values, so it needs the package. Put the LLMSTU-pipeline root on "
            f"sys.path (the scripts do this via --llmstu, default '..').") from e
    cc = CaptionConfig()
    for k, v in (caption_block or {}).items():
        if hasattr(cc, k):
            setattr(cc, k, v)
    return {k: getattr(cc, k) for k in LOCKED_FIELDS}


def assert_identical(our_cfg: Dict[str, Any], llmstu_config_path: Path) -> Dict[str, Any]:
    """Fail loudly if our caption settings differ from LLMSTU's locked ones."""
    theirs = _effective((yaml.safe_load(Path(llmstu_config_path).read_text(encoding="utf-8"))
                         or {}).get("caption", {}))
    ours = _effective(our_cfg.get("caption", {}))
    diffs = {k: {"raer": ours.get(k), "llmstu": theirs.get(k)}
             for k in LOCKED_FIELDS if ours.get(k) != theirs.get(k)}
    if diffs:
        raise SystemExit(
            "caption config differs from LLMSTU-pipeline — the two tasks would not be "
            f"comparable. Fix config.yaml, do not tune here:\n{yaml.safe_dump(diffs)}")
    return theirs


def pipeline_commit(llmstu_root: Path) -> str:
    """Record which LLMSTU-pipeline revision produced this run."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(llmstu_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def build_caption_config(our_cfg: Dict[str, Any]):
    from llmstu.config import CaptionConfig

    cc = CaptionConfig()
    for k, v in (our_cfg.get("caption") or {}).items():
        if hasattr(cc, k):
            setattr(cc, k, v)
    return cc


def run(manifest: Path, frames_dir: Path, out_labels: Path,
        our_cfg: Dict[str, Any], llmstu_root: Path,
        limit: Optional[int] = None) -> Path:
    """Label `manifest` and append schema-valid rows to `out_labels`."""
    from llmstu import caption as llmstu_caption

    assert_identical(our_cfg, Path(llmstu_root) / "config.yaml")
    cc = build_caption_config(our_cfg)
    sha = pipeline_commit(llmstu_root)
    print(f"[label] llmstu-pipeline @ {sha or 'unknown'} | {cc.model_id} "
          f"| prompt={cc.prompt_name} schema={cc.schema_name}")

    out_labels = Path(out_labels)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    llmstu_caption.run(Path(manifest), Path(frames_dir), out_labels, cc, limit=limit)

    (out_labels.parent / "run_provenance.yaml").write_text(yaml.safe_dump({
        "llmstu_pipeline_commit": sha,
        "caption": {k: getattr(cc, k) for k in LOCKED_FIELDS},
        "batch_size": cc.batch_size,
        "manifest": str(manifest),
    }), encoding="utf-8")
    return out_labels
