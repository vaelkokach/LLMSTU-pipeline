"""Stage 04 — build the human-audit payload for the three-way comparison.

Each item shows a reviewer one clip as a strip of frames, RAER's own label, and
our aggregated Qwen cue labels, and asks for two independent judgements:

  * **cue correction** — fix any of our observable-cue fields that are wrong
  * **RAER verdict** — does RAER's academic-emotion label fit this clip?

Keeping those separate is the point. Merging them would make it impossible to
tell "Qwen misread the gaze" from "RAER's label is arguable", which is exactly
the distinction the external-task comparison exists to measure.

Sampling reuses `llmstu.sampling.select` unchanged, stratified over `raer_label`
so confusion (37 clips in the whole dataset) actually appears, plus a hard subset
drawn from the lowest mean model confidence.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

THUMB_WIDTH = 360
THUMB_QUALITY = 78


def _thumb(path: Path, width: int = THUMB_WIDTH) -> Optional[str]:
    """Downscale a frame to a data: URI. Full-size frames would make the payload
    hundreds of MB; the reviewer is judging pose and gaze, not pixel detail."""
    from PIL import Image

    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=THUMB_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build(clip_labels: Path, frames_dir: Path, out_dir: Path, schema, sampling,
          n_clips: int = 200, min_per_class: int = 25, hard_frac: float = 0.2,
          strip: int = 4, seed: int = 13,
          stratify_by: str = "raer_label") -> Path:
    """Write labeling/data.json for the three-way review UI."""
    rows = [json.loads(l) for l in Path(clip_labels).open(encoding="utf-8") if l.strip()]
    if not rows:
        raise SystemExit(f"no clips in {clip_labels}")

    picks = sampling.select(rows, n_clips, [stratify_by], min_per_class, hard_frac, seed=seed)

    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cue_fields = {k: v for k, v in schema.FIELDS.items() if v["type"] in ("enum", "bool")}
    items: List[Dict[str, Any]] = []
    missing: List[str] = []
    for k, (i, group) in enumerate(picks):
        r = rows[i]
        stem = r["clip_stem"]
        # stage 01 wrote <clip_stem>_fNN.jpg, flat
        avail = sorted(frames_dir.glob(f"{stem}_f*.jpg"))
        if not avail:
            missing.append(stem)
            continue
        step = max(1, len(avail) // strip)
        chosen = avail[::step][:strip]
        thumbs = [t for t in (_thumb(p) for p in chosen) if t]
        if not thumbs:
            continue
        items.append({
            "id": k,
            "clip_stem": stem,
            "group": group,
            "identity_key": r.get("identity_key", ""),
            "raer_label": r.get("raer_label", ""),
            "frames": thumbs,
            "n_frames": r.get("n_frames"),
            "pseudo": {f: r.get(f) for f in cue_fields},
            "shares": {f: r.get(f"{f}_share") for f in cue_fields},
            "caption": r.get("caption", ""),
            "model_confidence": r.get("model_confidence"),
            "cue_instability": r.get("cue_instability"),
        })

    payload = {
        "fields": cue_fields,
        "raer_classes": ["enjoyment", "neutrality", "confusion", "fatigue", "distraction"],
        "items": items,
    }
    out = out_dir / "data.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    n_repr = sum(1 for it in items if it["group"] == "representative")
    print(f"[golden] {len(items)} clips ({n_repr} representative + "
          f"{len(items) - n_repr} hard), {out.stat().st_size / 1e6:.1f} MB -> {out}")
    if missing:
        print(f"[golden] WARNING: asked for {n_clips}, dropped {len(missing)} clips with no "
              f"frames under {frames_dir} (e.g. {missing[:3]}). Stage 01 has not covered "
              f"the whole set — the review sample is smaller and less balanced than requested.")
    by_cls: Dict[str, int] = {}
    for it in items:
        by_cls[it["raer_label"] or "(none)"] = by_cls.get(it["raer_label"] or "(none)", 0) + 1
    print(f"[golden] RAER class coverage: {by_cls}")
    return out
