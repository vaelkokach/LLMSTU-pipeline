"""Stage 03 — frame-level cue labels -> one verdict per clip.

RAER labels a clip; we label frames. This module closes that gap, and the way it
closes it is a modelling choice worth stating rather than burying:

* **enum fields** (activity, gaze_direction, posture, hand_state, …) take a
  **majority vote** across the clip's frames. A student's posture over 5-15 s has
  a modal value; outvoting the odd bad frame is the point.

* **bool fields split into two kinds.** `phone_visible` and `laptop_visible` are
  *object presence* — a phone seen in one frame is a phone that is there, so they
  use **ANY**. `talking` and `occluded` are *states* that hold or don't for most
  of the clip, so they use **majority**. Using majority for phone_visible would
  systematically erase exactly the cue RAER's distraction class turns on.

* **`model_confidence`** averages, and every field also keeps its **vote share**,
  so a 5/8 majority is never mistaken for an 8/8 one downstream.

The by-product is `cue_instability`: the mean disagreement across enum fields
within a clip. High instability means the cues changed during the clip — shifting
gaze, fidgeting hands — which is itself one of RAER's stated distraction
descriptors, so it is kept as a feature rather than discarded as noise.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bools where presence in ANY frame is the honest reading (see module docstring).
ANY_BOOL_FIELDS = ("phone_visible", "laptop_visible")

# Manifest fields carried through unchanged from stage 01.
CARRY = ("clip_stem", "identity_key", "session", "location",
         "class_id", "person", "raer_label", "source")


def _fields(schema) -> Dict[str, Dict[str, Any]]:
    return schema.FIELDS


def aggregate_clip(rows: List[Dict[str, Any]], schema) -> Dict[str, Any]:
    """Collapse one clip's frame labels into a single record."""
    out: Dict[str, Any] = {k: rows[0].get(k) for k in CARRY}
    out["n_frames"] = len(rows)

    shares: List[float] = []
    per_field_share: Dict[str, float] = {}
    for name, spec in _fields(schema).items():
        vals = [r.get(name) for r in rows if r.get(name) is not None]
        if not vals:
            out[name] = None
            out[f"{name}_share"] = 0.0
            continue

        if spec["type"] == "bool":
            truthy = [bool(v) for v in vals]
            n_true = sum(truthy)
            if name in ANY_BOOL_FIELDS:
                out[name] = n_true > 0
                out[f"{name}_share"] = n_true / len(truthy)
                out[f"{name}_rule"] = "any"
            else:
                out[name] = n_true * 2 > len(truthy)
                out[f"{name}_share"] = (n_true if out[name] else len(truthy) - n_true) / len(truthy)
                out[f"{name}_rule"] = "majority"
        elif spec["type"] == "enum":
            counts = Counter(str(v) for v in vals)
            top, n_top = counts.most_common(1)[0]
            out[name] = top
            share = n_top / len(vals)
            out[f"{name}_share"] = share
            shares.append(share)
            # keep the runner-up so genuinely split clips are visible downstream
            rest = counts.most_common(2)
            out[f"{name}_alt"] = rest[1][0] if len(rest) > 1 else ""
        else:  # free text — keep the caption from the most confident frame
            best = max(rows, key=lambda r: _conf(r) if _conf(r) is not None else -1)
            out[name] = best.get(name, "")

    confs = [_conf(r) for r in rows if _conf(r) is not None]
    out["model_confidence"] = sum(confs) / len(confs) if confs else None
    out["model_confidence_min"] = min(confs) if confs else None
    out["cue_instability"] = round(1 - (sum(shares) / len(shares)), 4) if shares else 0.0
    return out


def _conf(r: Dict[str, Any]) -> Optional[float]:
    c = r.get("model_confidence")
    return c if isinstance(c, (int, float)) else None


def run(frame_labels: Path, out_path: Path, schema,
        min_frames: int = 1) -> Path:
    """Aggregate work/frame_labels.jsonl -> work/clip_labels.jsonl."""
    frame_labels, out_path = Path(frame_labels), Path(out_path)
    by_clip: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for line in frame_labels.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        key = r.get("clip_stem")
        if key:
            by_clip[key].append(r)

    dropped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for key in sorted(by_clip):
            rows = by_clip[key]
            if len(rows) < min_frames:
                dropped += 1
                continue
            fh.write(json.dumps(aggregate_clip(rows, schema)) + "\n")

    print(f"[aggregate] {len(by_clip) - dropped} clips "
          f"({dropped} dropped for <{min_frames} frames) -> {out_path}")
    return out_path


def summarise(clip_labels: Path, schema) -> Dict[str, Any]:
    rows = [json.loads(l) for l in Path(clip_labels).open(encoding="utf-8") if l.strip()]
    if not rows:
        return {"n_clips": 0}
    enums = [f for f, s in _fields(schema).items() if s["type"] == "enum"]
    inst = sorted(r["cue_instability"] for r in rows)
    return {
        "n_clips": len(rows),
        "n_identities": len({r["identity_key"] for r in rows}),
        "raer_label_counts": dict(Counter(r.get("raer_label") or "(none)" for r in rows)),
        "median_cue_instability": inst[len(inst) // 2],
        "low_agreement_fields": sorted(
            ((f, round(sum(r.get(f"{f}_share", 0) for r in rows) / len(rows), 3))
             for f in enums), key=lambda kv: kv[1])[:5],
    }
