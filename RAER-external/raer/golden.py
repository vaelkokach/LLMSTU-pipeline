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


def load_reviewed(path: Optional[Path]) -> Dict[str, str]:
    """clip_stem -> status, for clips already reviewed in an earlier golden.json.

    Asking to carry work forward and silently getting none is the worst outcome
    here: the run looks fine and you re-review clips you already did. So a path
    that was explicitly given but is missing or unreadable is an error, not a
    shrug. Only `path is None` (the flag was not passed) means "no seeding".
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"--seed-golden {path} does not exist. Upload the earlier golden.json "
            f"there first, or drop the flag if you meant to start fresh.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"--seed-golden {path} is not readable JSON: {e}")

    reviewed = {it["clip_stem"]: it.get("status", "")
                for it in data.get("items", [])
                if it.get("clip_stem") and it.get("status") in ("verified", "edited", "reviewed")}
    if not reviewed:
        raise SystemExit(
            f"--seed-golden {path} has {len(data.get('items', []))} items but none "
            f"marked verified/edited/reviewed. Did you export before reviewing?")
    return reviewed


DEGENERATE_AT = 0.95   # a field this constant tells the reviewer nothing


def find_degenerate(rows: List[Dict[str, Any]], schema):
    """Fields whose modal value covers >= DEGENERATE_AT of the corpus."""
    from collections import Counter
    out = []
    for f, spec in schema.FIELDS.items():
        if spec["type"] not in ("enum", "bool"):
            continue
        c = Counter(str(r.get(f)) for r in rows)
        n = sum(c.values())
        if not n:
            continue
        val, k = c.most_common(1)[0]
        if k / n >= DEGENERATE_AT:
            out.append((f, val, 100.0 * k / n))
    return out


def hardest(rows: List[Dict[str, Any]], n: int) -> List[int]:
    """Indices of the clips the model was least self-consistent on.

    NOT model_confidence: Qwen3.5 self-rates 0.95 on essentially everything here
    (min 0.80, median 0.95, max 0.95), so confidence-based triage selects a
    random subset. `cue_instability` — how much the cue values disagreed across
    the clip's own frames — actually varies (0 to 0.58) and is a real signal of
    a clip the model found difficult.
    """
    scored = [(r.get("cue_instability") or 0.0, i) for i, r in enumerate(rows)]
    scored.sort(reverse=True)
    return [i for _, i in scored[:n]]


def build(clip_labels: Path, frames_dir: Path, out_dir: Path, schema, sampling,
          n_clips: int = 200, min_per_class: int = 25, hard_frac: float = 0.2,
          strip: int = 4, seed: int = 13,
          stratify_by: str = "raer_label",
          seed_golden: Optional[Path] = None,
          drop_degenerate: bool = True,
          blind: bool = True,
          require_label: bool = True) -> Path:
    """Write labeling/data.json for the three-way review UI.

    `seed_golden` carries work forward from an earlier review (typically the pilot).
    Those clips are force-included rather than left to chance: sampling 200 from
    2,649 would re-draw only ~4 of a 60-clip pilot, throwing away the rest of the
    human effort. Their pseudo-labels are unchanged between runs — temperature is
    0.0 and frame selection is deterministic — so loading that golden.json in the
    UI pre-fills them correctly.
    """
    rows = [json.loads(l) for l in Path(clip_labels).open(encoding="utf-8") if l.strip()]
    if not rows:
        raise SystemExit(f"no clips in {clip_labels}")

    # The mirror ships 2,669 clips but annotates only 2,649. Reviewing one of the
    # 20 extras asks "does RAER's label fit?" about a clip that has no label —
    # an unanswerable question that still costs a review slot, and whose answer
    # then pollutes the agreement rate. Drop them.
    if require_label:
        labelled = [r for r in rows if r.get(stratify_by)]
        if len(labelled) != len(rows):
            print(f"[golden] excluding {len(rows) - len(labelled)} clips with no "
                  f"{stratify_by} (unannotated extras)")
        rows = labelled
        if not rows:
            raise SystemExit(f"no clips have a {stratify_by}")

    # Fields that carry no information in this corpus are dropped from review.
    # Asking a reviewer to confirm a value that is 99% constant costs attention
    # and buys a meaningless 100% accuracy score.
    degenerate = find_degenerate(rows, schema) if drop_degenerate else []
    if drop_degenerate:
        if degenerate:
            print(f"[golden] dropping near-constant fields from review: "
                  + ", ".join(f"{f} ({p:.0f}% {v})" for f, v, p in degenerate))

    reviewed = load_reviewed(seed_golden)
    seeded_idx = [i for i, r in enumerate(rows) if r["clip_stem"] in reviewed]
    if seeded_idx:
        print(f"[golden] carrying forward {len(seeded_idx)} already-reviewed clips "
              f"from {seed_golden}")

    taken = set(seeded_idx)
    rest = [r for i, r in enumerate(rows) if i not in taken]
    remap = {id(r): i for i, r in enumerate(rows)}
    n_new = max(0, n_clips - len(seeded_idx))
    picks = [(i, "seeded") for i in seeded_idx]

    if n_new:
        # Hard set comes from cue_instability, not sampling.select's
        # model_confidence path — see hardest(). Representative set is then drawn
        # from what remains, so the two groups stay disjoint.
        n_hard = int(hard_frac * n_new)
        hard_local = hardest(rest, n_hard)
        hard_global = {remap[id(rest[j])] for j in hard_local}
        picks += [(i, "hard") for i in hard_global]

        pool = [r for j, r in enumerate(rest) if j not in set(hard_local)]
        n_repr = n_new - len(hard_global)
        # sampling.select applies min_per_class as a FLOOR per class per field, so
        # 5 classes x 25 can return 112 when you asked for 60. Cap the floor so
        # n_clips means n_clips; warn rather than silently shrinking coverage.
        floor = min(min_per_class, max(1, n_repr // max(1, len(
            {str(r.get(stratify_by)) for r in pool}))))
        if floor < min_per_class:
            print(f"[golden] min_per_class {min_per_class} would exceed n={n_clips}; "
                  f"using {floor}/class. Raise --n for fuller rare-class coverage.")
        sub = sampling.select(pool, n_repr, [stratify_by], floor, 0.0, seed=seed)
        picks += [(remap[id(pool[j])], "representative") for j, _ in sub][:n_repr]

    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dropped = {f for f, _, _ in (degenerate if drop_degenerate else [])}
    cue_fields = {k: v for k, v in schema.FIELDS.items()
                  if v["type"] in ("enum", "bool") and k not in dropped}
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
        # blind: the UI hides RAER's label until the reviewer has committed to the
        # cues. Showing it first anchors them, which is how a 200-clip review came
        # back with six fields at exactly 100%.
        "blind": blind,
        "dropped_fields": [{"field": f, "value": v, "pct": round(p, 1)}
                           for f, v, p in degenerate],
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
