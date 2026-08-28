"""Stage 01 — clip -> N sampled frames, written as an LLMSTU-format manifest.

The output manifest is consumed by `llmstu.caption.run()` **unchanged**, which
constrains the layout in one non-obvious way: that function resolves a row's
image as `crops_root / Path(row["crop_path"]).name`, i.e. by *basename*. The
mirror's frames live in per-clip folders and are named `frame_0004.jpg`, so
their basenames collide across clips. We therefore materialise the sampled
frames into one flat directory as `<clip_stem>_f<NN>.jpg`, which is unique.

Frames come from the pre-extracted jpgs when the copy ships them, and are decoded
from the mp4 otherwise — same manifest either way, with `source` recording which.

Per `config.yaml`, the frame is labeled **whole**: no crop. RAER's distraction
class turns on context (a phone in the lap, a book on the desk) that an
upper-body crop would remove.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import naming


def pick_indices(n_available: int, n_want: int) -> List[int]:
    """Evenly spaced indices across the clip, endpoints included.

    Clips are the middle 5-15 s of a segment, so the emotion is present
    throughout — uniform coverage beats sampling the middle.
    """
    if n_available <= 0:
        return []
    if n_available <= n_want:
        return list(range(n_available))
    step = (n_available - 1) / (n_want - 1) if n_want > 1 else 0
    return [round(i * step) for i in range(n_want)]


def _copy_frames(srcs: List[Path], clip_stem: str, out_dir: Path) -> List[Path]:
    out = []
    for i, src in enumerate(srcs):
        dst = out_dir / f"{clip_stem}_f{i:02d}.jpg"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        out.append(dst)
    return out


def _decode_frames(video: Path, idxs_of: int, clip_stem: str,
                   out_dir: Path, n_want: int) -> List[Path]:
    """Extract n_want evenly spaced frames from a video with OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:  # some encoders report 0; fall back to sequential read
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        total = len(frames)
        picks = pick_indices(total, n_want)
        out = []
        for i, p in enumerate(picks):
            dst = out_dir / f"{clip_stem}_f{i:02d}.jpg"
            if not dst.exists():
                cv2.imwrite(str(dst), frames[p])
            out.append(dst)
        cap.release()
        return out

    out = []
    for i, p in enumerate(pick_indices(total, n_want)):
        dst = out_dir / f"{clip_stem}_f{i:02d}.jpg"
        if not dst.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, p)
            ok, img = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(dst), img)
        out.append(dst)
    cap.release()
    return out


def build(clips: Dict[str, Dict[str, Any]], media_root: Path, out_dir: Path,
          manifest_path: Path, n_per_clip: int = 8,
          use_existing_frames: bool = True,
          labels: Optional[Dict[str, str]] = None,
          resume: bool = True, limit: Optional[int] = None) -> Path:
    """Materialise sampled frames + write the caption manifest.

    `clips` is `raer.audit.index_clips(...)["clips"]`. `labels` maps clip_stem ->
    RAER fine label; it is carried through so stage 04 can stratify over it and
    stage 05 can compare against it. Missing labels are recorded as "" rather
    than dropped — an unlabeled clip is a finding, not a row to silently lose.
    """
    media_root = Path(media_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if resume and manifest_path.exists():
        for line in manifest_path.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["clip_stem"])
            except Exception:
                pass

    todo = [k for k in sorted(clips) if k not in done]
    if limit:
        todo = todo[:limit]
    print(f"[frames] {len(todo)} clips to sample "
          f"({len(done)} already done), {n_per_clip} frames each")

    stats = {"from_frames": 0, "from_video": 0, "skipped": 0}
    with manifest_path.open("a", encoding="utf-8") as fh:
        for n, key in enumerate(todo, 1):
            c = clips[key]
            f = c.get("fields") or naming.parse(key)
            if not f:
                stats["skipped"] += 1
                continue

            written: List[Path] = []
            source = ""
            if use_existing_frames and c.get("frames"):
                # sort NUMERICALLY — frames are 1.jpg, 2.jpg, ..., 10.jpg, so a
                # lexicographic sort would put 10 before 2 and scramble time.
                srcs = [media_root / r for r in
                        sorted(c["frames"], key=naming.frame_number)]
                srcs = [s for s in srcs if s.exists()]
                if srcs:
                    picks = [srcs[i] for i in pick_indices(len(srcs), n_per_clip)]
                    written = _copy_frames(picks, key, out_dir)
                    source = "frames"
                    stats["from_frames"] += 1
            if not written and c.get("videos"):
                vid = media_root / sorted(c["videos"])[0]
                if vid.exists():
                    written = _decode_frames(vid, 0, key, out_dir, n_per_clip)
                    source = "video"
                    stats["from_video"] += 1
            if not written:
                stats["skipped"] += 1
                continue

            for i, p in enumerate(written):
                fh.write(json.dumps({
                    "crop_path": p.name,          # basename — llmstu.caption resolves on this
                    "clip_stem": key,
                    "frame_idx": i,
                    "n_frames": len(written),
                    "source": source,
                    "identity_key": f["identity_key"],
                    "session": f["session"],
                    "location": f["location"],
                    "class_id": f["class_id"],
                    "person": f["person"],
                    "raer_label": (labels or {}).get(key, ""),
                }) + "\n")
            if n % 200 == 0:
                fh.flush()
                print(f"[frames] {n}/{len(todo)}", flush=True)

    print(f"[frames] {stats} -> {manifest_path}")
    return manifest_path


def summarise(manifest_path: Path) -> Dict[str, Any]:
    rows = [json.loads(l) for l in Path(manifest_path).open(encoding="utf-8") if l.strip()]
    clips = {r["clip_stem"] for r in rows}
    unlabeled = {r["clip_stem"] for r in rows if not r["raer_label"]}
    return {
        "n_frames": len(rows),
        "n_clips": len(clips),
        "frames_per_clip": round(len(rows) / len(clips), 2) if clips else 0,
        "n_identities": len({r["identity_key"] for r in rows}),
        "n_clips_without_label": len(unlabeled),
        "by_source": {s: sum(1 for r in rows if r["source"] == s)
                      for s in {r["source"] for r in rows}},
    }
