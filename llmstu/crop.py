"""Student detection + wide face/upper-body cropping.

Best-practice choices baked in:
  * Ultralytics YOLO (yolo11) person detector, run at high imgsz because the
    source frames are wide-angle rooms with many small students.
  * Each detected person becomes ONE crop. For classroom footage we crop the
    top `head_frac` of the person box (head + shoulders) and pad by
    `margin_frac` so the face keeps surrounding context ("wide angle face").
  * Tiny detections (min_side_px) are dropped so we don't caption a 20px blob.
  * Everything is streamed to JSONL so a 100k-frame run is resumable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

from .config import CropConfig


def _load_detector(cfg: CropConfig):
    from ultralytics import YOLO
    return YOLO(cfg.detector_model)


def _wide_face_box(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int, cfg: CropConfig
) -> Tuple[int, int, int, int]:
    """Turn a full-person box into a padded head/upper-body crop box."""
    bw, bh = x2 - x1, y2 - y1
    if cfg.crop_mode == "person":
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2
    elif cfg.crop_mode == "upper_body":
        cx1, cy1, cx2, cy2 = x1, y1, x2, y1 + 0.65 * bh
    else:  # head_wide (default)
        cx1, cy1, cx2, cy2 = x1, y1, x2, y1 + cfg.head_frac * bh
    # pad with context; heads need a little extra above for hair/hands-on-face
    px, py = cfg.margin_frac * (cx2 - cx1), cfg.margin_frac * (cy2 - cy1)
    ex1 = max(0, int(cx1 - px))
    ey1 = max(0, int(cy1 - py * 1.2))
    ex2 = min(W, int(cx2 + px))
    ey2 = min(H, int(cy2 + py))
    return ex1, ey1, ex2, ey2


def crop_frame(
    img: Image.Image, detector, cfg: CropConfig
) -> List[Tuple[Image.Image, Dict]]:
    """Return list of (crop_image, meta) for one frame."""
    W, H = img.size
    res = detector.predict(
        img, classes=[cfg.person_class_id], conf=cfg.conf, iou=cfg.iou,
        imgsz=cfg.imgsz, verbose=False,
    )[0]
    out: List[Tuple[Image.Image, Dict]] = []
    boxes = res.boxes
    if boxes is None:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    # keep the most confident detections if a frame is very crowded
    order = confs.argsort()[::-1][: cfg.max_per_frame]
    for i in order:
        x1, y1, x2, y2 = xyxy[i]
        bx1, by1, bx2, by2 = _wide_face_box(x1, y1, x2, y2, W, H, cfg)
        if min(bx2 - bx1, by2 - by1) < cfg.min_side_px:
            continue
        crop = img.crop((bx1, by1, bx2, by2))
        if cfg.out_size and cfg.out_size > 0:
            crop = _square_pad_resize(crop, cfg.out_size)
        meta = {
            "person_idx": int(i),
            "bbox_person": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_crop": [bx1, by1, bx2, by2],
            "det_conf": float(confs[i]),
        }
        out.append((crop, meta))
    return out


def _square_pad_resize(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def run(
    frames: Iterable[Path],
    out_dir: Path,
    cfg: CropConfig,
    manifest_path: Optional[Path] = None,
    resume: bool = True,
) -> Path:
    """Crop every frame; write crops to out_dir and a JSONL manifest.

    Manifest rows: {crop_path, src_frame, person_idx, bbox_person, bbox_crop, det_conf}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or (out_dir.parent / "crops_manifest.jsonl")

    done_frames = set()
    if resume and manifest_path.exists():
        for line in manifest_path.open():
            try:
                done_frames.add(json.loads(line)["src_frame"])
            except Exception:
                pass

    detector = _load_detector(cfg)
    n_crops = 0
    with manifest_path.open("a") as mf:
        for frame in frames:
            key = str(frame.name)
            if key in done_frames:
                continue
            try:
                img = Image.open(frame).convert("RGB")
            except Exception:
                continue
            stem = frame.stem
            for crop, meta in crop_frame(img, detector, cfg):
                crop_name = f"{stem}__p{meta['person_idx']:02d}.jpg"
                crop_path = out_dir / crop_name
                crop.save(crop_path, quality=92)
                row = {"crop_path": str(crop_path.relative_to(out_dir.parent)),
                       "src_frame": key, **meta}
                mf.write(json.dumps(row) + "\n")
                n_crops += 1
            mf.flush()
    print(f"[crop] wrote {n_crops} crops -> {out_dir}")
    return manifest_path
