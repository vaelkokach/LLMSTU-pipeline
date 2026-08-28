"""Parse RAER filenames into identity / provenance fields.

Every RAER artifact is named after the clip it came from, e.g.

    RAER_images_SelfStudy_Laboratory_P040_012
    RAER_images_Class_Classroom_Class002_P001_021

so the stem carries: modality, session type, learning environment, class id
(classroom sessions only), a person index, and a clip index.

IMPORTANT — `person` is almost certainly a *session-local* index, not a global
subject id. The paper reports 136 students; if the distinct `P###` count comes
out far below that, the id resets per session and only `identity_key()` (which
folds in session/location/class) is safe to split on. `audit.py` checks this
explicitly and refuses to guess. Nothing downstream may split on `person` alone.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

SESSIONS = ("Class", "SelfStudy")
LOCATIONS = ("Classroom", "Library", "Laboratory", "Dormitory")

_STEM_RE = re.compile(
    r"^RAER_(?P<modality>[A-Za-z]+)_"
    r"(?P<session>Class|SelfStudy)_"
    r"(?P<location>Classroom|Library|Laboratory|Dormitory)_"
    r"(?:(?P<class_id>Class\d+)_)?"
    r"P(?P<person>\d+)_"
    r"(?P<clip>\d+)$"
)

# Tolerant fallback: pull the same tokens out of a stem that carries extra
# prefixes/suffixes (frame indices, other modalities, mirror-specific renames).
_LOOSE_RE = re.compile(
    r"(?P<session>Class|SelfStudy)_"
    r"(?P<location>Classroom|Library|Laboratory|Dormitory)_"
    r"(?:(?P<class_id>Class\d+)_)?"
    r"P(?P<person>\d+)_"
    r"(?P<clip>\d+)"
)

# a trailing frame index appended to a clip stem, e.g. ..._012_frame_0007 / ..._012_0007
_FRAME_SUFFIX_RE = re.compile(r"_(?:frame[_-]?)?\d+$", re.IGNORECASE)


def parse(stem: str, loose: bool = True) -> Optional[Dict[str, str]]:
    """Parse a clip stem into its fields, or None if it does not look like RAER."""
    m = _STEM_RE.match(stem)
    if m is None and loose:
        m = _LOOSE_RE.search(stem)
    if m is None:
        return None
    d = m.groupdict()
    out = {
        "session": d["session"],
        "location": d["location"],
        "class_id": d.get("class_id") or "",
        "person": d["person"],
        "clip": d["clip"],
    }
    out["modality"] = d.get("modality") or ""
    out["clip_stem"] = clip_stem(out)
    out["identity_key"] = identity_key(out)
    return out


def clip_stem(f: Dict[str, str]) -> str:
    """Canonical, modality-free clip id — the join key across mp4 / jpg / labels."""
    parts = [f["session"], f["location"]]
    if f.get("class_id"):
        parts.append(f["class_id"])
    parts += [f"P{f['person']}", f["clip"]]
    return "_".join(parts)


def identity_key(f: Dict[str, str]) -> str:
    """Best available subject identity.

    Folds session + location + class into the person index, because `P001` in
    `Class002` and `P001` in `Library` are not known to be the same student.
    This OVER-splits if the ids are in fact global (costing a little training
    data) rather than UNDER-splitting (which would leak a subject across the
    train/test boundary). Erring toward over-splitting is deliberate.
    """
    scope = [f["session"], f["location"]]
    if f.get("class_id"):
        scope.append(f["class_id"])
    return "/".join(scope) + f"/P{f['person']}"


def clip_key_from_path(path, root=None) -> Optional[str]:
    """Recover the clip stem from any artifact path (video, frame, feature).

    Tries, in order: the path's own stem, the stem with a trailing frame index
    stripped, then the parent directory name (frames are often foldered per clip).
    """
    p = Path(path)
    candidates = [p.stem, _FRAME_SUFFIX_RE.sub("", p.stem), p.parent.name]
    for c in candidates:
        f = parse(c)
        if f:
            return f["clip_stem"]
    return None
