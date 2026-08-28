"""Pre-flight audit of a RAER copy — run this BEFORE any labeling or training.

Answers, from the data itself rather than from the paper:

  1. inventory      what files are actually here, and how big
  2. metadata       what the non-media files (json/txt/csv) contain
  3. clips          how many clips, how many frames each, what is missing
  4. identity       is `P###` a global subject id or a per-session index?
  5. distribution   does this copy's class balance match the published one?
  6. split          is an official split shipped, and is it subject-disjoint?
  7. leakage        how badly would a naive clip-level split leak subjects?

Everything is discovery-first: the mirror ships no data card, so this module
reports what it finds and marks anything it could not verify as UNVERIFIED
rather than assuming a layout.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import naming

# --- published reference values (ICCV 2025, arXiv:2507.00586) ------------------
PAPER = {
    "n_clips": 2649,
    "n_subjects": 136,
    "classes": {  # percent of clips
        "neutrality": 65.23,
        "distraction": 19.89,
        "fatigue": 7.40,
        "enjoyment": 6.08,
        "confusion": 1.40,
    },
    "coarse_map": {  # fine -> coarse, per the paper's stated relationship
        "neutrality": "engaged",
        "enjoyment": "engaged",
        "confusion": "engaged",
        "fatigue": "engaged",
        "distraction": "distracted",
    },
    "split": "80/20 train/test, 'striving to exclude' overlapping individuals",
    "clip_seconds": "5-15",
    "fleiss_kappa": 0.832,
}

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
META_EXT = {".json", ".txt", ".csv", ".tsv", ".yaml", ".yml"}


# --------------------------------------------------------------------------- #
# 0. file listing — the audit runs on names+sizes, so most of it works from
#    Kaggle's file manifest without downloading 100 GB first.
# --------------------------------------------------------------------------- #
Listing = List[tuple]  # [(relative_path:str, size_bytes:int)]


def collect(root: Path) -> Listing:
    """Walk a real directory into the same listing shape the manifest gives us."""
    root = Path(root)
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        out.append((str(p.relative_to(root)).replace("\\", "/"), sz))
    return out


# --------------------------------------------------------------------------- #
# 1. inventory
# --------------------------------------------------------------------------- #
def inventory(listing: Listing, source: str = "") -> Dict[str, Any]:
    by_ext: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    dirs: Counter = Counter()
    for rel, sz in listing:
        ext = Path(rel).suffix.lower()
        by_ext[ext]["count"] += 1
        by_ext[ext]["bytes"] += sz
        dirs[str(Path(rel).parent).replace("\\", "/")] += 1
    return {
        "root": source,
        "total_files": len(listing),
        "total_bytes": sum(s for _, s in listing),
        "by_ext": {k: dict(v) for k, v in sorted(
            by_ext.items(), key=lambda kv: -kv[1]["bytes"])},
        "top_dirs": dirs.most_common(25),
    }


# --------------------------------------------------------------------------- #
# 2. metadata files
# --------------------------------------------------------------------------- #
def preview_metadata(root: Path, max_bytes: int = 4000,
                     max_files: int = 40) -> List[Dict[str, Any]]:
    """Head every small non-media file so we can see what ships with the data."""
    out = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file() or p.suffix.lower() not in META_EXT:
            continue
        if len(out) >= max_files:
            break
        try:
            size = p.stat().st_size
            head = p.read_text(encoding="utf-8", errors="replace")[:max_bytes]
        except OSError as e:
            size, head = -1, f"<unreadable: {e}>"
        entry = {"path": str(p.relative_to(root)), "bytes": size, "head": head}
        if p.suffix.lower() == ".json":
            entry["structure"] = _json_shape(p)
        else:
            entry["n_lines"] = _count_lines(p)
        out.append(entry)
    return out


def _count_lines(p: Path) -> int:
    try:
        with p.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return -1


def _json_shape(p: Path, depth: int = 3) -> Any:
    """Describe a JSON file's shape without loading the whole thing into a report."""
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return f"<unparseable: {e}>"

    def shape(x, d):
        if d <= 0:
            return type(x).__name__
        if isinstance(x, dict):
            keys = list(x)[:8]
            return {k: shape(x[k], d - 1) for k in keys} | (
                {"...": f"+{len(x) - 8} more keys"} if len(x) > 8 else {})
        if isinstance(x, list):
            return [f"list[{len(x)}]", shape(x[0], d - 1)] if x else "list[0]"
        return type(x).__name__

    return shape(data, depth)


# --------------------------------------------------------------------------- #
# 3. clip index
# --------------------------------------------------------------------------- #
def index_clips(listing: Listing) -> Dict[str, Dict[str, Any]]:
    """Group every media file under its clip stem."""
    clips: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"videos": [], "frames": [], "features": [], "fields": None})
    unparsed: List[str] = []
    for rel, _sz in listing:
        ext = Path(rel).suffix.lower()
        if ext not in VIDEO_EXT and ext not in IMAGE_EXT and ext != ".pt":
            continue
        key = naming.clip_key_from_path(rel)
        if key is None:
            unparsed.append(rel)
            continue
        bucket = "videos" if ext in VIDEO_EXT else ("frames" if ext in IMAGE_EXT else "features")
        clips[key][bucket].append(rel)
        if clips[key]["fields"] is None:
            clips[key]["fields"] = naming.parse_any(rel) or naming.parse(key)
    return {"clips": dict(clips), "unparsed": unparsed}


# --------------------------------------------------------------------------- #
# 4. identity
# --------------------------------------------------------------------------- #
def identity_report(clips: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Decide whether `P###` is a global subject id — never assume it."""
    persons, identities, scopes = set(), set(), defaultdict(set)
    for key, c in clips.items():
        f = c.get("fields") or naming.parse(key)
        if not f:
            continue
        persons.add(f["person"])
        identities.add(f["identity_key"])
        scopes[f["person"]].add(f["identity_key"])

    n_person, n_identity = len(persons), len(identities)
    # A person index that repeats across scopes is session-local, not global.
    multi_scope = {p: sorted(s) for p, s in scopes.items() if len(s) > 1}

    if n_person == 0:
        verdict, safe = "UNVERIFIED — no parseable person ids", None
    elif multi_scope:
        verdict = (f"SESSION-LOCAL — {len(multi_scope)} person ids appear under more than "
                   f"one session/location, so P### is an index within a recording, not a "
                   f"global subject id. Split on identity_key, never on person.")
        safe = "identity_key"
    elif abs(n_person - PAPER["n_subjects"]) <= 5:
        verdict = (f"LIKELY GLOBAL — {n_person} distinct person ids, close to the "
                   f"{PAPER['n_subjects']} subjects reported in the paper.")
        safe = "person"
    else:
        verdict = (f"AMBIGUOUS — {n_person} distinct person ids vs {PAPER['n_subjects']} "
                   f"subjects in the paper. Treat as session-local until confirmed by the "
                   f"authors; identity_key is the conservative choice.")
        safe = "identity_key"

    return {
        "n_distinct_person_ids": n_person,
        "n_distinct_identity_keys": n_identity,
        "paper_n_subjects": PAPER["n_subjects"],
        "person_ids_spanning_scopes": len(multi_scope),
        "examples_spanning_scopes": dict(list(multi_scope.items())[:5]),
        "verdict": verdict,
        "safe_split_key": safe,
    }


# --------------------------------------------------------------------------- #
# 5. class distribution
# --------------------------------------------------------------------------- #
def class_distribution(labels: Dict[str, str]) -> Dict[str, Any]:
    """Compare this copy's label balance against the published percentages."""
    if not labels:
        return {"status": "UNVERIFIED — no label mapping found", "rows": []}
    counts = Counter(str(v).strip().lower() for v in labels.values())
    n = sum(counts.values())
    rows, max_delta = [], 0.0
    for cls in list(PAPER["classes"]) + [c for c in counts if c not in PAPER["classes"]]:
        got = counts.get(cls, 0)
        pct = 100.0 * got / n if n else 0.0
        exp = PAPER["classes"].get(cls)
        delta = (pct - exp) if exp is not None else None
        if delta is not None:
            max_delta = max(max_delta, abs(delta))
        rows.append({"class": cls, "n": got, "pct": pct,
                     "paper_pct": exp, "delta_pp": delta})
    status = ("MATCHES the published distribution" if max_delta <= 2.0
              else f"DIFFERS from the published distribution by up to {max_delta:.1f}pp "
                   f"— this copy may be a subset or re-annotation")
    return {"status": status, "n_labeled": n, "max_delta_pp": max_delta, "rows": rows}


# --------------------------------------------------------------------------- #
# RAER's own annotation format
# --------------------------------------------------------------------------- #
# `RAER/annotation/{train,test}.txt` ship one clip per line:
#     RAER/images/Class/Classroom/Class009/P001/032 32 5
#     <clip directory>  <n_frames>  <label index 1-5>
# The index-to-class mapping is not documented anywhere in the copy, so we
# recover it by matching observed class proportions against the published ones
# (see infer_label_map) rather than assuming an order.
_ANNOT_RE = re.compile(r"^(?P<path>\S+)\s+(?P<n_frames>\d+)\s+(?P<label>\d+)\s*$")


def read_annotation(path: Path) -> List[Dict[str, Any]]:
    """Parse a RAER annotation file. Returns [] if it is not in that format."""
    rows = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        m = _ANNOT_RE.match(line.strip())
        if not m:
            continue
        key = naming.clip_key_from_path(m.group("path"))
        if key:
            rows.append({"clip_stem": key,
                         "n_frames": int(m.group("n_frames")),
                         "label_idx": int(m.group("label"))})
    return rows


def infer_label_map(counts: Dict[int, int]) -> Dict[str, Any]:
    """Recover label-index -> class name by matching published proportions.

    The five published percentages (65.23 / 19.89 / 7.40 / 6.08 / 1.40) are far
    apart, so rank-matching observed frequencies against them is unambiguous
    *provided the observed frequencies actually match*. The max per-class
    deviation is returned so a mapping built on a different subset is caught
    rather than silently trusted.
    """
    n = sum(counts.values())
    if not n:
        return {"map": {}, "max_delta_pp": None,
                "status": "UNVERIFIED — no labels to infer from"}
    obs = sorted(counts.items(), key=lambda kv: -kv[1])
    ref = sorted(PAPER["classes"].items(), key=lambda kv: -kv[1])
    if len(obs) != len(ref):
        return {"map": {}, "max_delta_pp": None,
                "status": f"UNVERIFIED — {len(obs)} label values, expected {len(ref)}"}
    mapping, deltas = {}, []
    for (idx, c), (cls, pct) in zip(obs, ref):
        mapping[idx] = cls
        deltas.append(abs(100.0 * c / n - pct))
    md = max(deltas)
    status = (f"CONFIRMED — every class within {md:.2f}pp of the published "
              f"distribution" if md <= 0.5 else
              f"AMBIGUOUS — deviates by up to {md:.2f}pp; this copy may be a "
              f"different subset, do not trust the mapping")
    return {"map": mapping, "max_delta_pp": md, "status": status}


def find_labels(root: Path) -> Dict[str, str]:
    """Best-effort recovery of clip -> fine-grained label from whatever ships.

    Handles the shapes actually plausible here: a JSON dict, a JSON list of
    records, or a whitespace/comma separated `path label` text file. Returns {}
    when nothing recognisable is present — which is itself an audit finding.
    """
    # 1) RAER's own `<clip> <n_frames> <idx>` annotation files, if present.
    annot: List[Dict[str, Any]] = []
    for p in Path(root).rglob("*.txt"):
        annot.extend(read_annotation(p))
    if annot:
        by_clip = {r["clip_stem"]: r["label_idx"] for r in annot}
        inferred = infer_label_map(Counter(by_clip.values()))
        if inferred["map"]:
            print(f"[audit] label map {inferred['map']} — {inferred['status']}")
            return {k: inferred["map"][v] for k, v in by_clip.items()}

    # 2) fall back to any file that names the classes in words.
    labels: Dict[str, str] = {}
    known = set(PAPER["classes"]) | set(PAPER["coarse_map"].values())
    for p in Path(root).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".json", ".txt", ".csv", ".tsv"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if p.suffix.lower() == ".json":
            try:
                data = json.loads(text)
            except Exception:
                continue
            for k, v in _walk_json_pairs(data):
                key = naming.clip_key_from_path(k)
                if key and str(v).strip().lower() in known:
                    labels[key] = str(v).strip().lower()
        else:
            for line in text.splitlines():
                parts = re.split(r"[,\t ]+", line.strip())
                if len(parts) < 2:
                    continue
                key = naming.clip_key_from_path(parts[0])
                cand = parts[-1].strip().lower()
                if key and cand in known:
                    labels[key] = cand
    return labels


def _walk_json_pairs(data, _depth=0):
    """Yield (path-like, label-like) pairs from arbitrarily shaped JSON."""
    if _depth > 6:
        return
    if isinstance(data, dict):
        # record form: {"image_path"/"video"/"clip": ..., "label"/"emotion": ...}
        pk = next((k for k in data if re.search(r"path|video|clip|file|name", k, re.I)), None)
        lk = next((k for k in data if re.search(r"label|emotion|class|category", k, re.I)), None)
        if pk and lk:
            yield data[pk], data[lk]
        for k, v in data.items():
            if isinstance(v, str):
                yield k, v
            else:
                yield from _walk_json_pairs(v, _depth + 1)
    elif isinstance(data, list):
        for item in data:
            yield from _walk_json_pairs(item, _depth + 1)


# --------------------------------------------------------------------------- #
# 6/7. split + leakage
# --------------------------------------------------------------------------- #
def find_splits(root: Path) -> Dict[str, List[str]]:
    """Look for shipped train/val/test lists and resolve them to clip stems."""
    splits: Dict[str, List[str]] = {}
    for p in Path(root).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".txt", ".csv", ".json"}:
            continue
        m = re.search(r"(train|val|valid|test)", p.stem, re.I)
        if not m:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # `train_subset.txt` etc. are auxiliary lists, not the split itself
        if re.search(r"subset|sampled|description", p.stem, re.I):
            continue
        keys = []
        for line in text.splitlines():
            tok = re.split(r"[,\t ]+", line.strip())[0]
            k = naming.clip_key_from_path(tok) if tok else None
            if k:
                keys.append(k)
        if keys:
            splits.setdefault(m.group(1).lower(), []).extend(keys)
    return {k: sorted(set(v)) for k, v in splits.items()}


def split_check(splits: Dict[str, List[str]]) -> Dict[str, Any]:
    if not splits:
        return {"status": "UNVERIFIED — no split files found in this copy",
                "note": "The paper's 80/20 split is not reproducible without them; "
                        "we must build our own subject-disjoint split."}
    names = sorted(splits)
    out: Dict[str, Any] = {"status": f"found: {', '.join(f'{k}={len(v)}' for k, v in splits.items())}",
                           "clip_overlaps": {}, "identity_overlaps": {}}
    out["identity_overlap_pct"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = set(splits[a]), set(splits[b])
            out["clip_overlaps"][f"{a}&{b}"] = len(sa & sb)
            ia = {naming.parse(k)["identity_key"] for k in sa if naming.parse(k)}
            ib = {naming.parse(k)["identity_key"] for k in sb if naming.parse(k)}
            shared, total = len(ia & ib), len(ia | ib)
            out["identity_overlaps"][f"{a}&{b}"] = shared
            out["identity_overlap_pct"][f"{a}&{b}"] = (
                round(100.0 * shared / total, 1) if total else 0.0)
    leaks = {k: v for k, v in out["identity_overlaps"].items() if v > 0}
    if not leaks:
        out["verdict"] = "SUBJECT-DISJOINT — safe to report numbers on this split"
    else:
        worst = max(out["identity_overlap_pct"].values())
        out["verdict"] = (
            f"LEAKY — the shipped split has zero clip overlap but shares subjects: "
            f"{', '.join(f'{k} {v}' for k, v in leaks.items())} "
            f"({worst}% of identities appear on both sides). A model can memorise a "
            f"face in training and be scored on it at test. Numbers from this split "
            f"are not comparable to a subject-disjoint benchmark — build our own.")
    return out


def leakage_sim(clips: Dict[str, Dict[str, Any]], test_frac: float = 0.2,
                seed: int = 13, trials: int = 20) -> Dict[str, Any]:
    """Quantify what a naive clip-level random split would leak."""
    keys = sorted(clips)
    ident = {k: (clips[k].get("fields") or naming.parse(k) or {}).get("identity_key")
             for k in keys}
    ident = {k: v for k, v in ident.items() if v}
    if not ident:
        return {"status": "UNVERIFIED — no parseable identities"}
    rng = random.Random(seed)
    straddle, contaminated = [], []
    ks = list(ident)
    for _ in range(trials):
        rng.shuffle(ks)
        cut = int(len(ks) * (1 - test_frac))
        tr = {ident[k] for k in ks[:cut]}
        te_keys = ks[cut:]
        te = {ident[k] for k in te_keys}
        straddle.append(len(tr & te))
        contaminated.append(sum(1 for k in te_keys if ident[k] in tr))
    n_ident = len(set(ident.values()))
    avg_c = sum(contaminated) / len(contaminated)
    n_test = len(ks) - int(len(ks) * (1 - test_frac))
    return {
        "n_clips": len(ks),
        "n_identities": n_ident,
        "clips_per_identity": round(len(ks) / n_ident, 1),
        "avg_identities_straddling": round(sum(straddle) / len(straddle), 1),
        "avg_test_clips_with_seen_subject": round(avg_c, 1),
        "pct_test_contaminated": round(100.0 * avg_c / n_test, 1) if n_test else 0.0,
        "verdict": ("A random clip-level split contaminates "
                    f"{round(100.0 * avg_c / n_test, 1) if n_test else 0}% of the test set "
                    f"with subjects seen in training. Split on identity, not on clips."),
    }


# --------------------------------------------------------------------------- #
# driver + report
# --------------------------------------------------------------------------- #
def run(root: Optional[Path] = None, out_dir: Path = Path("./work/audit"),
        listing: Optional[Listing] = None, source: str = "") -> Dict[str, Any]:
    """Audit a RAER copy.

    Pass `root` for a downloaded copy (full audit), or `listing` — [(path, bytes)]
    from Kaggle's file manifest — to audit names and sizes only. Listing mode
    answers the clip-count, identity and leakage questions for free; the label,
    split and metadata checks need the actual files and are reported as pending.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if listing is None:
        if root is None:
            raise ValueError("pass either root= (downloaded copy) or listing= (manifest)")
        root = Path(root)
        listing = collect(root)
        source = source or str(root)

    inv = inventory(listing, source)
    idx = index_clips(listing)
    clips = idx["clips"]

    if root is not None:
        meta = preview_metadata(root)
        labels = find_labels(root)
        splits = find_splits(root)
    else:
        meta_names = [rel for rel, _ in listing if Path(rel).suffix.lower() in META_EXT]
        meta = [{"path": p, "bytes": next(s for r, s in listing if r == p),
                 "head": "<listing mode — download this file to inspect it>"}
                for p in meta_names[:40]]
        labels, splits = {}, {}

    frames_per = [len(c["frames"]) for c in clips.values() if c["frames"]]
    frames_per.sort()

    result = {
        "mode": "full" if root is not None else "listing-only",
        "paper_reference": PAPER,
        "inventory": inv,
        "metadata_files": meta,
        "clips": {
            "n_clips": len(clips),
            "paper_n_clips": PAPER["n_clips"],
            "delta_vs_paper": len(clips) - PAPER["n_clips"],
            "with_video": sum(1 for c in clips.values() if c["videos"]),
            "with_frames": sum(1 for c in clips.values() if c["frames"]),
            "with_features": sum(1 for c in clips.values() if c["features"]),
            "frames_per_clip": {
                "min": frames_per[0] if frames_per else 0,
                "median": frames_per[len(frames_per) // 2] if frames_per else 0,
                "max": frames_per[-1] if frames_per else 0,
                "total": sum(frames_per),
            },
            "unparsed_files": len(idx["unparsed"]),
            "unparsed_examples": idx["unparsed"][:10],
        },
        "identity": identity_report(clips),
        "distribution": class_distribution(labels),
        "splits": split_check(splits),
        "leakage": leakage_sim(clips),
    }
    (out_dir / "audit.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    html = render_html(result, out_dir / "audit_report.html")
    print(f"[audit] json -> {out_dir/'audit.json'}")
    print(f"[audit] html -> {html}")
    return result


def _flag(text: str) -> str:
    t = text.upper()
    cls = ("bad" if any(w in t for w in ("LEAKY", "DIFFERS", "SESSION-LOCAL")) else
           "warn" if any(w in t for w in ("UNVERIFIED", "AMBIGUOUS")) else "ok")
    return f'<span class="{cls}">{escape(text)}</span>'


def render_html(r: Dict[str, Any], path: Path) -> Path:
    def table(rows, cols):
        head = "".join(f"<th>{escape(c)}</th>" for c in cols)
        body = "".join(
            "<tr>" + "".join(
                f"<td>{escape('' if x.get(c) is None else (f'{x[c]:.2f}' if isinstance(x[c], float) else str(x[c])))}</td>"
                for c in cols) + "</tr>"
            for x in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    ext_rows = [{"ext": k or "(none)", "count": v["count"],
                 "GB": round(v["bytes"] / 1e9, 2)} for k, v in r["inventory"]["by_ext"].items()]
    c = r["clips"]
    parts = [f"""<!doctype html><meta charset="utf-8"><title>RAER audit</title><style>
 body{{font:14px/1.55 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#111}}
 h1{{font-size:1.5rem}} h2{{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0}}
 th,td{{border:1px solid #ddd;padding:.35rem .5rem;text-align:left}} th{{background:#f6f6f6}}
 .ok{{color:#0a7d33;font-weight:600}} .warn{{color:#a86500;font-weight:600}} .bad{{color:#b3261e;font-weight:600}}
 pre{{background:#f6f6f6;padding:.6rem;overflow-x:auto;font-size:12px}}
 .kv td:first-child{{width:16rem;color:#555}}</style>
<h1>RAER pre-flight audit</h1>
<p>Source: <code>{escape(r['inventory']['root'] or '(manifest)')}</code> —
 {r['inventory']['total_files']:,} files,
 {r['inventory'].get('total_bytes', 0) / 1e9:.1f} GB.
 Mode: <b>{escape(r.get('mode', 'full'))}</b>.
 Reference values are from the ICCV 2025 paper (arXiv:2507.00586).</p>
{'<p class="warn">Listing-only mode: clip, identity and leakage checks are complete; '
 'label, split and metadata checks need the actual files and are shown as UNVERIFIED.</p>'
 if r.get('mode') == 'listing-only' else ''}

<h2>1. Inventory</h2>{table(ext_rows, ['ext', 'count', 'GB'])}

<h2>2. Clips</h2><table class="kv">
<tr><td>clips found</td><td>{c['n_clips']:,} (paper: {c['paper_n_clips']:,}, delta {c['delta_vs_paper']:+d})</td></tr>
<tr><td>with video / frames / features</td><td>{c['with_video']:,} / {c['with_frames']:,} / {c['with_features']:,}</td></tr>
<tr><td>frames per clip</td><td>min {c['frames_per_clip']['min']}, median {c['frames_per_clip']['median']}, max {c['frames_per_clip']['max']} ({c['frames_per_clip']['total']:,} total)</td></tr>
<tr><td>files with unparseable names</td><td>{c['unparsed_files']:,}</td></tr></table>

<h2>3. Subject identity</h2><p>{_flag(r['identity']['verdict'])}</p><table class="kv">
<tr><td>distinct P### ids</td><td>{r['identity']['n_distinct_person_ids']}</td></tr>
<tr><td>distinct identity keys</td><td>{r['identity']['n_distinct_identity_keys']}</td></tr>
<tr><td>subjects reported in paper</td><td>{r['identity']['paper_n_subjects']}</td></tr>
<tr><td>P ids spanning >1 session</td><td>{r['identity']['person_ids_spanning_scopes']}</td></tr>
<tr><td>safe split key</td><td><code>{escape(str(r['identity']['safe_split_key']))}</code></td></tr></table>

<h2>4. Class distribution</h2><p>{_flag(r['distribution']['status'])}</p>
{table(r['distribution']['rows'], ['class', 'n', 'pct', 'paper_pct', 'delta_pp'])}

<h2>5. Official split</h2><p>{_flag(str(r['splits'].get('status')))}</p>
<pre>{escape(json.dumps(r['splits'], indent=1))}</pre>

<h2>6. Leakage under a naive clip-level split</h2>
<p>{_flag(str(r['leakage'].get('verdict', r['leakage'].get('status'))))}</p>
<pre>{escape(json.dumps(r['leakage'], indent=1))}</pre>

<h2>7. Metadata files shipped with this copy</h2>"""]
    for m in r["metadata_files"]:
        parts.append(f"<h3><code>{escape(m['path'])}</code> — {m['bytes']:,} bytes</h3>")
        if "structure" in m:
            parts.append(f"<pre>{escape(json.dumps(m['structure'], indent=1)[:2000])}</pre>")
        parts.append(f"<pre>{escape(m['head'][:1500])}</pre>")
    Path(path).write_text("".join(parts), encoding="utf-8")
    return Path(path)
