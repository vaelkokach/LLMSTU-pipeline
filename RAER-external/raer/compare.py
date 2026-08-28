"""Stage 05 — RAER's label vs our Qwen cue labels vs human adjudication.

Three questions, kept separate:

**A. Do the taxonomies agree at the coarse level?** RAER's fine labels collapse to
engaged/distracted by its own stated rule (distraction -> distracted, the rest ->
engaged). Our `engagement_level` collapses the same way. That is the only place
the two taxonomies are directly commensurable, so it is the only place we compare
them head to head. Reported as a confusion matrix plus Cohen's kappa — *not*
accuracy, because 80% of clips are "engaged" and accuracy would flatter us.

**B. Which observable cues actually carry each academic emotion?** For every RAER
class, the distribution of our cue values. This is the part that does not depend
on the taxonomies lining up: it says, empirically, that RAER "distraction" clips
show `phone_visible` at rate X and `gaze_direction=away` at rate Y. That is a
statement about visible behaviour, which is what our taxonomy is for.

**C. When they disagree, who is right?** On the human-audited subset only, the
reviewer's verdict adjudicates. Reported separately for the representative and
hard groups, because the hard group is deliberately low-confidence and mixing
them would understate real-world accuracy.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

# RAER's own stated relationship between its two label sets.
FINE_TO_COARSE = {
    "enjoyment": "engaged", "neutrality": "engaged",
    "confusion": "engaged", "fatigue": "engaged",
    "distraction": "distracted",
}

# Our engagement_level -> the same two-way split. partially_engaged sits with
# engaged because RAER's "engaged" also covers confusion and fatigue, which are
# effortful rather than absent states.
OURS_TO_COARSE = {
    "engaged": "engaged", "partially_engaged": "engaged",
    "disengaged": "distracted",
}


def _coarse_raer(lbl: Optional[str]) -> Optional[str]:
    return FINE_TO_COARSE.get((lbl or "").strip().lower())


def _coarse_ours(lbl: Optional[str]) -> Optional[str]:
    return OURS_TO_COARSE.get((lbl or "").strip().lower())


def cohens_kappa(pairs: List[tuple]) -> Optional[float]:
    """Chance-corrected agreement on a list of (a, b) label pairs."""
    n = len(pairs)
    if not n:
        return None
    labels = sorted({x for p in pairs for x in p})
    obs = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    exp = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return None if exp >= 1.0 else round((obs - exp) / (1 - exp), 4)


# --------------------------------------------------------------------------- #
# A. coarse agreement
# --------------------------------------------------------------------------- #
def coarse_agreement(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs, matrix = [], defaultdict(Counter)
    for r in clips:
        a, b = _coarse_raer(r.get("raer_label")), _coarse_ours(r.get("engagement_level"))
        if a and b:
            pairs.append((a, b))
            matrix[a][b] += 1
    n = len(pairs)
    if not n:
        return {"status": "UNVERIFIED — no comparable clips", "n": 0}
    agree = sum(1 for a, b in pairs if a == b)
    # per-class recall matters more than overall accuracy on an 80/20 prior
    recall = {}
    for a in matrix:
        tot = sum(matrix[a].values())
        recall[a] = round(matrix[a][a] / tot, 4) if tot else None
    return {
        "n": n,
        "raw_agreement": round(agree / n, 4),
        "cohens_kappa": cohens_kappa(pairs),
        "per_class_recall": recall,
        "matrix": {a: dict(c) for a, c in matrix.items()},
        "note": "rows = RAER coarse, cols = ours. Raw agreement is inflated by the "
                "65% neutrality prior; read kappa and per-class recall instead.",
    }


# --------------------------------------------------------------------------- #
# B. cue profile per RAER class
# --------------------------------------------------------------------------- #
def cue_profiles(clips: List[Dict[str, Any]], schema,
                 min_support: int = 20) -> Dict[str, Any]:
    """For each RAER class, the distribution of each cue field."""
    by_class: Dict[str, List[Dict]] = defaultdict(list)
    for r in clips:
        if r.get("raer_label"):
            by_class[r["raer_label"]].append(r)

    fields = [f for f, s in schema.FIELDS.items() if s["type"] in ("enum", "bool")]
    overall = {f: Counter(str(r.get(f)) for r in clips) for f in fields}
    n_all = len(clips)

    out: Dict[str, Any] = {"classes": {}, "low_support": []}
    for cls, rows in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        if len(rows) < min_support:
            out["low_support"].append({"class": cls, "n": len(rows)})
        prof = {}
        for f in fields:
            c = Counter(str(r.get(f)) for r in rows)
            n = sum(c.values()) or 1
            # keep values that are over-represented vs the corpus baseline
            vals = []
            for v, k in c.most_common(4):
                base = overall[f][v] / n_all if n_all else 0
                vals.append({"value": v, "pct": round(100 * k / n, 1),
                             "baseline_pct": round(100 * base, 1),
                             "lift": round((k / n) / base, 2) if base else None})
            prof[f] = vals
        out["classes"][cls] = {"n": len(rows), "fields": prof}
    return out


# --------------------------------------------------------------------------- #
# C. human adjudication
# --------------------------------------------------------------------------- #
def adjudication(golden: Dict[str, Any], schema) -> Dict[str, Any]:
    items = [it for it in golden.get("items", [])
             if it.get("status") in ("verified", "edited", "reviewed")]
    if not items:
        return {"status": "UNVERIFIED — nothing reviewed yet", "n": 0}

    fields = [f for f, s in schema.FIELDS.items() if s["type"] in ("enum", "bool")]
    res: Dict[str, Any] = {"n_reviewed": len(items)}

    # C1 — was RAER's own label judged to fit?
    verd = Counter(it.get("raer_verdict") or "(none)" for it in items)
    n_v = sum(v for k, v in verd.items() if k in ("agree", "disagree", "unclear"))
    res["raer_label_verdict"] = {
        "counts": dict(verd),
        "pct_agree": round(100 * verd["agree"] / n_v, 1) if n_v else None,
        "pct_disagree": round(100 * verd["disagree"] / n_v, 1) if n_v else None,
    }
    # where reviewers disagreed, what did they prefer?
    swaps = Counter((it["raer_label"], it.get("raer_correct") or "?")
                    for it in items if it.get("raer_verdict") == "disagree")
    res["raer_label_swaps"] = [{"from": a, "to": b, "n": n}
                               for (a, b), n in swaps.most_common(12)]

    # C2 — per-field accuracy of our cue labels against the reviewer
    per_field, groups = [], {}
    for f in fields:
        n = correct = 0
        for it in items:
            g, p = it.get("golden", {}), it.get("pseudo", {})
            if f not in g:
                continue
            n += 1
            correct += str(g[f]) == str(p.get(f))
        per_field.append({"field": f, "n": n,
                          "accuracy": round(correct / n, 4) if n else None})
    res["cue_accuracy"] = sorted(
        [x for x in per_field if x["accuracy"] is not None], key=lambda x: x["accuracy"])

    for gname in ("representative", "hard"):
        rows = [it for it in items if it.get("group") == gname]
        if not rows:
            continue
        accs = []
        for f in fields:
            pairs = [(it["pseudo"].get(f), it["golden"].get(f))
                     for it in rows if f in it.get("golden", {})]
            if pairs:
                accs.append(sum(str(a) == str(b) for a, b in pairs) / len(pairs))
        groups[gname] = {"n": len(rows),
                         "macro_accuracy": round(sum(accs) / len(accs), 4) if accs else None}
    res["by_group"] = groups
    return res


# --------------------------------------------------------------------------- #
# driver + report
# --------------------------------------------------------------------------- #
def run(clip_labels: Path, out_dir: Path, schema,
        golden_path: Optional[Path] = None) -> Dict[str, Any]:
    clips = [json.loads(l) for l in Path(clip_labels).open(encoding="utf-8") if l.strip()]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    golden = {}
    if golden_path and Path(golden_path).exists():
        golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))

    result = {
        "n_clips": len(clips),
        "coarse": coarse_agreement(clips),
        "cues": cue_profiles(clips, schema),
        "human": adjudication(golden, schema) if golden else
                 {"status": "UNVERIFIED — no golden.json yet", "n": 0},
    }
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    html = render_html(result, out_dir / "comparison_report.html")
    print(f"[compare] json -> {out_dir/'comparison.json'}")
    print(f"[compare] html -> {html}")
    return result


def render_html(r: Dict[str, Any], path: Path) -> Path:
    co, hu = r["coarse"], r["human"]
    p = [f"""<!doctype html><meta charset="utf-8"><title>RAER vs cues vs human</title><style>
 body{{font:14px/1.55 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#111}}
 h1{{font-size:1.5rem}} h2{{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 h3{{font-size:.95rem;margin:1.2rem 0 .3rem}}
 table{{border-collapse:collapse;margin:.5rem 0}} th,td{{border:1px solid #ddd;padding:.3rem .55rem;text-align:left}}
 th{{background:#f6f6f6}} .num{{text-align:right}} .mut{{color:#666;font-size:12px}}
 .lift{{color:#0a7d33;font-weight:600}} .lo{{color:#b3261e}}</style>
<h1>RAER label vs our cue labels vs human review</h1>
<p class="mut">{r['n_clips']:,} clips.</p>

<h2>A · Coarse agreement (engaged / distracted)</h2>"""]

    if co.get("n"):
        p.append(f"""<p>n = {co['n']:,} · raw agreement <b>{co['raw_agreement']*100:.1f}%</b>
         · Cohen's κ <b>{co['cohens_kappa']}</b></p><p class="mut">{escape(co['note'])}</p>
         <table><tr><th>RAER ↓ / ours →</th><th>engaged</th><th>distracted</th><th>recall</th></tr>""")
        for a in ("engaged", "distracted"):
            row = co["matrix"].get(a, {})
            p.append(f"<tr><th>{a}</th><td class='num'>{row.get('engaged',0)}</td>"
                     f"<td class='num'>{row.get('distracted',0)}</td>"
                     f"<td class='num'>{co['per_class_recall'].get(a)}</td></tr>")
        p.append("</table>")
    else:
        p.append(f"<p>{escape(str(co.get('status')))}</p>")

    p.append("<h2>B · Which visible cues carry each RAER class</h2>"
             "<p class='mut'>lift = how much more often this cue value appears in this "
             "class than across the corpus. Lift &gt; 1.5 is where the taxonomies connect.</p>")
    for cls, d in r["cues"]["classes"].items():
        p.append(f"<h3>{escape(cls)} <span class='mut'>n={d['n']}</span></h3><table>"
                 "<tr><th>field</th><th>top value</th><th>in class</th><th>baseline</th><th>lift</th></tr>")
        for f, vals in d["fields"].items():
            if not vals:
                continue
            v = max(vals, key=lambda x: x["lift"] or 0)
            cls_attr = "lift" if (v["lift"] or 0) >= 1.5 else ""
            p.append(f"<tr><td>{escape(f)}</td><td>{escape(v['value'])}</td>"
                     f"<td class='num'>{v['pct']}%</td><td class='num'>{v['baseline_pct']}%</td>"
                     f"<td class='num {cls_attr}'>{v['lift']}</td></tr>")
        p.append("</table>")
    if r["cues"]["low_support"]:
        p.append("<p class='lo'>Low support (treat as anecdotal): "
                 + escape(str(r["cues"]["low_support"])) + "</p>")

    p.append("<h2>C · Human adjudication</h2>")
    if hu.get("n_reviewed"):
        v = hu["raer_label_verdict"]
        p.append(f"<p>{hu['n_reviewed']} clips reviewed. RAER's label judged to fit "
                 f"<b>{v['pct_agree']}%</b> of the time (disagree {v['pct_disagree']}%).</p>")
        if hu["raer_label_swaps"]:
            p.append("<table><tr><th>RAER said</th><th>reviewer preferred</th><th>n</th></tr>")
            for s in hu["raer_label_swaps"]:
                p.append(f"<tr><td>{escape(s['from'])}</td><td>{escape(s['to'])}</td>"
                         f"<td class='num'>{s['n']}</td></tr>")
            p.append("</table>")
        p.append("<h3>Our cue accuracy vs the reviewer</h3><table>"
                 "<tr><th>field</th><th>accuracy</th><th>n</th></tr>")
        for x in hu["cue_accuracy"]:
            p.append(f"<tr><td>{escape(x['field'])}</td>"
                     f"<td class='num'>{x['accuracy']*100:.1f}%</td><td class='num'>{x['n']}</td></tr>")
        p.append("</table>")
        if hu.get("by_group"):
            p.append("<p class='mut'>by sample group: " + escape(str(hu["by_group"])) + "</p>")
    else:
        p.append(f"<p>{escape(str(hu.get('status')))}</p>")

    Path(path).write_text("".join(p), encoding="utf-8")
    return Path(path)
