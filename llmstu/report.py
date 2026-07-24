"""Tiny HTML report builders for the tuning step (no external deps)."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Dict, List

_CSS = """
body{font:13px system-ui,Segoe UI,Roboto,sans-serif;background:#0f1115;color:#e6e8ec;margin:0;padding:20px}
h1{font-size:18px} h2{font-size:15px;margin-top:26px;border-top:1px solid #2a2f3a;padding-top:14px}
.meta{color:#9aa3b2;font-size:12px;margin:2px 0 10px}
.grid{display:flex;flex-wrap:wrap;gap:8px}
.grid img{height:150px;border:1px solid #2a2f3a;border-radius:6px;background:#000}
table{border-collapse:collapse;width:100%;margin-top:8px}
td,th{border:1px solid #2a2f3a;padding:6px 8px;vertical-align:top;font-size:12px}
th{background:#171a21;position:sticky;top:0}
td.k{color:#9aa3b2;white-space:nowrap}
.diff{background:#3a2a15}
img.thumb{height:120px;border-radius:6px}
code{color:#8fd0ff}
"""


def _b64(p: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def contact_sheet(variants: List[Dict], out_html: Path) -> Path:
    """variants: [{name, params(str), stats(str), images:[Path,...]}]"""
    parts = [f"<style>{_CSS}</style>", "<h1>Crop tuning — contact sheet</h1>",
             "<div class='meta'>Compare coverage &amp; tightness across crop configs. "
             "Each section is one config run on the same sample frames.</div>"]
    for v in variants:
        parts.append(f"<h2>{html.escape(v['name'])}</h2>")
        parts.append(f"<div class='meta'>{html.escape(v.get('params',''))} &nbsp;·&nbsp; "
                     f"{html.escape(v.get('stats',''))}</div>")
        parts.append("<div class='grid'>")
        for img in v["images"]:
            parts.append(f"<img src='{_b64(img)}'/>")
        parts.append("</div>")
    out_html = Path(out_html)
    out_html.write_text("\n".join(parts))
    return out_html


def golden_report(scores: List[Dict], disagreements: Dict[str, List[Dict]],
                  meta: Dict, out_html: Path) -> Path:
    """scores: [{field, accuracy, n}]; disagreements: {field:[{crop, pseudo, golden}]}."""
    parts = [f"<style>{_CSS}</style>", "<h1>Pseudo-labels vs golden — accuracy</h1>",
             f"<div class='meta'>{html.escape(str(meta.get('n_reviewed',0)))} human-reviewed "
             f"crops · overall macro-accuracy "
             f"<b>{meta.get('overall',0)*100:.1f}%</b></div>"]
    parts.append("<table><tr><th>field</th><th>accuracy</th><th>n</th></tr>")
    for s in scores:
        pct = s["accuracy"] * 100
        parts.append(f"<tr><td class='k'>{html.escape(s['field'])}</td>"
                     f"<td>{pct:.0f}%</td><td>{s['n']}</td></tr>")
    parts.append("</table>")
    for field, rows in disagreements.items():
        if not rows:
            continue
        parts.append(f"<h2>{html.escape(field)} — disagreements (pseudo → golden)</h2>")
        parts.append("<table><tr><th>crop</th><th>pseudo</th><th>golden</th></tr>")
        for r in rows[:20]:
            parts.append(f"<tr><td class='k'>{html.escape(str(r['crop']))}</td>"
                         f"<td class='diff'>{html.escape(str(r['pseudo']))}</td>"
                         f"<td>{html.escape(str(r['golden']))}</td></tr>")
        parts.append("</table>")
    out_html = Path(out_html)
    out_html.write_text("\n".join(parts))
    return out_html


def caption_comparison(crops: List[Dict], variant_names: List[str],
                       fields: List[str], out_html: Path) -> Path:
    """crops: [{image:Path, labels:{variant_name:{field:val}}}]  -> side-by-side table."""
    parts = [f"<style>{_CSS}</style>", "<h1>Caption tuning — variant comparison</h1>",
             "<div class='meta'>Each row = one crop. Cells that DISAGREE across variants "
             "for a field are highlighted. Use this to pick model/prompt/params.</div>"]
    # agreement summary
    parts.append("<h2>Field agreement across variants</h2><table><tr><th>field</th>"
                 "<th>% crops where all variants agree</th></tr>")
    for f in fields:
        agree = 0
        for c in crops:
            vals = {str(c["labels"].get(v, {}).get(f)) for v in variant_names}
            if len(vals) == 1:
                agree += 1
        pct = 100 * agree / max(1, len(crops))
        parts.append(f"<tr><td class='k'>{html.escape(f)}</td><td>{pct:.0f}%</td></tr>")
    parts.append("</table>")

    # per-crop table
    parts.append("<h2>Per-crop outputs</h2><table><tr><th>crop</th><th>field</th>"
                 + "".join(f"<th>{html.escape(v)}</th>" for v in variant_names) + "</tr>")
    for c in crops:
        thumb = f"<img class='thumb' src='{_b64(c['image'])}'/>"
        parts.append(f"<tr><td rowspan='{len(fields)}'>{thumb}</td>")
        for i, f in enumerate(fields):
            vals = [str(c["labels"].get(v, {}).get(f, "")) for v in variant_names]
            diff = len(set(vals)) > 1
            if i > 0:
                parts.append("<tr>")
            parts.append(f"<td class='k'>{html.escape(f)}</td>")
            for val in vals:
                cls = " class='diff'" if diff else ""
                parts.append(f"<td{cls}>{html.escape(val)}</td>")
            parts.append("</tr>")
    parts.append("</table>")
    out_html = Path(out_html)
    out_html.write_text("\n".join(parts))
    return out_html
