"""Subject-disjoint train/test split for RAER.

The shipped split shares 93.6% of subjects between train and test (see
`DATA_PROVENANCE.md`), so it cannot be used to report generalisation. This builds
a replacement that keeps every clip of a subject on one side.

Two objectives pull against each other: no subject may straddle the boundary, and
the class balance should match on both sides. Since whole subjects move at once
and a subject's clips are not class-pure, the balance can only be approximated —
so the split is *checked and reported*, never assumed. `evaluate()` returns the
achieved per-class deviation, and `build()` refuses to return a split that leaks.

Confusion is the binding constraint: 37 clips across ~141 subjects, so subjects
carrying a confusion clip are placed first, before the greedy has spent its
budget elsewhere.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def group_by_identity(rows: List[Dict[str, Any]],
                      label_field: str = "raer_label",
                      id_field: str = "identity_key") -> Dict[str, Counter]:
    """identity -> Counter of its clips' classes."""
    groups: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        groups[r[id_field]][r.get(label_field) or ""] += 1
    return dict(groups)


def build(rows: List[Dict[str, Any]], test_frac: float = 0.2, seed: int = 13,
          label_field: str = "raer_label",
          id_field: str = "identity_key") -> Dict[str, Any]:
    """Assign whole identities to train/test, balancing classes greedily.

    Identities are placed rarest-class-first, largest-first. Each is put on the
    side that is furthest behind its quota for the classes that identity carries,
    which keeps the scarce classes from all landing on one side.
    """
    groups = group_by_identity(rows, label_field, id_field)
    totals = Counter()
    for c in groups.values():
        totals.update(c)
    classes = sorted(totals)

    # per-side quotas
    target = {"test": {c: totals[c] * test_frac for c in classes},
              "train": {c: totals[c] * (1 - test_frac) for c in classes}}
    have = {"test": Counter(), "train": Counter()}
    assign: Dict[str, str] = {}

    # rarity weight: a clip of a scarce class is worth more attention
    rarity = {c: 1.0 / max(totals[c], 1) for c in classes}

    def order_key(item: Tuple[str, Counter]):
        ident, cnt = item
        scarcity = max((rarity[c] for c in cnt), default=0.0)
        return (-scarcity, -sum(cnt.values()), ident)

    def cost(side: str, cnt: Counter) -> float:
        """How full this side becomes, on its most-constrained class."""
        worst = 0.0
        for c, k in cnt.items():
            t = target[side][c]
            if t <= 0:
                # this side wants none of this class; adding is maximally bad
                worst = max(worst, float("inf") if k else 0.0)
            else:
                worst = max(worst, (have[side][c] + k) / t)
        return worst

    for ident, cnt in sorted(groups.items(), key=order_key):
        side = min(("test", "train"), key=lambda s: (cost(s, cnt), s))
        assign[ident] = side
        have[side].update(cnt)

    train = [r for r in rows if assign[r[id_field]] == "train"]
    test = [r for r in rows if assign[r[id_field]] == "test"]
    return {"assign": assign, "train": train, "test": test,
            "evaluation": evaluate(train, test, label_field, id_field)}


def evaluate(train: List[Dict[str, Any]], test: List[Dict[str, Any]],
             label_field: str = "raer_label",
             id_field: str = "identity_key") -> Dict[str, Any]:
    """Verify disjointness and report the class balance actually achieved."""
    itr = {r[id_field] for r in train}
    ite = {r[id_field] for r in test}
    shared = itr & ite

    ctr, cte = Counter(r.get(label_field) for r in train), Counter(r.get(label_field) for r in test)
    ntr, nte = len(train), len(test)
    rows, worst = [], 0.0
    for c in sorted(set(ctr) | set(cte)):
        ptr = 100.0 * ctr[c] / ntr if ntr else 0.0
        pte = 100.0 * cte[c] / nte if nte else 0.0
        worst = max(worst, abs(ptr - pte))
        rows.append({"class": c, "train_n": ctr[c], "train_pct": ptr,
                     "test_n": cte[c], "test_pct": pte, "delta_pp": pte - ptr})

    return {
        "subject_disjoint": not shared,
        "shared_identities": len(shared),
        "n_train": ntr, "n_test": nte,
        "test_frac": round(nte / (ntr + nte), 4) if (ntr + nte) else 0.0,
        "n_train_identities": len(itr), "n_test_identities": len(ite),
        "max_class_delta_pp": round(worst, 2),
        "per_class": rows,
        "empty_test_classes": [r["class"] for r in rows if r["test_n"] == 0],
    }


def save(result: Dict[str, Any], out_dir: Path,
         id_field: str = "identity_key") -> Path:
    """Write split assignment + per-clip lists. Refuses to write a leaky split."""
    ev = result["evaluation"]
    if not ev["subject_disjoint"]:
        raise SystemExit(f"refusing to save: {ev['shared_identities']} identities "
                         f"appear on both sides — this is the bug we are fixing")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test"):
        with (out_dir / f"{name}.txt").open("w", encoding="utf-8") as fh:
            for r in result[name]:
                fh.write(f"{r['clip_stem']} {r.get('raer_label', '')} {r[id_field]}\n")
    (out_dir / "split.json").write_text(json.dumps(
        {"assign": result["assign"], "evaluation": ev}, indent=1), encoding="utf-8")
    print(f"[split] {ev['n_train']} train / {ev['n_test']} test clips, "
          f"{ev['n_train_identities']}/{ev['n_test_identities']} identities, "
          f"max class delta {ev['max_class_delta_pp']}pp -> {out_dir}")
    if ev["empty_test_classes"]:
        print(f"[split] WARNING: no test clips for {ev['empty_test_classes']} — "
              f"per-class metrics on those are undefined")
    return out_dir
