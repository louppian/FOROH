#!/usr/bin/env python3
"""Print the actionable discrepancies from 00_Inventory reports."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Result" / "00_Inventory"


def load(name):
    p = OUT / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    print("\n=== CHECKPOINT RE-EVALUATION FAILURES ===")
    rep = load("checkpoint_verification_reeval.json")
    if rep is None:
        print("No re-evaluation report found")
    else:
        rows = rep if isinstance(rep, list) else rep.get("rows", rep.get("results", []))
        bad = []
        for r in rows:
            ok = r.get("reeval_exact")
            if ok is False:
                bad.append(r)
        if not bad:
            print("None")
        for r in bad:
            print(f"\n- checkpoint: {r.get('checkpoint_path') or r.get('checkpoint')}")
            print(f"  json      : {r.get('json_path')}")
            print(f"  method    : {r.get('method')}  dataset={r.get('dataset')}  backbone={r.get('backbone')}  fold={r.get('fold')}")
            for k in ("mae", "qwk", "acc", "macro_f1", "off_by_1"):
                a = r.get(f"json_{k}")
                b = r.get(f"reeval_{k}")
                d = r.get(f"diff_{k}")
                if a is not None or b is not None:
                    print(f"  {k:10s}: json={a}  reeval={b}  diff={d}")
            if r.get("error"):
                print(f"  error     : {r['error']}")

    print("\n=== PAPER TARGET MATCHING ===")
    pm = load("paper_match.json")
    if pm is None:
        print("No paper_match.json found")
        return
    rows = pm if isinstance(pm, list) else pm.get("rows", pm.get("results", []))
    order = ["METRIC_MISMATCH", "CONFIG_MISMATCH", "MISSING", "MATCH"]
    for status in order:
        subset = [r for r in rows if r.get("status") == status]
        print(f"\n[{status}] {len(subset)}")
        for r in subset:
            target = r.get("target_id") or r.get("id") or r.get("name")
            cand = r.get("json_path") or r.get("candidate_json") or r.get("candidate")
            print(f"- {target}")
            if cand:
                print(f"  candidate : {cand}")
            reason = r.get("reason") or r.get("details") or r.get("mismatches")
            if reason:
                print(f"  reason    : {reason}")


if __name__ == "__main__":
    main()
