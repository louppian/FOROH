#!/usr/bin/env python3
"""Print actionable discrepancies from 00_Inventory reports."""

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


def fmt_metrics(d):
    if not isinstance(d, dict):
        return ""
    keys = ("mae", "qwk", "acc", "macro_f1")
    vals = []
    for k in keys:
        if d.get(k) is not None:
            vals.append(f"{k}={d[k]:.4f}" if isinstance(d[k], (int, float)) else f"{k}={d[k]}")
    return " ".join(vals)


def main():
    print("\n=== CHECKPOINT RE-EVALUATION FAILURES ===")
    rep = load("checkpoint_verification_reeval.json")
    if rep is None:
        print("No re-evaluation report found")
    else:
        rows = rep if isinstance(rep, list) else rep.get("rows", rep.get("results", []))
        bad = [r for r in rows if r.get("reeval_exact") is False]
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
            target = r.get("target") or {}
            best = r.get("best") or {}
            target_id = target.get("id") or "<unnamed target>"
            print(f"- {target_id}  ({target.get('section', '')})")
            print(f"  target    : {target.get('dataset')}/{target.get('method')}/{target.get('backbone')}  {fmt_metrics(target.get('metrics', {}))}")

            if best:
                print(f"  candidate : {best.get('json_path')}")
                print(f"  config    : fold={best.get('fold')} n_folds={best.get('n_folds')} proj={best.get('proj_dim')} loss={best.get('loss_fn')} freeze={best.get('freeze_layers')}")
                print(f"  result    : {fmt_metrics(best)}")

            if status == "METRIC_MISMATCH":
                diffs = r.get("metric_diffs") or {}
                for k, d in diffs.items():
                    print(f"  diff {k:8s}: target={d.get('target')} result={d.get('result')} delta={d.get('delta')}")
            elif status == "CONFIG_MISMATCH":
                diffs = r.get("config_diffs") or {}
                for k, d in diffs.items():
                    print(f"  diff {k:18s}: target={d.get('target')} result={d.get('result')}")
            elif status == "MISSING":
                print("  candidate : none")


if __name__ == "__main__":
    main()
