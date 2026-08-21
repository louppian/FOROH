#!/usr/bin/env python3
"""Match historical inventory rows against paper targets.

Requires inventory.py to have produced Result/00_Inventory/inventory.json.
Classifies each target as MATCH, CONFIG_MISMATCH, METRIC_MISMATCH, or MISSING.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CONFIG_KEYS = [
    "dataset", "method", "backbone", "fold", "n_folds", "proj_dim",
    "loss_fn", "freeze_layers", "no_arccos", "fixed_w", "no_projector",
]


def metric_error(row: dict[str, Any], target: dict[str, Any]) -> float:
    vals = []
    for k, tv in target.get("metrics", {}).items():
        rv = row.get(k)
        if rv is None:
            continue
        vals.append(abs(float(rv) - float(tv)))
    return sum(vals) / len(vals) if vals else float("inf")


def config_mismatch(row: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    bad = {}
    for k in CONFIG_KEYS:
        if k not in target:
            continue
        tv = target[k]
        rv = row.get(k)
        # fold=-1 target describes aggregate runs; rely on fold_results_count instead.
        if k == "fold" and tv == -1:
            continue
        if rv is not None and rv != tv:
            bad[k] = {"target": tv, "result": rv}
        elif rv is None:
            bad[k] = {"target": tv, "result": None}
    if target.get("fold") == -1 and target.get("n_folds"):
        cnt = row.get("fold_results_count")
        if cnt != target["n_folds"]:
            bad["fold_results_count"] = {"target": target["n_folds"], "result": cnt}
    return bad


def main() -> None:
    p = argparse.ArgumentParser()
    root_default = Path(__file__).resolve().parents[2]
    p.add_argument("--root", type=Path, default=root_default)
    p.add_argument("--metric-tol", type=float, default=0.002)
    args = p.parse_args()

    root = args.root.resolve()
    inv_path = root / "Result" / "00_Inventory" / "inventory.json"
    target_path = Path(__file__).with_name("paper_targets.json")
    if not inv_path.exists():
        raise SystemExit(f"Missing {inv_path}; run inventory.py first")

    rows = json.loads(inv_path.read_text(encoding="utf-8"))
    targets = json.loads(target_path.read_text(encoding="utf-8"))
    report = []

    for t in targets:
        broad = [r for r in rows if r.get("dataset") == t.get("dataset") and
                 r.get("method") == t.get("method") and
                 r.get("backbone") == t.get("backbone")]
        exact_cfg = [(r, config_mismatch(r, t)) for r in broad]
        exact_cfg = [(r, d) for r, d in exact_cfg if not d]

        if exact_cfg:
            ranked = sorted(exact_cfg, key=lambda x: metric_error(x[0], t))
            best = ranked[0][0]
            diffs = {}
            for k, tv in t.get("metrics", {}).items():
                rv = best.get(k)
                if rv is None:
                    diffs[k] = {"target": tv, "result": None, "delta": None}
                else:
                    d = float(rv) - float(tv)
                    if abs(d) > args.metric_tol:
                        diffs[k] = {"target": tv, "result": rv, "delta": d}
            status = "MATCH" if not diffs else "METRIC_MISMATCH"
            report.append({"target": t, "status": status, "best": best,
                           "metric_diffs": diffs})
            continue

        if broad:
            ranked = sorted(broad, key=lambda r: metric_error(r, t))
            best = ranked[0]
            report.append({"target": t, "status": "CONFIG_MISMATCH", "best": best,
                           "config_diffs": config_mismatch(best, t)})
        else:
            report.append({"target": t, "status": "MISSING", "best": None})

    out = root / "Result" / "00_Inventory" / "paper_match.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    counts = {}
    for r in report:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("Paper target matching")
    for k in ("MATCH", "METRIC_MISMATCH", "CONFIG_MISMATCH", "MISSING"):
        print(f"  {k:16s}: {counts.get(k, 0)}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
