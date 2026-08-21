#!/usr/bin/env python3
"""Inventory every historical results.json under outputs/ and Result/.

The script is designed for the original local FOROH folder, where checkpoints
may exist beside JSON files even though model weights are ignored by git.

Outputs:
  Result/00_Inventory/inventory.json
  Result/00_Inventory/inventory.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "mae", "qwk", "acc", "macro_f1", "off_by_1",
    "recall_g0", "recall_g1", "recall_g2", "recall_g3", "recall_g4",
]
ARG_KEYS = [
    "method", "dataset", "backbone", "proj_dim", "no_arccos", "fixed_w",
    "no_projector", "loss_fn", "optimize_boundaries", "freq_weight", "aux_ce",
    "aux_lambda", "epochs", "batch_size", "lr", "lr_head", "weight_decay",
    "img_size", "freeze_layers", "patience", "optimizer", "scheduler",
    "oversample", "augment_preset", "exp", "n_folds", "fold", "seed",
]


def _metric_block(obj: dict[str, Any]) -> dict[str, Any]:
    for key in ("mean", "test", "test_final", "test_metrics"):
        v = obj.get(key)
        if isinstance(v, dict) and any(k in v for k in METRIC_KEYS):
            return v
    fr = obj.get("fold_results")
    if isinstance(fr, list) and len(fr) == 1 and isinstance(fr[0], dict):
        return fr[0]
    return {}


def _fold_count(obj: dict[str, Any]) -> int | None:
    fr = obj.get("fold_results")
    if isinstance(fr, list):
        return len(fr)
    return 1 if _metric_block(obj) else None


def _checkpoints(folder: Path) -> list[str]:
    exts = {".pt", ".pth", ".ckpt"}
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def scan(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for base_name in ("outputs", "Result"):
        base = root / base_name
        if not base.exists():
            continue
        for path in sorted(base.rglob("results.json")):
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                rows.append({
                    "json_path": str(path.relative_to(root)),
                    "parse_error": repr(e),
                    "checkpoints": _checkpoints(path.parent),
                })
                continue

            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            metrics = _metric_block(obj)
            row: dict[str, Any] = {
                "json_path": str(path.relative_to(root)),
                "folder": str(path.parent.relative_to(root)),
                "fold_results_count": _fold_count(obj),
                "flags": obj.get("flags", []),
                "checkpoints": _checkpoints(path.parent),
                "checkpoint_count": len(_checkpoints(path.parent)),
            }
            for k in ARG_KEYS:
                row[k] = args.get(k)
            for k in METRIC_KEYS:
                row[k] = metrics.get(k)
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    root = args.root.resolve()
    out = (args.out or root / "Result" / "00_Inventory").resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = scan(root)
    (out / "inventory.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    fields = [
        "json_path", "folder", "method", "dataset", "backbone", "fold",
        "n_folds", "fold_results_count", "checkpoint_count", "checkpoints",
        "proj_dim", "loss_fn", "no_arccos", "fixed_w", "no_projector",
        "batch_size", "lr", "lr_head", "optimizer", "scheduler",
        "freeze_layers", "seed", "mae", "qwk", "acc", "macro_f1",
        "off_by_1", "recall_g0", "recall_g1", "recall_g2", "recall_g3",
        "recall_g4", "flags", "parse_error",
    ]
    with (out / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["checkpoints"] = ";".join(rr.get("checkpoints") or [])
            rr["flags"] = json.dumps(rr.get("flags") or [], ensure_ascii=False)
            w.writerow(rr)

    with_ckpt = sum(bool(r.get("checkpoint_count")) for r in rows)
    print(f"Found {len(rows)} results.json files")
    print(f"Folders with checkpoint(s): {with_ckpt}")
    print(f"Wrote: {out / 'inventory.json'}")
    print(f"Wrote: {out / 'inventory.csv'}")


if __name__ == "__main__":
    main()
