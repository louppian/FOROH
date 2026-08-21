#!/usr/bin/env python3
"""Validate historical checkpoints against sibling results.json files.

Designed for the original local FOROH working directory, where .pt/.pth files
exist beside results.json even though weights are ignored by git.

Validation levels
-----------------
1. Structural (always):
   - checkpoint is readable by torch.load
   - model_state/state_dict exists
   - checkpoint args/fold agree with sibling results.json when both are present
   - recorded checkpoint metrics agree with the corresponding JSON fold metrics

2. Re-evaluation (--reevaluate):
   - rebuilds model and dataset from checkpoint/JSON args
   - loads model_state
   - evaluates the original test split
   - compares recomputed metrics to JSON metrics

Outputs a machine-readable report JSON and CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


METRICS = [
    "mae", "qwk", "acc", "macro_f1", "off_by_1",
    "recall_g0", "recall_g1", "recall_g2", "recall_g3", "recall_g4",
]
ARG_KEYS = [
    "method", "dataset", "backbone", "proj_dim", "loss_fn", "no_arccos",
    "fixed_w", "no_projector", "freeze_layers", "img_size", "n_folds",
    "seed", "augment_preset", "aux_ce", "aux_lambda", "freq_weight",
    "optimize_boundaries",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_results_json(folder: Path) -> Path | None:
    p = folder / "results.json"
    return p if p.exists() else None


def fold_from_name(path: Path) -> int | None:
    stem = path.stem.lower()
    if stem.startswith("fold"):
        tail = stem[4:]
        if tail.isdigit():
            return int(tail)
    return None


def extract_state(ckpt: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(ckpt, dict):
        for k in ("model_state", "state_dict", "model_state_dict"):
            v = ckpt.get(k)
            if isinstance(v, dict) and v:
                return v
        if ckpt and all(isinstance(k, str) for k in ckpt) and any(
            isinstance(v, torch.Tensor) for v in ckpt.values()
        ):
            return ckpt
    return None


def extract_args(ckpt: Any, json_obj: dict[str, Any]) -> dict[str, Any]:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        return dict(ckpt["args"])
    if isinstance(json_obj.get("args"), dict):
        return dict(json_obj["args"])
    return {}


def extract_fold(ckpt: Any, path: Path, json_obj: dict[str, Any]) -> int | None:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("fold"), int):
        return int(ckpt["fold"])
    f = fold_from_name(path)
    if f is not None:
        return f
    a = json_obj.get("args")
    if isinstance(a, dict) and isinstance(a.get("fold"), int) and a.get("fold") >= 0:
        return int(a["fold"])
    return None


def json_fold_metrics(obj: dict[str, Any], fold: int | None) -> dict[str, Any]:
    fr = obj.get("fold_results")
    if isinstance(fr, list) and fr:
        if fold is not None and 0 <= fold < len(fr) and isinstance(fr[fold], dict):
            return fr[fold]
        if len(fr) == 1 and isinstance(fr[0], dict):
            return fr[0]
    for k in ("test", "test_final", "test_metrics", "mean"):
        v = obj.get(k)
        if isinstance(v, dict):
            return v
    return {}


def checkpoint_metrics(ckpt: Any) -> dict[str, Any]:
    if not isinstance(ckpt, dict):
        return {}
    for k in ("test_final", "test_metrics", "test", "metrics"):
        v = ckpt.get(k)
        if isinstance(v, dict):
            return v
    return {}


def close(a: Any, b: Any, tol: float) -> bool | None:
    try:
        if a is None or b is None:
            return None
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except Exception:
        return False


def load_train_module(root: Path):
    p = root / "Experiment" / "train.py"
    spec = importlib.util.spec_from_file_location("foroh_repro_train", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def namespace_from_args(arg_dict: dict[str, Any]):
    # Fill fields required by Experiment/train.py and 3_train.py build helpers.
    defaults = {
        "method": "FOROH", "dataset": "limuc", "backbone": "resnet50",
        "proj_dim": 128, "loss_fn": "huber", "no_arccos": False,
        "fixed_w": False, "no_projector": False, "aux_ce": False,
        "aux_lambda": 0.3, "freq_weight": False, "optimize_boundaries": False,
        "boundary_grid": 200, "epochs": 50, "batch_size": 64,
        "lr": 1e-4, "lr_head": 1e-3, "weight_decay": 1e-4,
        "img_size": 224, "freeze_layers": 2, "patience": 10,
        "optimizer": "adamw", "scheduler": "cosine", "scheduler_patience": 10,
        "oversample": False, "augment_preset": "default", "exp": 0,
        "n_folds": 5, "fold": 0, "seed": 42, "num_workers": 4,
        "output_dir": "Result",
    }
    defaults.update({k: v for k, v in arg_dict.items() if v is not None})
    if defaults["backbone"] == "inception_v3" and defaults["img_size"] == 224:
        defaults["img_size"] = 299
    if defaults["backbone"] == "efficientnet_b3" and defaults["img_size"] == 224:
        defaults["img_size"] = 256
    return argparse.Namespace(**defaults)


def reevaluate(root: Path, ckpt: Any, arg_dict: dict[str, Any], fold: int | None,
               batch_size: int | None, num_workers: int | None) -> dict[str, Any]:
    M = load_train_module(root)
    args = namespace_from_args(arg_dict)
    if fold is not None:
        args.fold = fold
    if batch_size is not None:
        args.batch_size = batch_size
    if num_workers is not None:
        args.num_workers = num_workers

    model, head, c_max = M.build_model(args)
    state = extract_state(ckpt)
    if state is None:
        raise RuntimeError("No model_state/state_dict in checkpoint")

    missing, unexpected = model.load_state_dict(state, strict=False)
    # A historical checkpoint may include small auxiliary differences; report them
    # but do not suppress actual inference when strict=False can load the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    _train_ds, _val_ds, test_ds = M.T.build_datasets(args, args.fold)
    loader = DataLoader(
        test_ds,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    theta_boundaries = None
    if isinstance(ckpt, dict) and ckpt.get("theta_boundaries") is not None:
        theta_boundaries = ckpt.get("theta_boundaries")
    if theta_boundaries is not None:
        import numpy as np
        theta_boundaries = np.asarray(theta_boundaries, dtype=float)

    metrics = M.T.evaluate(
        model, loader, args.method, head, c_max, device,
        theta_boundaries=theta_boundaries,
    )
    return {
        "metrics": metrics,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "device": str(device),
        "test_size": len(test_ds),
    }


def verify_one(root: Path, ckpt_path: Path, tol: float, do_reeval: bool,
               batch_size: int | None, num_workers: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint": str(ckpt_path.relative_to(root)),
        "folder": str(ckpt_path.parent.relative_to(root)),
        "load_ok": False,
        "state_ok": False,
        "json_found": False,
        "metadata_ok": None,
        "recorded_metric_ok": None,
        "reeval_ok": None,
    }
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        row["load_ok"] = True
    except Exception as e:
        row["error"] = f"torch.load: {e!r}"
        return row

    state = extract_state(ckpt)
    row["state_ok"] = state is not None
    if state is not None:
        row["state_tensor_count"] = sum(isinstance(v, torch.Tensor) for v in state.values())

    jp = find_results_json(ckpt_path.parent)
    obj: dict[str, Any] = {}
    if jp is not None:
        row["json_found"] = True
        row["json_path"] = str(jp.relative_to(root))
        try:
            obj = load_json(jp)
        except Exception as e:
            row["json_error"] = repr(e)

    fold = extract_fold(ckpt, ckpt_path, obj)
    row["fold"] = fold
    cargs = extract_args(ckpt, obj)
    jargs = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    row["method"] = cargs.get("method")
    row["dataset"] = cargs.get("dataset")
    row["backbone"] = cargs.get("backbone")

    diffs = {}
    if jargs and isinstance(ckpt, dict) and isinstance(ckpt.get("args"), dict):
        for k in ARG_KEYS:
            a, b = ckpt["args"].get(k), jargs.get(k)
            if a is not None and b is not None and a != b:
                diffs[k] = {"checkpoint": a, "json": b}
    row["metadata_diffs"] = diffs
    row["metadata_ok"] = (not diffs) if jargs else None

    jm = json_fold_metrics(obj, fold)
    cm = checkpoint_metrics(ckpt)
    rec_cmp = {}
    if jm and cm:
        oks = []
        for k in METRICS:
            if k in jm and k in cm:
                ok = close(jm.get(k), cm.get(k), tol)
                rec_cmp[k] = {
                    "json": jm.get(k), "checkpoint": cm.get(k), "ok": ok,
                }
                if ok is not None:
                    oks.append(ok)
        row["recorded_metric_ok"] = all(oks) if oks else None
    row["recorded_metric_compare"] = rec_cmp

    if do_reeval:
        try:
            ev = reevaluate(root, ckpt, cargs, fold, batch_size, num_workers)
            row["reevaluated"] = ev
            cmp = {}
            oks = []
            for k in METRICS:
                if k in jm and k in ev["metrics"]:
                    ok = close(jm.get(k), ev["metrics"].get(k), tol)
                    cmp[k] = {
                        "json": jm.get(k), "reevaluated": ev["metrics"].get(k), "ok": ok,
                    }
                    if ok is not None:
                        oks.append(ok)
            row["reeval_metric_compare"] = cmp
            row["reeval_ok"] = all(oks) if oks else None
        except Exception as e:
            row["reeval_ok"] = False
            row["reeval_error"] = repr(e)
            row["reeval_traceback"] = traceback.format_exc()

    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--search-root", action="append", default=["outputs", "Result"],
                   help="Relative directory to scan; repeatable")
    p.add_argument("--reevaluate", action="store_true")
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override eval batch size base value")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    root = args.root.resolve()
    out = (args.out or root / "Result" / "00_Inventory").resolve()
    out.mkdir(parents=True, exist_ok=True)

    ckpts: list[Path] = []
    for rel in args.search_root:
        base = root / rel
        if not base.exists():
            continue
        for ext in ("*.pt", "*.pth", "*.ckpt"):
            ckpts.extend(base.rglob(ext))
    ckpts = sorted(set(p.resolve() for p in ckpts))

    rows = [
        verify_one(root, p, args.tolerance, args.reevaluate,
                   args.batch_size, args.num_workers)
        for p in ckpts
    ]

    suffix = "reeval" if args.reevaluate else "structural"
    json_out = out / f"checkpoint_verification_{suffix}.json"
    csv_out = out / f"checkpoint_verification_{suffix}.csv"
    json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    fields = [
        "checkpoint", "folder", "method", "dataset", "backbone", "fold",
        "load_ok", "state_ok", "json_found", "metadata_ok",
        "recorded_metric_ok", "reeval_ok", "error", "reeval_error",
    ]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Checkpoints scanned: {len(rows)}")
    print(f"Load OK: {sum(bool(r.get('load_ok')) for r in rows)}")
    print(f"State OK: {sum(bool(r.get('state_ok')) for r in rows)}")
    print(f"Sibling JSON: {sum(bool(r.get('json_found')) for r in rows)}")
    if args.reevaluate:
        print(f"Re-evaluation exact within tol={args.tolerance}: "
              f"{sum(r.get('reeval_ok') is True for r in rows)}")
    print(f"Wrote: {json_out}")
    print(f"Wrote: {csv_out}")


if __name__ == "__main__":
    main()
