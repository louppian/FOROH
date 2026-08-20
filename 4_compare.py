"""
FOROH Results Viewer
====================
Scans output directories and displays comparison tables.

Usage:
  python compare.py                          # all results
  python compare.py --dir outputs/freq       # specific directory
  python compare.py --dataset limuc          # filter by dataset
  python compare.py --group dataset          # group tables by dataset
  python compare.py --sort qwk              # sort by metric
  python compare.py --csv                    # export CSV
"""

import json, argparse, sys
from pathlib import Path
from collections import defaultdict


def load_results(base_dir):
    """Scan all results.json files under base_dir."""
    base = Path(base_dir)
    entries = []

    for rj in sorted(base.rglob("results.json")):
        try:
            data = json.loads(rj.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        args = data.get("args", {})
        mean = data.get("mean", {})
        std = data.get("std", {})
        folds = data.get("fold_results", [])
        flags = data.get("flags", [])

        if not mean:
            continue

        # Build tag from path
        tag = rj.parent.name

        entry = {
            "tag": tag,
            "path": str(rj.parent.relative_to(base)),
            "method": args.get("method", "?"),
            "dataset": args.get("dataset", "?"),
            "backbone": args.get("backbone", "resnet50"),
            "n_folds": len(folds),
            "flags": flags,
            # key metrics
            "mae":  mean.get("mae"),
            "qwk":  mean.get("qwk"),
            "acc":  mean.get("acc"),
            "f1":   mean.get("macro_f1"),
            "ob1":  mean.get("off_by_1"),
            # std
            "mae_std": std.get("mae"),
            "qwk_std": std.get("qwk"),
            # class recall
        }

        # c_max detection
        c_max = args.get("c_max") or {"limuc": 3, "retinamnist": 4,
                                       "aptos": 4, "kneexray": 4}.get(
                                           args.get("dataset"), 4)
        entry["c_max"] = c_max
        for i in range(c_max + 1):
            key = f"recall_g{i}"
            entry[f"G{i}"] = mean.get(key)

        # extra info
        entry["freq_weight"] = args.get("freq_weight", False)
        entry["aux_ce"] = args.get("aux_ce", False)
        entry["aux_lambda"] = args.get("aux_lambda")
        entry["opt_bound"] = args.get("optimize_boundaries", False)
        entry["no_arccos"] = args.get("no_arccos", False)
        entry["proj_dim"] = args.get("proj_dim", 128)
        entry["loss_fn"] = args.get("loss_fn", "huber")
        entry["freeze_layers"] = args.get("freeze_layers", 2)
        entry["seed"] = args.get("seed", 42)

        entries.append(entry)

    return entries


def fmt(val, width=7, prec=3):
    """Format a metric value."""
    if val is None:
        return " " * width
    return f"{val:>{width}.{prec}f}"


def fmt_pm(mean_val, std_val, width=13, prec=3):
    """Format mean±std."""
    if mean_val is None:
        return " " * width
    if std_val is not None and std_val > 0:
        return f"{mean_val:.{prec}f}±{std_val:.{prec}f}"
    return f"{mean_val:.{prec}f}"


def print_table(entries, title="", sort_key="qwk", show_recall=True):
    """Print a formatted comparison table."""
    if not entries:
        return

    entries = sorted(entries, key=lambda e: -(e.get(sort_key) or 0))
    c_max = max(e["c_max"] for e in entries)

    # Best values for highlighting
    best = {}
    for k in ["mae", "qwk", "acc", "f1"]:
        vals = [e[k] for e in entries if e[k] is not None]
        if vals:
            best[k] = min(vals) if k == "mae" else max(vals)

    if title:
        print(f"\n{'─'*80}")
        print(f"  {title}")
        print(f"{'─'*80}")

    # Header
    recall_hdr = "".join(f"{'G'+str(i):>7}" for i in range(c_max + 1)) if show_recall else ""
    folds_hdr = "folds"
    flags_hdr = "flags"

    hdr = (f"  {'Tag':<36} {'MAE':>13} {'QWK':>13} {'ACC':>7} "
           f"{'F1':>7} {folds_hdr:>5} {recall_hdr}")
    print(hdr)
    print(f"  {'─'*34}  {'─'*13} {'─'*13} {'─'*7} {'─'*7} {'─'*5} "
          + ("─" * 7 * (c_max + 1) if show_recall else ""))

    for e in entries:
        mae_str = fmt_pm(e["mae"], e.get("mae_std"))
        qwk_str = fmt_pm(e["qwk"], e.get("qwk_std"))

        # Highlight best
        mae_mark = "*" if e["mae"] == best.get("mae") else " "
        qwk_mark = "*" if e["qwk"] == best.get("qwk") else " "

        recall_str = ""
        if show_recall:
            for i in range(c_max + 1):
                recall_str += fmt(e.get(f"G{i}"), 7, 3)

        n = e["n_folds"]
        fold_str = f"{n:>5}" if n > 1 else f"{'1':>5}"

        tag = e["tag"]
        if len(tag) > 36:
            tag = tag[:33] + "..."

        line = (f"  {tag:<36}{mae_mark}{mae_str:>12} {qwk_mark}{qwk_str:>12} "
                f"{fmt(e['acc'])} {fmt(e['f1'])} {fold_str} {recall_str}")

        # Append flags
        fl = []
        if e.get("freq_weight"): fl.append("freq")
        if e.get("aux_ce"): fl.append(f"aux{e['aux_lambda']}")
        if e.get("opt_bound"): fl.append("opt")
        if e.get("no_arccos"): fl.append("cos")
        if fl:
            line += f"  [{','.join(fl)}]"

        print(line)

    print()


def print_summary(entries):
    """Print high-level stats."""
    datasets = set(e["dataset"] for e in entries)
    methods = set(e["method"] for e in entries)

    print(f"\n{'='*80}")
    print(f"  FOROH Results Summary")
    print(f"  {len(entries)} experiments | "
          f"Datasets: {', '.join(sorted(datasets))} | "
          f"Methods: {', '.join(sorted(methods))}")
    print(f"{'='*80}")


def main():
    p = argparse.ArgumentParser(description="FOROH Results Viewer")
    p.add_argument("--dir", default="outputs", help="Root output directory")
    p.add_argument("--dataset", help="Filter by dataset")
    p.add_argument("--method", help="Filter by method")
    p.add_argument("--group", default="dataset",
                   choices=["dataset", "method", "none"],
                   help="Group tables by")
    p.add_argument("--sort", default="qwk",
                   choices=["qwk", "mae", "acc", "f1"],
                   help="Sort metric (descending for QWK/ACC/F1, ascending for MAE)")
    p.add_argument("--no-recall", action="store_true",
                   help="Hide class-wise recall columns")
    p.add_argument("--csv", action="store_true",
                   help="Export as CSV instead of table")
    args = p.parse_args()

    entries = load_results(args.dir)
    if not entries:
        print(f"No results found in {args.dir}/")
        print(f"Expected: {{dir}}/**/results.json")
        sys.exit(1)

    # Filter
    if args.dataset:
        entries = [e for e in entries if e["dataset"] == args.dataset]
    if args.method:
        entries = [e for e in entries if e["method"] == args.method]

    if not entries:
        print("No results match filters.")
        sys.exit(1)

    # CSV export
    if args.csv:
        c_max = max(e["c_max"] for e in entries)
        cols = ["tag", "method", "dataset", "backbone", "n_folds",
                "mae", "qwk", "acc", "f1", "ob1"]
        cols += [f"G{i}" for i in range(c_max + 1)]
        cols += ["freq_weight", "aux_ce", "aux_lambda", "opt_bound",
                 "no_arccos", "loss_fn", "proj_dim", "seed"]

        print(",".join(cols))
        for e in sorted(entries, key=lambda x: (x["dataset"], -(x.get("qwk") or 0))):
            vals = []
            for c in cols:
                v = e.get(c, "")
                if v is None:
                    v = ""
                elif isinstance(v, float):
                    v = f"{v:.4f}"
                elif isinstance(v, bool):
                    v = str(v)
                vals.append(str(v))
            print(",".join(vals))
        return

    # Display
    print_summary(entries)

    if args.group == "dataset":
        for ds in sorted(set(e["dataset"] for e in entries)):
            sub = [e for e in entries if e["dataset"] == ds]
            c_max = max(e["c_max"] for e in sub)
            dist_info = {
                "limuc": "G0:54% G1:28% G2:11% G3:8%",
                "aptos": "G0:49% G1:10% G2:27% G3:5% G4:8%",
                "kneexray": "G0:40% G1:18% G2:26% G3:13% G4:3%",
                "retinamnist": "DR 0-4",
            }
            title = f"{ds.upper()} (C={c_max+1})  {dist_info.get(ds, '')}"
            print_table(sub, title, args.sort, not args.no_recall)

    elif args.group == "method":
        for m in sorted(set(e["method"] for e in entries)):
            sub = [e for e in entries if e["method"] == m]
            print_table(sub, m, args.sort, not args.no_recall)

    else:
        print_table(entries, "All Results", args.sort, not args.no_recall)

    # Cross-dataset comparison for FOROH variants
    foroh = [e for e in entries if e["method"] == "FOROH"]
    if len(set(e["dataset"] for e in foroh)) > 1:
        print(f"{'─'*80}")
        print(f"  FOROH Cross-Dataset QWK")
        print(f"{'─'*80}")

        # Find unique configs (by flags combination)
        configs = defaultdict(dict)
        for e in foroh:
            flags_key = tuple(sorted(e.get("flags", [])))
            tag_short = e["tag"].replace(f"_{e['dataset']}", "").replace("FOROH_", "")
            if not tag_short or tag_short == "FOROH":
                tag_short = "default"
            configs[tag_short][e["dataset"]] = e["qwk"]

        datasets = sorted(set(e["dataset"] for e in foroh))
        hdr = f"  {'Config':<30}" + "".join(f"{d:>12}" for d in datasets)
        print(hdr)
        print(f"  {'─'*28}  " + "─" * 12 * len(datasets))
        for cfg, vals in sorted(configs.items()):
            row = f"  {cfg:<30}"
            for d in datasets:
                v = vals.get(d)
                row += f"{v:>12.4f}" if v else f"{'—':>12}"
            print(row)
        print()


if __name__ == "__main__":
    main()