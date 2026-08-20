"""Dataset structure verification."""

from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent / "data"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(path):
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in EXTS)


def get_size_mb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def verify_retinamnist():
    print("\n[RetinaMNIST]")
    path = ROOT / "retinamnist"
    if not path.exists():
        print("  NOT FOUND")
        return

    import numpy as np
    for npz in sorted(path.rglob("*.npz")):
        data = np.load(npz)
        print(f"  {npz.name}:")
        for split in ("train", "val", "test"):
            key = f"{split}_labels"
            if key in data:
                labels = data[key].flatten()
                dist = dict(sorted(Counter(labels).items()))
                print(f"    {split:5s} n={len(labels):4d}  dist={dist}")


def verify_aptos():
    print("\n[APTOS 2019]")
    path = ROOT / "aptos2019"
    if not path.exists():
        print("  NOT FOUND")
        return

    csv_file = path / "train.csv"
    if csv_file.exists():
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(f"  train.csv: {len(df)} rows  cols={list(df.columns)}")
        for grade, count in df["diagnosis"].value_counts().sort_index().items():
            print(f"    Grade {grade}: {count:5d} ({count / len(df) * 100:5.1f}%)")
    else:
        print("  WARN: train.csv not found")

    img_dir = path / "train_images"
    if img_dir.exists():
        print(f"  train_images/: {count_images(img_dir)} images")


def verify_limuc():
    print("\n[LIMUC]")
    path = ROOT / "limuc"
    if not path.exists():
        print("  NOT FOUND")
        return

    # Find Mayo class folders anywhere under path
    candidates = [path] + [d for d in path.rglob("*") if d.is_dir()]
    for root in candidates:
        mayo = {d.name: d for d in root.iterdir()
                if d.is_dir() and d.name.startswith("Mayo")}
        if len(mayo) >= 4:
            print(f"  Mayo folders in: {root.relative_to(ROOT)}")
            total = 0
            for name in sorted(mayo):
                n = count_images(mayo[name])
                total += n
                print(f"    {name}: {n:5d}")
            print(f"    Total: {total}")
            return

    zips = list(path.glob("*.zip"))
    if zips:
        print("  WARN: zip files not extracted:")
        for z in zips:
            print(f"    {z.name} ({z.stat().st_size / 1e6:.1f} MB)")


def summary():
    print("\n[Summary]")
    datasets = [
        ("RetinaMNIST", ROOT / "retinamnist", 4),
        ("APTOS 2019",  ROOT / "aptos2019",   4),
        ("LIMUC",       ROOT / "limuc",       3),
    ]
    print(f"  {'Dataset':15s} {'Images':>8s} {'Size (MB)':>10s} {'C_max':>6s}")
    for name, path, cmax in datasets:
        if path.exists():
            print(f"  {name:15s} {count_images(path):8d} {get_size_mb(path):10.1f} {cmax:6d}")
        else:
            print(f"  {name:15s} {'--':>8s} {'--':>10s} {cmax:6d}  MISSING")


if __name__ == "__main__":
    verify_retinamnist()
    verify_aptos()
    verify_limuc()
    summary()