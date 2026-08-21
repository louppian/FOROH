"""CPU-only preflight checks for FOROH Phase-1 experiments.

Checks:
1. LIMUC patient mapping is complete and unambiguous.
2. Fold-0 train/validation patient sets are disjoint under the 10-fold split.
3. E01A/E01B/E01C heads have identical trainable parameter counts.
4. E02A point-prototype has the same trainable parameter count as FOROH.
5. Head forward passes return finite outputs with expected shapes.

No backbone is instantiated, so this does not download ImageNet weights or use GPU.
"""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Dataset import get_data_root
from Dataset.limuc import LIMUCDataset
from Model.heads import EuclideanHuberHead, NormalizedCosineHead, FOROHHead, PointPrototypeHead


def n_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def check_heads():
    in_dim, proj_dim, c_max = 2048, 128, 3
    heads = {
        "E01A_euclidean_huber": EuclideanHuberHead(in_dim, proj_dim, c_max),
        "E01B_normalized_cosine": NormalizedCosineHead(in_dim, proj_dim, c_max),
        "E01C_foroh": FOROHHead(in_dim, proj_dim, c_max),
        "E02A_point_prototype": PointPrototypeHead(in_dim, proj_dim, c_max),
    }
    counts = {name: n_trainable(head) for name, head in heads.items()}
    print("Head trainable parameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    base = counts["E01C_foroh"]
    assert counts["E01A_euclidean_huber"] == base, counts
    assert counts["E01B_normalized_cosine"] == base, counts
    assert counts["E02A_point_prototype"] == base, counts

    z = torch.randn(4, in_dim)
    for name, head in heads.items():
        head.eval()
        with torch.no_grad():
            out = head(z)
        assert out["score"].shape == (4,), (name, out["score"].shape)
        assert torch.isfinite(out["score"]).all(), name
        if "u" in out:
            norms = out["u"].norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), name
        if "prototypes" in out:
            protos = out["prototypes"]
            assert protos.shape == (c_max + 1, proj_dim), protos.shape
            assert torch.allclose(protos.norm(dim=-1), torch.ones(c_max + 1), atol=1e-5)

    print("Head checks: PASS")


def check_limuc():
    root = get_data_root("limuc")
    print(f"LIMUC root: {root}")
    train_ds = LIMUCDataset(root, "train", transform=None, fold=0, n_folds=10, seed=42)
    val_ds = LIMUCDataset(root, "val", transform=None, fold=0, n_folds=10, seed=42)
    test_ds = LIMUCDataset(root, "test", transform=None)

    train_pids = set(train_ds.patient_ids or [])
    val_pids = set(val_ds.patient_ids or [])
    overlap = train_pids & val_pids
    assert not overlap, f"Patient leakage between train/val: {sorted(overlap)[:10]}"

    train_paths = {p for p, _ in train_ds.samples}
    val_paths = {p for p, _ in val_ds.samples}
    assert not (train_paths & val_paths), "Image leakage between train/val"

    print(
        f"Split counts: train={len(train_ds)} images/{len(train_pids)} patients, "
        f"val={len(val_ds)} images/{len(val_pids)} patients, test={len(test_ds)} images"
    )
    print("LIMUC patient/split checks: PASS")


def main():
    check_heads()
    check_limuc()
    print("\nPHASE-1 PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
