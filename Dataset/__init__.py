"""FOROH Dataset package."""

import json
import os
from pathlib import Path

from .limuc import LIMUCDataset
from .transforms import get_transforms

_REGISTRY_PATH = Path(__file__).parent / "registry.json"

DATASETS = {
    "limuc": LIMUCDataset,
}

C_MAX = {
    "limuc": 3,
}


def _load_registry():
    with open(_REGISTRY_PATH) as f:
        return json.load(f)


def get_data_root(dataset_name):
    reg = _load_registry()
    entry = reg[dataset_name]
    env_var = entry.get("root_env")
    if env_var and os.environ.get(env_var):
        return Path(os.environ[env_var])
    project_root = Path(__file__).resolve().parents[1]
    return project_root / entry["default_relative"]


def build_datasets(dataset_name, fold=0, n_folds=10, seed=42, img_size=224):
    root = get_data_root(dataset_name)
    tr_tf = get_transforms("train", img_size)
    va_tf = get_transforms("val", img_size)

    ds_cls = DATASETS[dataset_name]
    kw = dict(fold=fold, n_folds=n_folds, seed=seed)

    # LIMUCDataset is strict by default: incomplete/ambiguous patient mapping
    # aborts the run rather than silently turning images into pseudo-patients.
    train_ds = ds_cls(root, "train", tr_tf, **kw)
    val_ds = ds_cls(root, "val", va_tf, **kw)
    test_ds = ds_cls(root, "test", va_tf)
    return train_ds, val_ds, test_ds
