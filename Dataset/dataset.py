import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MAYO_CLASSES = ("Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3")
NUM_CLASSES = 4
N_FOLDS = 10
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png"}


def _grade_from_name(name):
    if name not in MAYO_CLASSES:
        raise ValueError(f"Unknown LIMUC grade folder: {name}")
    return MAYO_CLASSES.index(name)


def _image_index(root):
    index = {}
    for split_dir in ("train_and_validation_sets", "test_set"):
        base = root / split_dir
        for grade_dir in sorted(base.iterdir()):
            if not grade_dir.is_dir() or grade_dir.name not in MAYO_CLASSES:
                continue
            label = _grade_from_name(grade_dir.name)
            for path in sorted(grade_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTS:
                    index.setdefault(path.name, []).append((path, label))
    return index


def _read_fold_file(path, index):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for grade, names in data.items():
        label = _grade_from_name(grade)
        for name in names:
            if name not in index:
                raise FileNotFoundError(f"{name} from {path} was not found under LIMUC image folders")
            matches = [(img_path, indexed_label) for img_path, indexed_label in index[name] if indexed_label == label]
            if not matches:
                labels = sorted({indexed_label for _, indexed_label in index[name]})
                raise ValueError(f"Label mismatch for {name}: fold={label}, available={labels}")
            img_path, _ = matches[0]
            samples.append((img_path, label))
    return samples


def _read_test_set(root):
    samples = []
    for grade_dir in sorted((root / "test_set").iterdir()):
        if not grade_dir.is_dir() or grade_dir.name not in MAYO_CLASSES:
            continue
        label = _grade_from_name(grade_dir.name)
        for path in sorted(grade_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                samples.append((path, label))
    return samples


def load_official_splits(root):
    root = Path(root)
    fold_root = root / "cross_validation_folds_train_val_info"
    index = _image_index(root)
    folds = []
    for fold_idx in range(N_FOLDS):
        fold_dir = fold_root / f"fold_{fold_idx}"
        train_file, val_file = fold_dir / "train.json", fold_dir / "val.json"
        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(f"Missing official LIMUC fold files in {fold_dir}")
        folds.append({"train": _read_fold_file(train_file, index), "val": _read_fold_file(val_file, index)})
    return {"folds": folds, "test": _read_test_set(root)}


def compute_channel_stats(samples):
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path, _ in samples:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        pixel_sum += flat.sum(axis=0)
        pixel_sq_sum += np.square(flat).sum(axis=0)
        pixel_count += flat.shape[0]
    mean = pixel_sum / max(pixel_count, 1)
    var = pixel_sq_sum / max(pixel_count, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.tolist(), std.tolist()


def make_transforms(backbone, train_samples=None):
    name = backbone.lower()
    if name == "coatnet_2":
        return _coatnet_train_transform(), _coatnet_eval_transform(), {"image_size": 224, "normalization": "imagenet"}
    if train_samples is None:
        raise ValueError("Polat-style transforms require train_samples for fold-specific mean/std")
    mean, std = compute_channel_stats(train_samples)
    size = 299 if name == "inception_v3" else None
    return _polat_train_transform(mean, std, size), _polat_eval_transform(mean, std, size), {"image_size": size, "normalization": "fold_train_mean_std", "mean": mean, "std": std}


def _polat_train_transform(mean, std, size):
    steps = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-180, 180)),
    ]
    if size is not None:
        steps.append(transforms.Resize((size, size)))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def _polat_eval_transform(mean, std, size):
    steps = []
    if size is not None:
        steps.append(transforms.Resize((size, size)))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(steps)


def _coatnet_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, value="random"),
    ])


def _coatnet_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LIMUCDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, path.name
