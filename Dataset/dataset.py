import json
from pathlib import Path

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


IMAGE_SIZES = {"inception_v3": 299, "resnet18": 224, "coatnet_2": 224}


def make_transforms(backbone):
    size = IMAGE_SIZES[backbone.lower()]
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    info = {"image_size": size, "normalization": "imagenet"}
    return train_tf, eval_tf, info


class LIMUCDataset(Dataset):
    def __init__(self, samples, transform, cache=False):
        self.samples = samples
        self.transform = transform
        self.cache = {}
        if cache:
            for i, (path, _) in enumerate(samples):
                self.cache[i] = Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.cache[idx] if idx in self.cache else Image.open(path).convert("RGB")
        return self.transform(image), label, path.name
