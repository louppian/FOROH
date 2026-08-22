"""LIMUC dataset/transforms for E01, split mechanically from root 3_train.py."""

from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms


class LIMUCDataset(Dataset):
    C_MAX = 3

    def __init__(self, root, split="train", transform=None, fold_index=None,
                 n_folds=5, split_seed=1):
        self.root, self.transform, self.c_max = Path(root), transform, self.C_MAX

        if split == "test":
            self.samples = self._load_dir(self.root / "test_set", range(4))
        else:
            base = self.root / "train_and_validation_sets"
            patient_dir = self.root / "patient_based_classified_images"
            fname_to_pid = self._build_pid_map(patient_dir)

            all_samples = []
            for grade in range(4):
                gdir = self._grade_dir(base, grade)
                if gdir and gdir.exists():
                    for p in sorted(gdir.iterdir()):
                        if self._is_image(p):
                            pid = fname_to_pid.get(p.name, f"unk_{p.stem}")
                            all_samples.append((str(p), grade, pid))

            unique_pids = sorted({s[2] for s in all_samples})
            pid_labels = defaultdict(list)
            for _, g, pid in all_samples:
                pid_labels[pid].append(g)
            strat = [Counter(pid_labels[p]).most_common(1)[0][0] for p in unique_pids]

            if fold_index is not None:
                skf = StratifiedKFold(
                    n_splits=n_folds, shuffle=True, random_state=split_seed
                )
                tr, va = list(skf.split(unique_pids, strat))[fold_index]
                idx = tr if split == "train" else va
                selected = {unique_pids[i] for i in idx}
            else:
                selected = set(unique_pids)

            self.samples = [(p, g) for p, g, pid in all_samples if pid in selected]
            print(f"    [{split}] {len(self.samples)} images, {len(selected)} patients")

    @staticmethod
    def _is_image(p):
        return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    @staticmethod
    def _grade_dir(base, grade):
        for fmt in (f"Mayo {grade}", f"Mayo{grade}"):
            d = base / fmt
            if d.exists():
                return d
        return None

    @staticmethod
    def _build_pid_map(patient_dir):
        m = {}
        if patient_dir.exists():
            for folder in patient_dir.iterdir():
                if folder.is_dir():
                    for img in folder.rglob("*"):
                        if LIMUCDataset._is_image(img):
                            m[img.name] = folder.name
        return m

    def _load_dir(self, base, grades):
        samples = []
        for g in grades:
            gdir = self._grade_dir(base, g)
            if gdir and gdir.exists():
                for p in sorted(gdir.iterdir()):
                    if self._is_image(p):
                        samples.append((str(p), g))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img, label


def get_transforms(split, img_size=224):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])
