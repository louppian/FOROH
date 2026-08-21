"""LIMUC dataset — 11,276 endoscopy images, Mayo 0-3, patient-level split."""

from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


class LIMUCDataset(Dataset):

    C_MAX = 3

    def __init__(self, root, split="train", transform=None,
                 fold=None, n_folds=10, seed=42, strict_patient_map=True):
        self.root = Path(root)
        self.transform = transform
        self.c_max = self.C_MAX
        self.patient_ids = None

        if split == "test":
            self.samples = self._load_dir(self.root / "test_set", range(4))
        else:
            base = self.root / "train_and_validation_sets"
            patient_dir = self.root / "patient_based_classified_images"
            fname_to_pid, ambiguous = self._build_pid_map(patient_dir)

            if strict_patient_map and ambiguous:
                preview = list(ambiguous.items())[:10]
                raise RuntimeError(
                    f"LIMUC patient map has {len(ambiguous)} ambiguous filenames. "
                    f"Examples: {preview}"
                )

            all_samples = []
            unmapped = []
            train_name_counts = Counter()
            for grade in range(4):
                gdir = self._grade_dir(base, grade)
                if gdir and gdir.exists():
                    for p in sorted(gdir.iterdir()):
                        if not self._is_image(p):
                            continue
                        train_name_counts[p.name] += 1
                        pid = fname_to_pid.get(p.name)
                        if pid is None:
                            unmapped.append(p.name)
                            if strict_patient_map:
                                continue
                            pid = f"unk_{p.stem}"
                        all_samples.append((str(p), grade, pid))

            duplicate_train_names = [k for k, v in train_name_counts.items() if v > 1]
            if strict_patient_map and duplicate_train_names:
                raise RuntimeError(
                    f"LIMUC train/val folders contain {len(duplicate_train_names)} duplicate "
                    f"filenames across grade folders. Examples: {duplicate_train_names[:10]}"
                )
            if strict_patient_map and unmapped:
                raise RuntimeError(
                    f"LIMUC patient mapping missing for {len(unmapped)} train/val images. "
                    f"Examples: {unmapped[:10]}"
                )

            print(
                "    [patient-map] "
                f"mapped={len(all_samples) - (0 if strict_patient_map else len(unmapped))} "
                f"unmapped={len(unmapped)} ambiguous={len(ambiguous)}"
            )

            unique_pids = sorted({s[2] for s in all_samples})
            pid_labels = defaultdict(list)
            for _, g, pid in all_samples:
                pid_labels[pid].append(g)
            strat = [Counter(pid_labels[p]).most_common(1)[0][0] for p in unique_pids]

            if fold is not None:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                tr, va = list(skf.split(unique_pids, strat))[fold]
                idx = tr if split == "train" else va
                selected = {unique_pids[i] for i in idx}
            else:
                selected = set(unique_pids)

            selected_samples = [(p, g, pid) for p, g, pid in all_samples if pid in selected]
            self.samples = [(p, g) for p, g, _ in selected_samples]
            self.patient_ids = [pid for _, _, pid in selected_samples]
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
        mapping = {}
        ambiguous = {}
        if patient_dir.exists():
            for folder in patient_dir.iterdir():
                if not folder.is_dir():
                    continue
                for img in folder.rglob("*"):
                    if not LIMUCDataset._is_image(img):
                        continue
                    old = mapping.get(img.name)
                    if old is not None and old != folder.name:
                        ambiguous.setdefault(img.name, {old}).add(folder.name)
                    else:
                        mapping[img.name] = folder.name
        ambiguous = {k: sorted(v) for k, v in ambiguous.items()}
        return mapping, ambiguous

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
