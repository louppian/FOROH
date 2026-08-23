from datasets import load_dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ============================================================
# Paths
# ============================================================
ROOT = Path(r"D:\FOROH\Dataset\TAIX-Ray")
IMAGE_DIR = ROOT / "images"
META_DIR = ROOT / "metadata"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_PATH = META_DIR / "annotation.csv"
SPLIT_PATH = META_DIR / "split.csv"


# ============================================================
# Load TAIX-Ray 512px version
# streaming=True:
#   - 전체 parquet를 로컬 cache에 먼저 받지 않음
#   - sample 단위로 읽어서 바로 PNG로 저장
# ============================================================
print("Loading TAIX-Ray default (512px) in streaming mode...")

dataset = load_dataset(
    "TLAIM/TAIX-Ray",
    name="default",
    streaming=True,
)

metadata = []

for split_name, split_dataset in dataset.items():

    print(f"\n========== {split_name.upper()} ==========")

    for item in tqdm(
        split_dataset,
        desc=f"Downloading {split_name}",
        unit="image",
    ):
        # copy: 원본 streaming item 수정 방지
        row = dict(item)

        uid = row["UID"]

        # PIL Image
        image = row.pop("Image")

        image_path = IMAGE_DIR / f"{uid}.png"

        # 이미 받은 영상이면 다시 쓰지 않음
        if not image_path.exists():
            image.save(
                image_path,
                format="PNG",
            )

        # HF에서 제공하는 split 정보 보존
        row["Split"] = split_name

        metadata.append(row)


# ============================================================
# Metadata
# ============================================================
df = pd.DataFrame(metadata)

print("\nDownloaded metadata:")
print(df.shape)
print(df.columns.tolist())


# ------------------------------------------------------------
# annotation.csv
# 논문에서 정의한 annotation fields
# ------------------------------------------------------------
annotation_cols = [
    "UID",
    "PatientID",
    "PhysicianID",
    "Age",
    "Sex",
    "StudyDate",
    "HeartSize",
    "PulmonaryCongestion",
    "PleuralEffusion_Right",
    "PleuralEffusion_Left",
    "PulmonaryOpacities_Right",
    "PulmonaryOpacities_Left",
    "Atelectasis_Right",
    "Atelectasis_Left",
]

annotation_cols = [
    c for c in annotation_cols
    if c in df.columns
]

df[annotation_cols].to_csv(
    ANNOTATION_PATH,
    index=False,
)


# ------------------------------------------------------------
# split.csv
# ------------------------------------------------------------
split_cols = ["UID", "Split"]

if "Fold" in df.columns:
    split_cols.append("Fold")

df[split_cols].to_csv(
    SPLIT_PATH,
    index=False,
)


print("\nSaved:")
print(ANNOTATION_PATH)
print(SPLIT_PATH)
print(f"Images: {IMAGE_DIR}")

print("\nDone.")