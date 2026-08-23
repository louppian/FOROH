    """
    LOR Experiment - Dataset Download Script
    ==========================================
    실행 전 필요사항:
    pip install medmnist kaggle gdown zenodo_get opendatasets

    사용법:
    python download_datasets.py --all          # 전부 다운로드
    python download_datasets.py --medmnist     # MedMNIST만
    python download_datasets.py --aptos        # APTOS만
    python download_datasets.py --limuc        # LIMUC만
    python download_datasets.py --messidor     # Messidor-2만
    python download_datasets.py --adience      # Adience만
    """

    import os
    import argparse
    import json
    from pathlib import Path

    ROOT = Path("./data")


    # ──────────────────────────────────────────────
    # 1. MedMNIST (RetinaMNIST) — pip installable
    # ──────────────────────────────────────────────
    def download_medmnist():
        """
        RetinaMNIST: 1,600 fundus images, 5 ordinal DR grades (0-4)
        Source: MedMNIST v2 (Scientific Data, 2023)
        License: CC BY 4.0
        """
        save_dir = ROOT / "retinamnist"
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            import medmnist
            from medmnist import RetinaMNIST

            # 28x28 version
            for split in ["train", "val", "test"]:
                dataset = RetinaMNIST(split=split, download=True, root=str(save_dir), size=28)
                print(f"  RetinaMNIST-28 {split}: {len(dataset)} samples")

            # 224x224 version (for real experiments)
            for split in ["train", "val", "test"]:
                dataset = RetinaMNIST(split=split, download=True, root=str(save_dir), size=224)
                print(f"  RetinaMNIST-224 {split}: {len(dataset)} samples")

            print(f"[OK] RetinaMNIST saved to {save_dir}")

        except ImportError:
            print("[ERROR] pip install medmnist 필요")
        except Exception as e:
            print(f"[ERROR] RetinaMNIST download failed: {e}")


    # ──────────────────────────────────────────────
    # 2. APTOS 2019 — Kaggle API 필요
    # ──────────────────────────────────────────────
    def download_aptos():
        """
        APTOS 2019 Blindness Detection
        3,662 fundus images, 5 ordinal DR grades (0-4)
        Distribution: G0=1805, G1=370, G2=999, G3=193, G4=295
        Source: Kaggle competition (Aravind Eye Hospital)

        사전 준비:
        1. https://www.kaggle.com/settings → API → Create New Token
        2. ~/.kaggle/kaggle.json 에 저장
        3. chmod 600 ~/.kaggle/kaggle.json
        """
        save_dir = ROOT / "aptos2019"
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            os.system(
                f"kaggle competitions download "
                f"-c aptos2019-blindness-detection "
                f"-p {save_dir}"
            )

            # unzip
            zip_file = save_dir / "aptos2019-blindness-detection.zip"
            if zip_file.exists():
                os.system(f"unzip -o -q {zip_file} -d {save_dir}")
                zip_file.unlink()
                print(f"[OK] APTOS 2019 saved to {save_dir}")
            else:
                print("[WARN] zip not found. Kaggle API 설정 확인:")
                print("  1. pip install kaggle")
                print("  2. https://www.kaggle.com/settings → API → Create New Token")
                print("  3. ~/.kaggle/kaggle.json 에 저장")

        except Exception as e:
            print(f"[ERROR] APTOS download failed: {e}")
            _print_manual_aptos()


    def _print_manual_aptos():
        print("\n  [Manual Download]")
        print("  URL: https://www.kaggle.com/c/aptos2019-blindness-detection/data")
        print("  → Download All → data/ 폴더에 train_images/, test_images/, train.csv 배치")


    # ──────────────────────────────────────────────
    # 3. LIMUC — Zenodo direct download
    # ──────────────────────────────────────────────
    def download_limuc():
        """
        LIMUC (Labeled Images for Ulcerative Colitis)
        11,276 endoscopy images, 4 ordinal Mayo scores (0-3)
        Distribution: MES0=6105(54%), MES1=3052(28%), MES2=1254(11%), MES3=865(8%)
        Source: Zenodo (DOI: 10.5281/zenodo.5827695)
        License: CC BY 4.0
        """
        save_dir = ROOT / "limuc"
        save_dir.mkdir(parents=True, exist_ok=True)

        zenodo_url = "https://zenodo.org/records/5827695/files"
        files = [
            "LIMUC_dataset.zip",
        ]

        try:
            import subprocess

            # Method 1: zenodo_get
            try:
                subprocess.run(
                    ["zenodo_get", "10.5281/zenodo.5827695", "-o", str(save_dir)],
                    check=True,
                )
                print(f"[OK] LIMUC saved to {save_dir}")
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Method 2: wget
            for f in files:
                url = f"{zenodo_url}/{f}?download=1"
                subprocess.run(
                    ["wget", "-q", "--show-progress", "-O", str(save_dir / f), url],
                    check=True,
                )

            # unzip
            zip_file = save_dir / "LIMUC_dataset.zip"
            if zip_file.exists():
                os.system(f"unzip -o -q {zip_file} -d {save_dir}")
                zip_file.unlink()

            print(f"[OK] LIMUC saved to {save_dir}")

        except Exception as e:
            print(f"[ERROR] LIMUC download failed: {e}")
            _print_manual_limuc()


    def _print_manual_limuc():
        print("\n  [Manual Download]")
        print("  URL: https://zenodo.org/records/5827695")
        print("  → LIMUC_dataset.zip 다운로드")
        print("  → data/limuc/ 에 압축 해제")


    # ──────────────────────────────────────────────
    # 4. Messidor-2 — Cross-dataset transfer 실험용
    # ──────────────────────────────────────────────
    def download_messidor():
        """
        Messidor-2: 1,748 fundus images, DR grades 0-4
        Source: ADCIS website
        License: Research use
        """
        save_dir = ROOT / "messidor2"
        save_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Messidor-2는 수동 다운로드가 필요합니다:")
        print("  1. https://www.adcis.net/en/third-party/messidor2/ 접속")
        print("  2. 이용약관 동의 후 다운로드")
        print(f"  3. {save_dir}/ 에 압축 해제")
        print("  4. Labels: https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades")


    # ──────────────────────────────────────────────
    # 5. Adience — Non-medical ordinal benchmark
    # ──────────────────────────────────────────────
    def download_adience():
        """
        Adience: ~26,580 face images, 8 age groups (ordinal)
        Groups: 0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60+
        Source: https://talhassner.github.io/home/projects/Adience/Adience-data.html
        """
        save_dir = ROOT / "adience"
        save_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Adience는 수동 다운로드가 필요합니다:")
        print("  1. https://talhassner.github.io/home/projects/Adience/Adience-data.html 접속")
        print("  2. aligned.tar.gz 다운로드")
        print(f"  3. {save_dir}/ 에 압축 해제")
        print("  4. fold_*.txt 파일도 같은 위치에 배치")


    # ──────────────────────────────────────────────
    # Dataset verification & statistics
    # ──────────────────────────────────────────────
    def verify_datasets():
        """다운로드된 데이터셋 상태 확인 및 통계 출력"""
        print("\n" + "=" * 60)
        print("Dataset Verification")
        print("=" * 60)

        datasets = {
            "RetinaMNIST": ROOT / "retinamnist",
            "APTOS 2019": ROOT / "aptos2019",
            "LIMUC": ROOT / "limuc",
            "Messidor-2": ROOT / "messidor2",
            "Adience": ROOT / "adience",
        }

        for name, path in datasets.items():
            if path.exists():
                # count files
                n_files = sum(1 for _ in path.rglob("*") if _.is_file())
                size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
                print(f"  [OK] {name:15s} | {n_files:6d} files | {size_mb:8.1f} MB | {path}")
            else:
                print(f"  [--] {name:15s} | not downloaded")


    # ──────────────────────────────────────────────
    # Dataset loading utilities
    # ──────────────────────────────────────────────
    def create_dataset_loaders():
        """
        각 데이터셋의 PyTorch Dataset class 생성 유틸리티.
        다운로드 후 실행하여 data/dataset_configs.json 생성.
        """
        configs = {}

        # APTOS
        aptos_csv = ROOT / "aptos2019" / "train.csv"
        if aptos_csv.exists():
            import pandas as pd

            df = pd.read_csv(aptos_csv)
            dist = df["diagnosis"].value_counts().sort_index().to_dict()
            configs["aptos2019"] = {
                "path": str(ROOT / "aptos2019"),
                "image_dir": "train_images",
                "label_file": "train.csv",
                "id_col": "id_code",
                "label_col": "diagnosis",
                "C_max": 4,
                "num_classes": 5,
                "distribution": dist,
                "total": len(df),
            }
            print(f"  APTOS: {len(df)} images, distribution: {dist}")

        # LIMUC
        limuc_dir = ROOT / "limuc"
        if limuc_dir.exists():
            # LIMUC uses folder structure: Mayo_0/, Mayo_1/, Mayo_2/, Mayo_3/
            dist = {}
            total = 0
            for grade in range(4):
                grade_dir = limuc_dir / f"Mayo_{grade}"
                if not grade_dir.exists():
                    # try alternate structure
                    for subdir in limuc_dir.iterdir():
                        if subdir.is_dir():
                            grade_dir = subdir / f"Mayo_{grade}"
                            if grade_dir.exists():
                                break
                if grade_dir.exists():
                    n = sum(1 for f in grade_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg", ".bmp"})
                    dist[grade] = n
                    total += n

            if total > 0:
                configs["limuc"] = {
                    "path": str(limuc_dir),
                    "C_max": 3,
                    "num_classes": 4,
                    "distribution": dist,
                    "total": total,
                }
                print(f"  LIMUC: {total} images, distribution: {dist}")

        # RetinaMNIST
        retina_dir = ROOT / "retinamnist"
        if retina_dir.exists():
            try:
                import numpy as np

                # MedMNIST stores as .npz
                npz_files = list(retina_dir.rglob("retinamnist*.npz"))
                if npz_files:
                    data = np.load(npz_files[0])
                    if "train_labels" in data:
                        labels = data["train_labels"].flatten()
                        unique, counts = np.unique(labels, return_counts=True)
                        dist = {int(u): int(c) for u, c in zip(unique, counts)}
                        configs["retinamnist"] = {
                            "path": str(retina_dir),
                            "C_max": 4,
                            "num_classes": 5,
                            "distribution": dist,
                            "total": len(labels),
                        }
                        print(f"  RetinaMNIST: {len(labels)} train images, distribution: {dist}")
            except Exception:
                pass

        # Save config
        config_path = ROOT / "dataset_configs.json"
        with open(config_path, "w") as f:
            json.dump(configs, f, indent=2)
        print(f"\n  Config saved to {config_path}")

        return configs


    # ──────────────────────────────────────────────
    # Main
    # ──────────────────────────────────────────────
    def main():
        parser = argparse.ArgumentParser(description="Download LOR experiment datasets")
        parser.add_argument("--all", action="store_true", help="Download all datasets")
        parser.add_argument("--medmnist", action="store_true", help="Download RetinaMNIST")
        parser.add_argument("--aptos", action="store_true", help="Download APTOS 2019")
        parser.add_argument("--limuc", action="store_true", help="Download LIMUC")
        parser.add_argument("--messidor", action="store_true", help="Download Messidor-2")
        parser.add_argument("--adience", action="store_true", help="Download Adience")
        parser.add_argument("--verify", action="store_true", help="Verify downloaded datasets")
        parser.add_argument("--config", action="store_true", help="Generate dataset configs")
        args = parser.parse_args()

        ROOT.mkdir(parents=True, exist_ok=True)

        if args.all or args.medmnist:
            print("\n[1/5] Downloading RetinaMNIST...")
            download_medmnist()

        if args.all or args.aptos:
            print("\n[2/5] Downloading APTOS 2019...")
            download_aptos()

        if args.all or args.limuc:
            print("\n[3/5] Downloading LIMUC...")
            download_limuc()

        if args.all or args.messidor:
            print("\n[4/5] Messidor-2 info...")
            download_messidor()

        if args.all or args.adience:
            print("\n[5/5] Adience info...")
            download_adience()

        if args.verify or args.all:
            verify_datasets()

        if args.config or args.all:
            print("\nGenerating dataset configs...")
            create_dataset_loaders()

        if not any(vars(args).values()):
            parser.print_help()


    if __name__ == "__main__":
        main()