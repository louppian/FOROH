# FOROH Repository Structure Plan

이 문서는 FOROH를 처음 보는 후임자도 **데이터가 어디에 있고, 어떤 코드가 어떤 실험을 실행하며, 결과와 가중치가 어디에 저장되는지** 경로만 보고 이해할 수 있도록 하는 표준 구조를 정의한다.

핵심 원칙은 다음 네 가지다.

1. **Experiment 번호와 Result 번호를 1:1로 맞춘다.**
2. raw medical images는 Git에 넣지 않지만, **dataset version / manifest / split은 Git으로 추적한다.**
3. 한 experiment run은 항상 **config + metrics + checkpoint + predictions + manifest**를 한 폴더에 남긴다.
4. 기존 `outputs/`는 삭제하지 않고 **legacy evidence**로 보존한 뒤 새 표준으로 점진적으로 이동한다.

---

# 1. Target repository layout

```text
FOROH/
├── .gitignore
├── README.md
├── REPOSITORY_STRUCTURE.md
│
├── Dataset/
│   ├── README.md
│   ├── registry.json
│   ├── manifests/
│   │   ├── limuc.csv
│   │   ├── aptos.csv
│   │   └── kneexray.csv
│   └── splits/
│       ├── limuc/
│       │   └── v1_patient10fold_seed42/
│       │       ├── split_manifest.json
│       │       ├── fold00.json
│       │       ├── fold01.json
│       │       └── ...
│       ├── aptos/
│       └── kneexray/
│
├── Model/
│   ├── common.py
│   ├── foroh.py
│   ├── scalar.py
│   ├── point_prototype.py
│   └── baselines/
│       ├── ce.py
│       ├── coral.py
│       ├── corn.py
│       └── gol.py
│
├── Experiment/
│   ├── README.md
│   ├── PLAN.md
│   ├── common/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── config.py
│   │
│   ├── 00_Inventory/
│   ├── 01_Coordinate_Necessity/
│   ├── 02_LevelSet_Necessity/
│   ├── 03_Ordinal_Positioning/
│   ├── 04_Small_N/
│   ├── 05_Imbalance/
│   ├── 06_Equal_Spacing/
│   ├── 07_Single_Axis/
│   ├── 08_Score_Transfer/
│   ├── 09_Residual_Probing/
│   └── 10_Temporal/
│
├── Result/
│   ├── README.md
│   ├── 00_Inventory/
│   ├── 01_Coordinate_Necessity/
│   ├── 02_LevelSet_Necessity/
│   ├── 03_Ordinal_Positioning/
│   ├── 04_Small_N/
│   ├── 05_Imbalance/
│   ├── 06_Equal_Spacing/
│   ├── 07_Single_Axis/
│   ├── 08_Score_Transfer/
│   ├── 09_Residual_Probing/
│   └── 10_Temporal/
│
├── Artifact/
│   ├── tables/
│   ├── figures/
│   └── paper/
│
└── Legacy/
    └── README.md
```

`Dataset/`, `Experiment/`, `Result/`의 번호와 이름은 가능한 한 서로 대응하도록 유지한다.

---

# 2. Dataset policy

## 2.1 Raw data

raw images / DICOM / NPZ 등 대용량 또는 민감 데이터는 repository에 commit하지 않는다.

실제 서버에서는 예를 들어 다음처럼 존재할 수 있다.

```text
data/
├── limuc/
├── aptos2019/
└── kneexray/
```

이 경로 자체는 machine마다 달라도 된다.

중요한 것은 **코드가 raw path를 hard-code하지 않고 `Dataset/registry.json`을 통해 찾도록 만드는 것**이다.

예시:

```json
{
  "limuc": {
    "root_env": "FOROH_LIMUC_ROOT",
    "default_relative": "data/limuc",
    "split_version": "v1_patient10fold_seed42"
  },
  "aptos": {
    "root_env": "FOROH_APTOS_ROOT",
    "default_relative": "data/aptos2019",
    "split_version": "v1_5fold_seed42"
  }
}
```

우선순위는:

1. environment variable
2. registry의 relative path
3. 없으면 명확한 error

순으로 한다.

---

## 2.2 Dataset manifest

각 dataset은 최소 하나의 manifest를 Git으로 추적한다.

예: `Dataset/manifests/limuc.csv`

```text
sample_id,path_relative,label,patient_id,site,split_source
000001,...,0,P001,...,...
```

필수 field:

- stable sample ID
- relative image path 또는 source ID
- ordinal label
- patient ID가 있으면 patient ID
- dataset/source name

선택 field:

- site
- scanner
- acquisition date
- phenotype annotation
- continuous target

**절대 경로는 manifest에 저장하지 않는다.**

---

## 2.3 Split versioning

재현성에서 가장 중요한 파일 중 하나다.

예:

```text
Dataset/splits/limuc/v1_patient10fold_seed42/
├── split_manifest.json
├── fold00.json
├── fold01.json
├── ...
└── fold09.json
```

`split_manifest.json`에는 최소 다음을 남긴다.

```json
{
  "dataset": "limuc",
  "version": "v1_patient10fold_seed42",
  "strategy": "patient-level stratified 10-fold",
  "seed": 42,
  "num_folds": 10,
  "heldout_test": true
}
```

각 fold 파일에는 **patient/sample ID 목록 자체**를 저장한다.

학습 때마다 `StratifiedKFold`를 다시 계산하는 것보다, 한번 확정한 split을 명시적으로 저장하고 모든 method가 같은 split을 읽도록 한다.

---

# 3. Experiment directory contract

각 실험 섹션은 동일한 내부 구조를 갖는다.

예:

```text
Experiment/01_Coordinate_Necessity/
├── README.md
├── run.sh
├── analyze.py
└── configs/
    ├── E01A_euclidean_huber.yaml
    ├── E01B_normalized_cosine.yaml
    └── E01C_foroh.yaml
```

`README.md`에는 후임자가 반드시 알아야 하는 다음 내용을 적는다.

- Research question
- Hypothesis
- Compared methods
- Fixed controls
- Dataset / split
- Primary metrics
- Success / failure interpretation
- 생성되는 Result 경로

---

# 4. Stable experiment IDs

실험 ID는 폴더 순서와 별도로 **한 번 부여하면 바꾸지 않는다.**

추천 체계:

```text
E01A  Matched Euclidean Huber
E01B  Normalized Cosine
E01C  FOROH

E02A  Point Prototype
E02B  FOROH Level Set

E03A  CE
E03B  Matched Huber
E03C  CORAL
E03D  CORN
E03E  GOL
E03F  FOROH
```

small-n처럼 조건이 반복되는 경우:

```text
E04_FOROH_p100_s42
E04_FOROH_p050_s42
E04_FOROH_p025_s42
E04_FOROH_p010_s42
```

단, 실제 폴더명에 모든 hyperparameter를 넣지는 않는다. 상세 설정은 config 파일과 run manifest에 저장한다.

---

# 5. Config-first execution

새 실험은 긴 shell argument를 직접 적는 방식보다 config 파일을 source of truth로 사용한다.

예:

```yaml
experiment_id: E01C
experiment: coordinate_necessity
method: FOROH

dataset: limuc
split_version: v1_patient10fold_seed42
fold: 0
seed: 42

backbone: resnet50
projector_dim: 128
projector_depth: 2
loss: huber
huber_delta: 0.5

optimizer: adamw
lr_backbone: 1.0e-4
lr_head: 1.0e-3
batch_size: 64
scheduler: cosine
patience: 10
```

실제 실행 command는 가능한 한 단순하게 유지한다.

```bash
python Experiment/common/train.py \
  --config Experiment/01_Coordinate_Necessity/configs/E01C_foroh.yaml
```

---

# 6. Result directory contract

각 run의 결과는 **한 폴더에 완결적으로 저장**한다.

예:

```text
Result/01_Coordinate_Necessity/
└── E01C_FOROH/
    └── limuc_r50/
        └── fold00_seed42/
            ├── config.yaml
            ├── run_manifest.json
            ├── metrics.json
            ├── checkpoint.pt
            ├── predictions.csv
            ├── history.json
            └── stdout.log
```

`checkpoint.pt`는 Git에 commit하지 않아도 된다. 그러나 같은 폴더에 두는 것을 원칙으로 한다.

JSON과 checkpoint가 떨어져 있으면 이후 어떤 weight가 어떤 수치를 만든 것인지 확인하기 어렵기 때문이다.

현재 inventory 검사에서 사용한 **sibling JSON + checkpoint 구조를 공식 규칙으로 승격**한다.

---

# 7. Required output files

## `config.yaml`

실제로 실행된 최종 configuration.

CLI override가 있더라도 최종 resolved config를 저장한다.

## `run_manifest.json`

실험 provenance.

예:

```json
{
  "experiment_id": "E01C",
  "git_commit": "...",
  "git_branch": "reproduce-paper",
  "hostname": "a100-n1",
  "device": "NVIDIA A100",
  "started_at": "...",
  "dataset_split": "v1_patient10fold_seed42",
  "fold": 0,
  "seed": 42
}
```

가능하면 다음도 기록한다.

- Python version
- PyTorch version
- torchvision version
- CUDA version

## `metrics.json`

최종 평가 metric만 저장한다.

```json
{
  "mae": 0.0,
  "qwk": 0.0,
  "acc": 0.0,
  "macro_f1": 0.0,
  "recall_by_grade": {}
}
```

## `predictions.csv`

재평가 없이 metric / figure를 다시 만들 수 있도록 sample-level prediction을 저장한다.

권장 column:

```text
sample_id,label,prediction,score,theta,fold,seed
```

FOROH가 아니면 `theta`는 비워둘 수 있다.

## `history.json`

training / validation curve.

최소:

- epoch
- training loss
- validation MAE
- validation QWK
- LR

## `checkpoint.pt`

같은 run folder에 저장한다.

checkpoint 내부에도 가능하면 다음을 포함한다.

```text
model_state
config
fold
seed
best_epoch
validation_metrics
test_metrics
git_commit
```

---

# 8. Aggregate result structure

fold/seed가 끝난 뒤 aggregate 파일을 run group의 상위 경로에 만든다.

```text
Result/01_Coordinate_Necessity/E01C_FOROH/limuc_r50/
├── fold00_seed42/
├── fold01_seed42/
├── ...
├── fold09_seed42/
├── summary.json
└── summary.csv
```

`summary.json`에는 mean/std뿐 아니라 개별 fold 값을 모두 유지한다.

```json
{
  "n": 10,
  "mae": {"mean": 0.0, "std": 0.0, "values": []},
  "qwk": {"mean": 0.0, "std": 0.0, "values": []}
}
```

---

# 9. Experiment sections and corresponding output folders

## 00 Inventory / Recovery

```text
Experiment/00_Inventory/
Result/00_Inventory/
```

목적:

- historical JSON inventory
- checkpoint integrity validation
- legacy result recovery

이 결과는 paper result가 아니라 provenance / audit용이다.

## 01 Coordinate Necessity

```text
Experiment/01_Coordinate_Necessity/
Result/01_Coordinate_Necessity/
```

Phase-1 Gate A:

- Matched Euclidean Huber
- Normalized Cosine
- FOROH

## 02 Level-Set Necessity

```text
Experiment/02_LevelSet_Necessity/
Result/02_LevelSet_Necessity/
```

Phase-1 Gate B:

- Hyperspherical Point Prototype
- FOROH Level Set

## 03 Ordinal Positioning

```text
Experiment/03_Ordinal_Positioning/
Result/03_Ordinal_Positioning/
```

principal methods:

- CE
- Matched Huber
- CORAL
- CORN
- GOL
- FOROH

## 04 Small-n

```text
Experiment/04_Small_N/
Result/04_Small_N/
```

100 / 50 / 25 / 10% patient-level subsampling × multiple seeds.

## 05 Imbalance

```text
Experiment/05_Imbalance/
Result/05_Imbalance/
```

rare/end-point grade retain ratio experiment.

## 06 Equal Spacing

```text
Experiment/06_Equal_Spacing/
Result/06_Equal_Spacing/
```

Equal / Mildly Unequal / Strongly Unequal.

## 07 Single Axis

```text
Experiment/07_Single_Axis/
Result/07_Single_Axis/
```

synthetic latent dimensionality stress test.

## 08 Score Transfer

```text
Experiment/08_Score_Transfer/
Result/08_Score_Transfer/
```

zero-shot / calibration / few-shot / head-only / full retraining.

## 09 Residual Probing

```text
Experiment/09_Residual_Probing/
Result/09_Residual_Probing/
```

$\theta$, $v$, full $u$ probing.

## 10 Temporal

```text
Experiment/10_Temporal/
Result/10_Temporal/
```

현재 static paper에서는 inactive / future work 상태로 둔다.

---

# 10. Artifact directory

`Result/`는 raw scientific output이고 `Artifact/`는 논문에 직접 들어가는 가공 결과다.

```text
Artifact/
├── tables/
│   ├── table01_coordinate.csv
│   ├── table02_levelset.csv
│   └── ...
├── figures/
│   ├── fig01_coordinate.pdf
│   ├── fig02_levelset.pdf
│   └── ...
└── paper/
    └── result_index.md
```

**figure script가 checkpoint를 직접 읽지 않는 것**을 원칙으로 한다.

가능하면:

```text
checkpoint -> predictions.csv / metrics.json -> Artifact
```

순서로 생성한다.

이렇게 해야 모델 코드가 바뀌어도 기존 figure/table을 재생성할 수 있다.

---

# 11. README hierarchy for handoff

후임자는 다음 순서로 문서를 읽으면 된다.

1. `/README.md` — 프로젝트 한 페이지 설명
2. `/Experiment/PLAN.md` — 왜 이 실험을 하는가
3. `/REPOSITORY_STRUCTURE.md` — 데이터/코드/결과가 어디 있는가
4. `/Dataset/README.md` — 데이터를 서버에서 어떻게 연결하는가
5. `/Experiment/<NN_...>/README.md` — 해당 연구 질문과 실행법
6. `/Result/<NN_...>/README.md` 또는 summary — 현재 완료 상태

각 experiment README의 첫 부분에는 아래 status block을 둔다.

```text
Status: PLANNED | RUNNING | COMPLETE | BLOCKED
Primary owner:
Last verified commit:
Dataset split:
Expected compute:
```

---

# 12. Legacy migration policy

현재 repository의 `outputs/`에는 서로 다른 시점의 recipe가 혼재한다. 기존 checkpoint는 재평가 결과 JSON과 일치하므로 삭제할 이유가 없다.

따라서 즉시 rename/move하지 않는다.

### Stage 1 — Freeze

기존:

```text
outputs/
```

를 historical read-only evidence로 취급한다.

새 experiment는 더 이상 `outputs/`에 저장하지 않는다.

### Stage 2 — Index

`Experiment/00_Inventory/`가 생성한 inventory를 사용해 각 historical result에 다음 label을 붙인다.

- VERIFIED_LEGACY
- CONFIG_MISMATCH
- PAPER_MATCH
- OBSOLETE

### Stage 3 — Optional migration

필요한 historical result만 다음처럼 복사/링크한다.

```text
Legacy/Result/<old-folder>/
```

단, original path를 manifest에 보존한다.

### Stage 4 — Remove old scripts only after replacement

기존:

```text
3_train.py
4_compare.py
5_figure.py
run.sh
run_huber.sh
temp.py
```

는 새 Model/Dataset/Experiment pipeline이 동일 functionality를 검증한 후에만 `Legacy/`로 이동한다.

---

# 13. Immediate implementation order

폴더를 한 번에 모두 옮기면 기존 checkpoint 재현성이 깨질 수 있으므로 아래 순서로 진행한다.

## Step 1 — Documentation / IDs

- `Experiment/PLAN.md`
- `REPOSITORY_STRUCTURE.md`
- experiment IDs 확정

## Step 2 — Dataset reproducibility

- `Dataset/registry.json`
- LIMUC manifest 생성
- LIMUC patient-level fold files 저장

가장 먼저 Phase-1에 필요한 LIMUC만 처리한다.

## Step 3 — Common training engine

기존 `3_train.py`에서 재사용 가능한 부분을 분리한다.

```text
Dataset/
Model/
Experiment/common/
```

기존 behavior와 checkpoint loading을 깨지 않는지 검증한다.

## Step 4 — Phase-1 experiment folders

먼저 두 폴더만 실제 구현한다.

```text
01_Coordinate_Necessity
02_LevelSet_Necessity
```

Phase-1 gate가 통과하기 전에는 03~10에 대규모 GPU 시간을 쓰지 않는다.

## Step 5 — Standard result writer

모든 새 run이 다음을 자동으로 남기게 한다.

```text
config.yaml
run_manifest.json
metrics.json
checkpoint.pt
predictions.csv
history.json
```

## Step 6 — Gate evaluation

LIMUC / ResNet50 / 동일 fold에서:

1. Matched Euclidean Huber
2. Normalized Cosine
3. Point Prototype
4. FOROH

를 수행한다.

결과에 따라 이후 03~09를 진행할지 결정한다.

---

# 14. Rules that should not be broken

1. method 비교에서 split을 각자 다시 만들지 않는다.
2. result JSON과 checkpoint를 다른 폴더에 저장하지 않는다.
3. output folder 이름만 보고 hyperparameter를 추측하게 만들지 않는다.
4. raw images를 Git에 올리지 않는다.
5. paper table 수치를 사람이 직접 복사해 적지 않는다. `Result`에서 생성한다.
6. test result를 보고 model/boundary 선택을 하지 않는다.
7. figure script가 stale checkpoint API에 직접 의존하지 않도록 prediction artifact를 남긴다.
8. legacy result와 새 result를 같은 namespace에 섞지 않는다.
9. 실험 번호는 연구 질문 번호이며 단순 실행 순서 번호가 아니다.
10. 모든 paper claim은 대응하는 experiment ID를 가져야 한다.

---

# 15. Definition of done for one experiment

하나의 실험이 `COMPLETE`가 되려면 최소 다음이 있어야 한다.

- [ ] config committed
- [ ] split version fixed
- [ ] all required folds/seeds finished
- [ ] sibling checkpoint validation passed
- [ ] metrics.json exists
- [ ] predictions.csv exists
- [ ] aggregate summary exists
- [ ] analysis/table generated from Result
- [ ] README에 결과와 해석 업데이트
- [ ] git commit hash recorded

이 기준을 만족하지 못한 실험은 논문 표에 사용하지 않는다.
