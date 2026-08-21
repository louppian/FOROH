# FOROH Experiments

이 디렉터리의 연구 기준 문서는 `PLAN.md`, 실제 구현/실행 진행도 기준 문서는 `STATUS.md`다.

## Current order

| No. | Directory | Question |
|---:|---|---|
| 00 | `00_Inventory` | 기존 JSON/checkpoint가 실제로 재현되는가? |
| 01 | `01_Coordinate_Necessity` | angular coordinate가 scalar/cosine control보다 필요한가? |
| 02 | `02_LevelSet_Necessity` | point prototype보다 level-set이 필요한가? |
| 03 | `03_Ordinal_Positioning` | CE/Huber/CORAL/CORN/GOL 대비 어디에 위치하는가? |
| 04 | `04_Small_N` | small-data에서 shared coordinate가 더 안정적인가? |
| 05 | `05_Imbalance` | sparse endpoint supervision에 강한가? |
| 06 | `06_Equal_Spacing` | equal-angle assumption의 적용 범위는 어디까지인가? |
| 07 | `07_Single_Axis` | single-axis progression assumption은 언제 성립하는가? |
| 08 | `08_Score_Transfer` | grading system이 바뀌어도 coordinate가 transfer되는가? |
| 09 | `09_Residual_Probing` | free azimuth residual에 어떤 정보가 남는가? |
| 10 | `10_Temporal` | temporal extension은 가능한가? |

## Immediate gate

먼저 다음 네 모델만 동일 LIMUC / ResNet50 / fold 조건에서 비교한다.

1. Matched Euclidean Huber
2. Normalized Cosine Regression
3. Hyperspherical Point Prototype
4. FOROH Level Set

01/02 gate가 확인되기 전에는 03~10에 큰 계산 자원을 사용하지 않는다.

현재 01/02의 core training implementation은 준비됐지만 아직 수정 후 로컬 실행 검증 전이다. 정확한 상태는 `STATUS.md`를 본다.

## Before any GPU run

```bash
python Experiment/common/preflight.py
```

`PHASE-1 PREFLIGHT: PASS`가 확인되어야 한다. 이 검사는 LIMUC patient mapping, train/validation patient leakage, Phase-1 head parameter count 동등성을 확인한다.

## Current training engine

새 연구 실험은 config-driven engine을 사용한다.

```text
Experiment/common/train.py
Model/
Dataset/
Experiment/<NN_...>/configs/
```

예:

```bash
bash Experiment/01_Coordinate_Necessity/run.sh
bash Experiment/02_LevelSet_Necessity/run.sh
```

`Experiment/train.py`와 root `3_train.py`는 historical checkpoint/paper reproduction compatibility용이다. 새 scientific claim용 실험에 사용하지 않는다.

## Historical paper reproduction

이전 paper-table reproduction 스크립트는 다음에 보존한다.

```text
Experiment/Legacy_Paper_Reproduction/
```

구조 정리 이전 전체 repository 상태는 다음 branch에 보존되어 있다.

```text
backup/pre-handoff-restructure-20260822
```

## Existing result verification

원본 로컬 `outputs/`의 JSON/checkpoint를 다시 검증하려면:

```bash
bash Experiment/00_Inventory/run.sh --reevaluate
python Experiment/00_Inventory/summarize.py
```

기존 checkpoint는 historical evidence로 보존한다. 새 연구 결과는 `Result/<same experiment number>/...` 아래 저장한다.
