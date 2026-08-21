# FOROH Experiments

연구 기준 문서는 `PLAN.md`, 실제 구현/실행 진행도 기준 문서는 `STATUS.md`다.

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

먼저 동일 LIMUC / ResNet50 / patient-level fold 조건에서 다음 네 모델만 검증한다.

1. Matched Euclidean Huber
2. Normalized Cosine Regression
3. Hyperspherical Point Prototype
4. FOROH Level Set

01/02 결과가 해석 가능하기 전에는 03~10에 큰 계산 자원을 사용하지 않는다.

## Execution style

YAML config layer는 사용하지 않는다. 각 실험 설정은 해당 폴더의 `run.py`에 직접 적는다.

```text
Experiment/01_Coordinate_Necessity/run.py
Experiment/02_LevelSet_Necessity/run.py
        ↓
Experiment/common/train.py
        ↓
Model/ + Dataset/
```

따라서 설정을 확인하려면 별도 config 파일이 아니라 `run.py` 자체를 보면 된다.

## Before GPU run

```bash
python Experiment/common/preflight.py
```

마지막에 `PHASE-1 PREFLIGHT: PASS`가 나와야 한다.

그 다음:

```bash
python Experiment/01_Coordinate_Necessity/run.py
python Experiment/02_LevelSet_Necessity/run.py
```

## Historical paper reproduction

이전 paper-table reproduction 스크립트는 다음에 그대로 보존한다.

```text
Experiment/Legacy_Paper_Reproduction/
```

구조 정리 이전 repository 상태는 다음 branch에 보존되어 있다.

```text
backup/pre-handoff-restructure-20260822
```

`Experiment/train.py`와 root `3_train.py`는 historical checkpoint/paper reproduction compatibility용이다. 새 scientific claim용 실험에 사용하지 않는다.

## Results

기존 `outputs/`는 historical evidence다. 새 연구 실험은 `Result/<experiment>/...`에 저장하며 `config.json`, `run_manifest.json`, `metrics.json`, `history.json`, `predictions.csv`, `checkpoint.pt`를 남긴다.
