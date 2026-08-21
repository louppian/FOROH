# FOROH Experiments

이 디렉터리의 **현재 기준 문서는 `PLAN.md`** 다. 실험 번호는 paper table 순서가 아니라 FOROH의 핵심 주장 검증 순서를 따른다.

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

## Historical paper reproduction

이전의 paper-table reproduction 스크립트는 삭제하지 않고 다음으로 이동했다.

```text
Experiment/Legacy_Paper_Reproduction/
```

해당 스크립트는 과거 결과 복원/검증용이며 현재 실험 번호가 아니다.

구조 정리 이전 전체 repository 상태는 다음 branch에 보존되어 있다.

```text
backup/pre-handoff-restructure-20260822
```

## Existing result verification

새 학습 전에 원본 로컬 `outputs/`의 JSON/checkpoint를 검증하려면:

```bash
bash Experiment/00_Inventory/run.sh --reevaluate
python Experiment/00_Inventory/summarize.py
```

기존 checkpoint는 historical evidence로 보존한다. 새 연구 실험 결과는 향후 `Result/<same experiment number>/...` 아래 저장한다.

## Shared implementation

현재 `Experiment/train.py`는 root `3_train.py`를 이용하는 transitional compatibility layer다. Phase-1 구현이 안정화되기 전에는 root training code를 대규모로 이동하지 않는다.
