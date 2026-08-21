# FOROH Handoff Guide

이 문서는 FOROH 프로젝트 인수인계의 기준 문서다.

## 1. 권장 로컬 위치

Windows 기준 repository root:

```text
D:\FOROH
```

최초 clone:

```powershell
Set-Location D:\
git clone --branch reproduce-paper --single-branch https://github.com/louppian/FOROH.git FOROH
Set-Location D:\FOROH
```

## 2. 먼저 읽을 파일

1. `00_README_FIRST.md`
2. `HANDOFF.md`
3. `Experiment/PLAN.md`
4. `Experiment/README.md`
5. `REPOSITORY_STRUCTURE.md`

## 3. 현재 실험 우선순위

현재 FOROH의 첫 목표는 기존 paper table 복원이 아니라 geometry necessity 검증이다.

Phase-1:

```text
Experiment/01_Coordinate_Necessity/
Experiment/02_LevelSet_Necessity/
```

첫 gate의 핵심 모델:

- Matched Euclidean Huber
- Normalized Cosine Regression
- Hyperspherical Point Prototype
- FOROH Level Set

01/02 결과가 해석 가능해진 뒤 `03_Ordinal_Positioning` 이후로 확장한다. 전체 연구 설계는 `Experiment/PLAN.md`를 따른다.

## 4. 기존 paper reproduction

이전 구조의 실험 스크립트는 삭제하지 않고 다음으로 이동했다.

```text
Experiment/Legacy_Paper_Reproduction/
├── 01_Main_Comparison/
├── 02_Backbone_Scaling/
├── 03_Score_Function/
├── 04_Ablation/
└── 05_Recovery/
```

이 디렉터리는 historical reproduction 용도이며 현재 연구 실험 번호가 아니다.

## 5. 되돌리기

2026-08-22 구조 정리 직전 repository 상태는 다음 branch에 그대로 보존되어 있다.

```text
backup/pre-handoff-restructure-20260822
```

문제가 생기면 해당 branch를 checkout해 구조 정리 이전 상태를 확인할 수 있다.

## 6. 데이터 전달

Raw medical data는 GitHub에 포함하지 않는다. 후임자의 로컬에서는 기존 코드 호환을 위해 우선 다음 경로를 유지한다.

```text
D:\FOROH\data\
├── limuc\
├── aptos2019\
└── kneexray\
```

장기적으로는 `REPOSITORY_STRUCTURE.md`에 정의한 `Dataset/registry + manifests + splits` 구조로 migration한다. 단, training code가 안정화되기 전에는 raw-data 경로를 무리하게 이동하지 않는다.

## 7. 기존 checkpoint / JSON

`.pt/.pth/.ckpt`는 Git에 없을 수 있으므로 서버/원본 로컬 폴더에서 별도 전달한다.

가능하면 다음 구조를 그대로 복사한다.

```text
outputs/**/results.json
outputs/**/*.pt
outputs/**/*.pth
outputs/**/*.ckpt
```

JSON과 해당 checkpoint는 같은 폴더 관계를 유지한다.

검증:

```bash
bash Experiment/00_Inventory/run.sh --reevaluate
python Experiment/00_Inventory/summarize.py
```

기존 검증에서는 발견된 historical checkpoint가 sibling JSON과 일관되게 재평가되었다. 다만 historical recipe와 현재 연구 계획의 config가 동일하다는 의미는 아니다.

## 8. 새 산출물 규칙

새 실험은 `Experiment/<NN_name>`과 동일 번호의 `Result/<NN_name>`을 사용한다.

각 run 폴더 권장 파일:

```text
config.yaml
run_manifest.json
metrics.json
checkpoint.pt
predictions.csv
history.json
stdout.log
```

기존 `outputs/`는 historical evidence, 새 `Result/`는 현재 연구 실험 결과로 구분한다.

## 9. 현재 구현 상태

- `Experiment/00_Inventory`: 사용 가능
- `Experiment/01_Coordinate_Necessity`: 설계 완료, 구현 pending
- `Experiment/02_LevelSet_Necessity`: 설계 완료, 구현 pending
- `Experiment/03` 이후: planned
- `Experiment/train.py`: root `3_train.py`를 사용하는 transitional compatibility layer
- root training code는 Phase-1 안정화 전까지 대규모 이동하지 않는다.

## 10. 인수인계 전 체크리스트

- branch가 `reproduce-paper`인지 확인
- `00_README_FIRST.md` 존재 확인
- `Experiment/PLAN.md`와 실제 Experiment 번호가 일치하는지 확인
- raw dataset 별도 전달 확인
- local checkpoint 별도 전달 확인
- `Result/00_Inventory` report 보존 확인
- `outputs/`를 새 결과 폴더로 오인하지 않도록 설명

Windows 점검 스크립트:

```powershell
powershell -ExecutionPolicy Bypass -File .\handoff_windows.ps1
```
