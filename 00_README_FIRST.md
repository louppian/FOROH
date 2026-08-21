# FOROH — READ THIS FIRST

후임자는 아래 순서로 확인한다.

1. `HANDOFF.md` — 인수인계와 로컬 데이터/checkpoint 전달 방법
2. `Experiment/PLAN.md` — 현재 연구 질문과 실험 우선순위
3. `Experiment/README.md` — 실제 실험 번호와 실행 상태
4. `REPOSITORY_STRUCTURE.md` — 장기적인 데이터/모델/산출물 표준 구조
5. `Experiment/00_Inventory/` — 기존 JSON/checkpoint 검증 도구

## 현재 가장 중요한 실험

Phase-1 gate:

- `Experiment/01_Coordinate_Necessity/`
- `Experiment/02_LevelSet_Necessity/`

01/02가 해석 가능해진 뒤에만 03 이후로 확장한다.

## 과거 실험

과거 paper-table reproduction 스크립트는 다음에 보존되어 있다.

```text
Experiment/Legacy_Paper_Reproduction/
```

기존 결과와 checkpoint는 `outputs/`를 historical evidence로 취급한다.

## 되돌리기

2026-08-22 구조 정리 직전 상태는 다음 branch에 그대로 보존되어 있다.

```text
backup/pre-handoff-restructure-20260822
```

현재 작업 branch는 `reproduce-paper`다.
