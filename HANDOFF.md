# FOROH Handoff Guide

이 문서는 FOROH 프로젝트를 후임자에게 넘길 때 사용하는 최상위 인수인계 문서다.

## 1. 권장 로컬 위치

Windows 기준 표준 로컬 경로는 다음으로 통일한다.

```text
D:\FOROH
```

후임자는 repository root가 반드시 위 경로가 되도록 clone한다.

## 2. 최초 다운로드

PowerShell에서 다음을 실행한다.

```powershell
Set-Location D:\
git clone --branch reproduce-paper --single-branch https://github.com/louppian/FOROH.git FOROH
Set-Location D:\FOROH
```

이미 `D:\FOROH`가 존재하면 clone하지 말고 다음을 사용한다.

```powershell
Set-Location D:\FOROH
git fetch origin
git switch reproduce-paper
git pull origin reproduce-paper
```

또는 repository에 포함된 스크립트를 사용할 수 있다.

```powershell
powershell -ExecutionPolicy Bypass -File .\handoff_windows.ps1
```

## 3. 후임자가 먼저 읽을 파일

다음 순서로 읽는다.

1. `HANDOFF.md` — 전체 인수인계 시작점
2. `REPOSITORY_STRUCTURE.md` — 데이터/모델/실험/산출물 구조
3. `Experiment/PLAN.md` — 연구 질문과 실험 우선순위
4. `Experiment/00_Inventory/` — 기존 JSON/체크포인트 검증 도구

## 4. 현재 연구 우선순위

현재 FOROH에서 가장 먼저 검증할 것은 leaderboard 성능이 아니라 geometry 자체의 필요성이다.

Phase-1 gate는 다음 네 모델로 제한한다.

1. Matched Euclidean Huber
2. Normalized Cosine Regression
3. Hyperspherical Point Prototype
4. FOROH Level Set

우선 대표 조건인 LIMUC / ResNet-50 / 동일 fold에서 비교한다.

Phase-1이 통과된 뒤에만 GOL/CORAL/CORN, multi-fold, small-n, imbalance, score transfer로 확장한다.

세부 계획은 `Experiment/PLAN.md`를 따른다.

## 5. 데이터 인수인계

Raw medical image는 Git repository에 포함하지 않는다.

후임자에게 별도로 전달해야 하는 로컬 데이터는 다음 위치로 맞춘다.

```text
D:\FOROH\data\
├── limuc\
├── aptos2019\
└── kneexray\
```

향후 표준 구조로 migration하면 raw data와 Git-tracked metadata를 분리한다.

```text
Dataset/
├── registry.json
├── manifests/
└── splits/
```

중요: dataset manifest와 split 파일은 Git에서 관리하고 raw image만 Git 밖에 둔다.

## 6. 기존 checkpoint / JSON

과거 실험에서 `.pt/.pth`는 `.gitignore` 때문에 GitHub에 없을 수 있다.

따라서 기존 서버/원본 로컬 폴더의 다음 파일은 별도 외장디스크/공유스토리지로 함께 넘겨야 한다.

```text
outputs/**/results.json
outputs/**/*.pt
outputs/**/*.pth
outputs/**/*.ckpt
```

가능하면 JSON과 해당 checkpoint를 같은 폴더 구조 그대로 복사한다.

기존 검증 결과에서는 발견된 checkpoint가 JSON과 일관되게 재평가되었다. 새 환경에서 다시 확인하려면 다음을 실행한다.

```bash
bash Experiment/00_Inventory/run.sh --reevaluate
python Experiment/00_Inventory/summarize.py
```

## 7. 산출물 규칙

새 실험은 `Result/<experiment-number>/...` 아래 한 run 폴더에 다음을 같이 저장한다.

```text
config.yaml
run_manifest.json
metrics.json
checkpoint.pt
predictions.csv
history.json
stdout.log
```

논문용 figure/table은 raw result와 분리해서 `Artifact/` 아래 생성한다.

## 8. 프로젝트를 넘기기 전 체크리스트

- `D:\FOROH`에서 repository가 정상 open되는지 확인
- `git status`가 의도한 상태인지 확인
- branch가 `reproduce-paper`인지 확인
- `Experiment/PLAN.md` 존재 확인
- `REPOSITORY_STRUCTURE.md` 존재 확인
- raw dataset 별도 전달 여부 확인
- `.pt/.pth` checkpoint 별도 전달 여부 확인
- `Result/00_Inventory` report 보존 여부 확인
- 후임자에게 첫 실행 명령 전달

첫 실행 권장 명령:

```bash
bash Experiment/00_Inventory/run.sh --reevaluate
```

## 9. 향후 정리 원칙

기존 `outputs/`는 검증된 historical evidence이므로 바로 삭제하지 않는다.

새 표준 실험은 `Experiment/01_...`, `Result/01_...` 구조를 사용하고, 기존 `outputs/`는 migration 완료 전까지 legacy로 취급한다.
