# Result Directory

새 연구 실험의 결과는 `Experiment/`와 같은 번호를 사용한다.

예:

```text
Experiment/01_Coordinate_Necessity/
Result/01_Coordinate_Necessity/
```

각 run은 가능하면 다음 파일을 한 폴더에 함께 남긴다.

```text
config.yaml
run_manifest.json
metrics.json
checkpoint.pt
predictions.csv
history.json
stdout.log
```

원칙:

- JSON과 checkpoint를 같은 run 폴더에 둔다.
- split/fold/seed를 경로 또는 manifest에서 식별 가능하게 한다.
- paper figure/table은 이 폴더에 직접 저장하지 않고 향후 `Artifact/`에서 생성한다.
- 기존 `outputs/`는 historical experiment evidence이며 새 `Result/`와 섞지 않는다.

현재 `Result/00_Inventory/`는 로컬에서 inventory/verification script 실행 시 생성된다.
