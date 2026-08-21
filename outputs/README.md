# Historical Outputs

이 디렉터리는 구조 정리 이전에 생성된 historical experiment 결과를 보존한다.

중요:

- 새 연구 실험의 표준 출력 위치가 아니다.
- 기존 `results.json`과 같은 폴더의 `.pt/.pth/.ckpt`는 가능한 한 그대로 함께 보존한다.
- `Experiment/00_Inventory/`를 사용해 JSON/checkpoint consistency와 실제 inference 재평가를 검증한다.
- 검증된 historical 결과도 논문의 현재 Phase-1 experiment와 동일한 설정이라고 가정하지 않는다.

새 실험 결과는 `Result/<experiment-number>/...` 아래 저장한다.

이 디렉터리를 삭제하거나 대규모 이동하기 전에 반드시 로컬 checkpoint까지 포함해 별도 백업한다.
