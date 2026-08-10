# models/

여기에 `.pt` 두 개를 넣습니다. `BetrayalRobotFactory`의 기본 탐색 위치입니다.

| 파일 | 출처 | 관측 의미 |
|---|---|---|
| `honest_v13_best.pt` | `honest_v13` 학습 결과 (Drive `result_honest_v13/<run>/`) | **life** |
| `cheater_v1_best.pt` | `cheater_v1` 학습 결과 (Drive `result_cheater_v1/<run>/`) | moisture |
| `honest_v13_best.pt` | v13 zip run1 (사후 태그 각인: obs_semantics=life, 가중치 불변) | **life** |

Colab에서 복사하는 예:

```bash
!cp "/content/drive/MyDrive/result_honest_v13/run1/honest_v13_best.pt"  scheduler_v2/models/
!cp "/content/drive/MyDrive/result_cheater_v1/run1/cheater_v1_best.pt" scheduler_v2/models/
```

## 관측 의미를 반드시 맞출 것

정직 모델의 토마토별 관측 5개 슬롯은 버전마다 **의미가 다릅니다.**

- `honest_v11` / `honest_v12` → 수분 `clip(1 − elapsed/500)`
- `honest_v13` 이후 → 남은 수명 `1 − elapsed/1000`

둘 다 `[0,1]` 실수 5개라 **틀리게 넣어도 예외가 안 나고 성능만 조용히 무너집니다.**
수분은 `elapsed = 500`부터 사망까지 계속 0이라, 정작 구조 판단이 필요한 구간에서
"방금 말랐다"와 "곧 죽는다"가 구별되지 않습니다.

`honest_v13/train.py`부터는 체크포인트에 `obs_semantics` 태그가 박히고
`BetrayalRobotFactory`가 이를 읽어 자동으로 맞춥니다. 태그가 없는 구버전
체크포인트를 쓸 때는 직접 지정하세요:

```python
BetrayalRobotFactory(honest_pt, cheater_pt, honest_obs="moisture")   # v11/v12
```

태그와 지정이 어긋나면 **경고가 아니라 예외**로 중단됩니다.

## 저장소에 커밋할 것인가

각 `.pt`는 약 1.2 MB입니다. `honest_v10_mo_best.pt`가 이미 저장소에 있으므로
전례상 커밋해도 무방하고, Colab에서 `git pull`만으로 재현할 수 있다는 이점이
있습니다. 대신 저장소를 가볍게 두려면 `.gitignore`에 `scheduler_v2/models/*.pt`를
추가하고 Drive에서 복사해 쓰세요. 어느 쪽이든 **두 모델의 출처와 관측 의미는
`describe()` 출력으로 기록해 두는 것을 권장합니다.**
