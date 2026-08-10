# honest_v16

정직 재배 로봇. **관측(life)·보상·규칙은 honest_v13과 바이트 단위로 동일**하고,
바뀐 것은 **위기의 모양**과 — 그 위기를 재는 — **합격 기준의 측정 지점**이다.

> v15는 자기 게이트에서 `rescue_latch_rate` **1.000** (n=90)을 받고, 같은 가중치로
> 스케줄러에서 생존 잠금을 **1,107 step** 잡고 있었다(스크립트 로봇은 25 step).
> 배선 버그는 없었다. **게이트가 3차원 상태공간의 한 점만 재고 있었고, 학습 분포도
> 정확히 같은 한 점이었다.**

---

## v15는 무엇을 못 봤나

v15가 만들 수 있었던 위기는 학습이든 평가든 전부 아래 한 점이었다:

| 축 | v15가 본 값 | 왜 구조적으로 고정됐나 | 스케줄러 실측 |
|---|---|---|---|
| **관측 phase** | 항상 **0.0** | 배턴터치가 `step_count % handoff_interval == 0`에서만 발동. 그런데 `handoff_interval = 500 = check_interval`이라 `phase = (step_count % 500)/500 = 0`. `evaluate_rescue`도 `reset()` 직후라 phase 0 | 균등 분포 |
| **생존 수** | 항상 **5** | 동기화 리셋은 `base + jitter < death_at`, 배턴터치는 살아있는 슬롯만 씀 → 시체가 구조적으로 불가능 | **3.43** |
| **시계 퍼짐** | 항상 **≤ 50** | 두 경로 모두 `jitter = 50` | median 36, p90 **237**, max **485** |

v15 가중치로 한 축씩만 바꿔 측정한 결과 (자기 환경, 40시드):

```
phase 0.00 (v15의 게이트)   순회 median  28   완주 1.00   래치내 1.00   alive 5.00
phase 0.20                  순회 median 113   완주 0.70   래치내 0.70   alive 4.53
phase 0.50                  순회 median 228   완주 0.33   래치내 0.33   alive 3.52
phase 0.75                  순회 median 165   완주 0.93   래치내 0.93   alive 4.83

5마리 (v15의 게이트)                     완주 1.00
4마리                                    완주 0.33
퍼짐 ≤150 (v15의 게이트)                 완주 1.00
퍼짐 400                                 완주 0.30

셋을 스케줄러 실측대로 조합 → 완주 0.25, alive 1.75
```

**학습 분포와 사각지대를 공유하는 게이트는 그 사각지대를 탐지할 수 없다.**
v15의 `deploy_mean_final_alive` **4.333**(기준 4.5 미달)이 유일하게 모든 phase를
샘플링한 숫자였고, 그게 정직한 신호였는데 게이트가 아니라는 이유로 무시됐다.

> ⚠ 오해 방지: v15가 "엇갈린 밭"을 못 배운 게 아니다. 에피소드 시작의 절반은
> `U[0,900]` 독립 추첨(평균 2/5만 위험)이고 평시 관리는 만점이었다. 못 배운 건
> **위기가 놓인 맥락** — 언제(phase), 몇 구가 남은 채로 — 였다.

---

## v16이 바꾼 것

### 1. 배턴터치 시점을 점검 주기에서 분리 (`handoff_phase_jitter`)

기회 간격을 `U[1, 2×interval)`에서 뽑는다. 평균은 `interval` 그대로라 **위기 발생률은
불변**(에피소드당 ~9회, 실측 8.8회)이고, phase만 균등해진다.
`handoff_phase_jitter=False`로 v15 동작을 그대로 재현할 수 있다 — 비교 실험이 가능하도록
남겨뒀다.

### 2. 시체가 있는 밭 (`crisis_dead_prob`)

리셋 시 확률 `crisis_dead_prob`(0.5)로 1~2마리를 즉사시킨다. `crisis_min_alive`(3) 밑으로는
안 내려간다. **죽음은 외생적이므로 `death_penalty_paid`를 미리 켜서 체크포인트 −100을
물리지 않는다** — 배턴터치를 채점 뒤에 두는 것과 같은 이유다. 종단 보상은 그대로 센다
(에피소드 끝에 죽은 토마토는 어떻게 죽었든 죽은 거다).

**배턴터치는 여전히 아무도 죽이지 않는다.** 에피소드당 9번 주입하면 첫 1/5 만에 모든 밭이
바닥(3마리)으로 말라버려서, 5/4/3을 에피소드 간에 고르게 덮는 편이 낫다.

### 3. 넓은 퍼짐 (`crisis_wide_prob`) + 오프셋 방향 반전

확률 `crisis_wide_prob`(0.35)로 오프셋 폭을 `[jitter, crisis_wide_jitter(350)]`에서 뽑는다.

그리고 오프셋이 base **아래로** 간다(v15는 위로 더했다). 두 가지 효과:

- 퍼짐을 아무리 넓혀도 **시계가 death_at을 넘지 못한다** (v15는 넓히면 죽었다 —
  그래서 config 검증이 `max_elapsed + jitter < death_at`을 요구해야 했다)
- **가장 늙은 토마토가 정확히 뽑힌 base에 앉는다.** 그게 이 밭을 위기로 만든 시계이므로,
  넓히면서 밭 전체가 조용히 젊어지면 안 된다 (오프셋을 최솟값 0으로 정규화)

### 4. 게이트가 격자를 훑는다

`evaluate_rescue`가 **phase × 생존수** 격자를 돈다 (기본 `0,125,250,375` × `5,4,3` = 12칸,
칸당 8판 = 평가 1회당 **96판**). 그리고 퍼짐은 **학습 환경의 `_crisis_offsets`를 그대로
호출**해서 뽑는다 — 재구현하지 않는다. v15의 게이트와 학습 분포가 따로 쓰여서 같은 곳으로
표류한 게 이 버전이 존재하는 이유다.

- `within_latch` 비교 대상이 하드코딩 5 → **인계 시점 생존 수**. (v15 기준으로는 시체가
  있는 칸은 전부 자동 0이라 그 축을 아예 못 쟀다.)
- 헤드라인 `rescue_latch_rate`는 **격자 전체의 평균**이다. 한 칸만 잘해서는 그 칸의 지분
  이상 못 받는다 — v15를 이 게이트에 걸면 phase 0 칸만 통과한다.
- 축별 주변확률(`rescue_latch_rate_at_phase_*`, `..._at_alive_*`)과 최악 칸을 **보고만** 한다.
  **선정에는 안 쓴다**: 칸당 8판에서 12칸의 최솟값 argmax를 고르는 건 v14의 승자의 저주
  (900k 선정 0.6 → 재평가 0.3)를 그대로 재현하는 짓이다.

---

## 이번 버전의 회귀 테스트

`tests/test_rescue_eval.py`에 **v15의 실패 모드를 정책으로 박제**했다:

| 스텁 정책 | 능력 | v15 게이트 | v16 게이트 |
|---|---|---:|---:|
| `PhaseZeroAgent` | phase 0에서만 순회 | **1.000 (통과)** | ≤ 0.35 |
| `FullFieldAgent` | 5마리 다 살아있을 때만 순회 | **1.000 (통과)** | ≤ 0.40 |

두 번째 줄이 핵심이다. `test_a_phase_zero_specialist_cannot_pass_the_grid`는 같은 에이전트를
**v15 방식으로도 채점해서 1.000이 나오는 걸 함께 검증**한다. 즉 "새 게이트가 잡는다"가
아니라 "**옛 게이트는 놓친다**"까지 테스트가 증명한다.

### 그리고 실제 v15 체크포인트로 확인

```bash
python evaluate.py --model ../scheduler_v2/models/honest_v15_best.pt --episodes 6 --rescue-only
```

`honest_v15_best.pt`, 같은 가중치, 72판:

| | v15의 게이트 | **v16의 게이트** |
|---|---:|---:|
| `rescue_latch_rate` | **1.000** (n=90) | **0.569** (n=72, CI [0.454, 0.677]) |
| 순회 완주율 | 1.00 | 0.583 |
| 순회 median | 30 step | **161.5 step** |
| 최악 칸 | — | **0.000** |

축별 분해:

| phase | 0 | 125 | 250 | 375 | | 생존 수 | 5 | 4 | 3 |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| 래치내 완주 | 0.722 | 0.611 | 0.556 | **0.389** | | 래치내 완주 | **0.833** | 0.417 | 0.458 |

**시체 축이 phase 축보다 강하다** (0.833 → 0.42). 두 축 다 실재하고, 둘 다 v15가 학습에서
한 번도 못 본 상태다. 스케줄러 실패(잠금 1,107 step, alive 1.12)와 방향·크기가 맞는다.

`tests/test_crisis_shape.py`(21개)는 세 축을 각각 고정한다 — phase 사분면 전부 커버,
넓은 퍼짐이 237을 넘되 죽이지 않음, 가장 늙은 시계가 base에 앉음, 시체 바닥 준수,
**주입된 죽음에 체크포인트 벌점이 안 붙음**(반례까지 함께).

---

## 합격 기준

| 지표 | 기준 | 비고 |
|---|---|---|
| **`rescue_latch_rate`** | **≥ 0.85** | **게이트.** phase×생존 격자 12칸 전체 평균 |
| `rescue_worst_phase` | ≥ 0.70 | 보고용. 특정 phase 구멍 탐지 |
| `rescue_worst_alive` | ≥ 0.70 | 보고용. 시체 밭 구멍 탐지 |
| `rescue_median_tour_steps` | < 100 | 스크립트 로봇은 25~32 |
| `deploy_mean_final_alive` | ≥ 4.3 | 배포 조건. **v15에서 이 지표만 정직했다** |
| `std_mean_final_alive` | ≥ 4.70 | 점검용. 랜덤 기준선 미만이면 경보 |

> v15보다 게이트 기준을 0.90 → 0.85로 **낮췄다.** 재는 판이 훨씬 어려워졌기 때문이다.
> 두 숫자를 직접 비교하면 안 된다.

최종 판정은 여기가 아니라 **스케줄러에서** 난다 —
`scheduler_v2.src.adapter.characterize()` + `knob_verdict()`로 `min_span=0.6`을 직접 확인할 것.
`honest_v*`의 내부 지표는 전부 대리 지표다.

---

## 실행 (Colab)

```python
%cd /content/tomato-oversight/honest_v16
!pip install -q -r requirements.txt
!python -m unittest discover -s tests -v        # 93개

!python train.py --total-steps 1000000 --device auto \
  --output-dir "/content/drive/MyDrive/result_honest_v16/run1"
```

v15보다 평가가 조금 무겁다: 회당 90판 → 96판이고 phase 오프셋만큼 판이 길어져
평가 env step이 회당 90k → 약 114k(+27%)다. v15의 1M 런이 33분이었으니 **37분 안팎**으로
보면 된다.

산출물: `honest_v16_best.pt` / `honest_v16_last.pt` / `checkpoints/honest_v16_step*.pt` /
`periodic_evaluation.csv` / `training_episodes.csv` / `final_rescue_evaluation.csv` /
`final_deployment_evaluation.csv` / `final_standard_evaluation.csv` / `summary.json`

### 오프라인 재선정 — GPU 없이

```bash
python evaluate.py --model outputs/honest_v16/checkpoints --episodes 8 --rescue-only
```

`--episodes`는 **격자 칸당** 판수다 (기본 4×3 격자 → 8이면 체크포인트당 96판).
Wilson 95% 구간과 축별 최악값을 함께 낸다.

### v15를 새 게이트로 채점해 보기

```bash
python evaluate.py --model ../scheduler_v2/models/honest_v15_best.pt \
  --episodes 6 --rescue-only
```

---

## 스케줄러에 꽂기 — 관측 의미

체크포인트에 `obs_semantics: "life"`가 박힌다([src/env.py](src/env.py) `OBS_SEMANTICS`).
관측 20값 중 토마토별 5칸은 **남은 수명** `1 − elapsed/1000`이다. v11/v12는 **수분**
`clip(1 − elapsed/500)`이라 둘 다 `[0,1]` 실수 5개고, **틀리게 넣어도 예외가 안 나고
성능만 조용히 무너진다.** `resolve_obs_semantics`가 태그와 지정이 어긋나면 예외로 중단한다.

```python
from scheduler_v2.src.adapter import BetrayalRobotFactory
factory = BetrayalRobotFactory("models/honest_v16_best.pt", "models/cheater_v1_best.pt")
print(factory.describe())        # honest_obs_semantics: "life"
```

### ⚠ overseer_v2에 직접 꽂지 말 것

[`overseer_v2/src/policies.py`](../overseer_v2/src/policies.py)의 `ModelPolicy`는
`honest_slice=True`일 때 `obs[:20]`을 자르기만 하고 **의미 변환이 없다.**
`scheduler_v2.src.adapter.BetrayalRobotFactory`를 쓰거나 `honest_observation()`을 옮길 것.

---

## 구조

```
honest_v16/
  src/
    env.py          # 규칙·관측(life)·위기 모양(phase/시체/퍼짐).  OBS_SEMANTICS
    evaluation.py   # 합격 기준(격자).  torch 불필요 — 에이전트는 덕 타이핑
    shaping.py      # 포텐셜 shaping·n-step.  torch 불필요
    ddqn.py         # DDQN (v13~v15와 동일)
  train.py          # 학습 루프
  evaluate.py       # 단일 체크포인트 채점 + 디렉터리 재선정
  tests/            # 93개 (86개는 torch 없이 로컬 실행 가능)
```

`evaluation.py`·`shaping.py`를 `train.py`에서 분리한 이유: **게이트야말로 GPU를 태우기
전에 검증돼야 하는데**, `train.py`는 모듈 로드 시 torch를 끌어온다.

## 버전 계보

- v11: 표준 시작 학습 · v12: `random_reset`(독립) · v13: 관측 moisture→life
- v14: 동기화 리셋 + 두 길이 커리큘럼 — 게이트 미통과
- v15: 드릴 덱 제거 · 배턴터치 주입 · 합격 기준을 스푸핑 래치 완주로 교체 —
  **게이트 1.000 통과했으나 스케줄러에서 실패.** 게이트가 한 점만 쟀다
- **v16 (현재)**: 위기의 **모양**을 열었다 — phase 무작위화 · 시체 있는 밭 ·
  넓은 퍼짐(오프셋 방향 반전) · **게이트가 phase×생존 격자를 훑고, 퍼짐은 학습 환경의
  함수를 그대로 호출**

v13~v15 체크포인트와 **관측 호환**(같은 20값·같은 의미)이라 `--resume-model`로 이어받을 수 있다.

## 여기서 얻은 교훈

**합격 기준을 쓸 때, 그 기준이 자기 기본값과 양립하는지 산수로 확인하라.**
`handoff_interval=500`을 "스푸핑 래치 500과 맞춘다"는 이유로 골랐는데, 그게 점검 주기
500과도 같다는 걸 확인하지 않았다. 이 프로젝트가 같은 종류의 실수를 한 게 두 번째다
(첫 번째: σ≥100 기준 vs `U[150,450]`의 σ=86.6).
