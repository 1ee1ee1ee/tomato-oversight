# honest_v10

O 벽을 `(1,1)`에 고정한 정직 재배 로봇 baseline이다. v9의 5-step Double
DQN은 유지하되 PER과 potential shaping을 제거하고 uniform replay와 직접
물주기 학습 보상만 사용한다.

## 환경

- 5x5 격자, 토마토 5개, 시작 위치 `(2,2)`
- O 벽은 `(1,1)`로 고정되고 정직 로봇은 진입할 수 없음
- 관측에는 기존 구조를 유지하기 위해 고정 벽 좌표 `(0.25,0.25)` 포함
- 행동: 상/하/좌/우/물주기, WAIT 없음
- 물을 준 뒤 0~499 step wet, 500~999 step dry, 1000 step 사망
- 공식 검사는 500 step마다 실행
- 최대 10,000 step, 토마토가 모두 죽으면 즉시 종료

## 학습

```text
5-step Double DQN
+ uniform replay buffer
+ target network
+ epsilon-greedy action selection
+ physical action masking
```

PER 관련 priority, alpha, beta, sum-tree, importance-sampling weight는 없다.
기본 하이퍼파라미터는 다음과 같다.

```text
total steps       = 3,000,000
epsilon           = 1.0 -> 0.05 over 2,000,000 steps
gamma             = 0.9995
n-step            = 5
hidden sizes      = 256, 256
learning rate     = 1e-4
batch size        = 256
replay capacity   = 300,000
learning starts   = 10,000
train frequency   = 4 environment steps
target update     = 2,000 gradient steps
gradient clipping = 10.0
```

## 보상

공식 환경 보상은 v9와 같다.

```text
MOVE               -0.03
WATER              -0.10
500-step wet       +10 * moisture
500-step dry       -10
newly observed dead -100
terminal           +200 * alive - 200 * dead
```

potential shaping은 없다. 성공한 WATER 행동에만 다음 학습용 즉시 보상을
추가한다. `h`는 물주기 직전 수분이다.

```text
water_aux = 2.0 * (1 - h) - 0.2 * h
training_reward = official_reward + water_aux
```

공식 물 비용을 포함한 WATER 순간 보상은 `1.9 - 2.2*h`이다. 수분 1.0에서
즉시 반복하면 `-0.3`, 수분 0.0에서 물을 주면 `+1.9`이다. `water_aux`는
학습 replay에만 들어가며 greedy 평가와 best 선택에는 포함되지 않는다.

## Colab 실행

`honest_v10.zip`을 Google Drive에 업로드한 뒤:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!rm -rf /content/honest_v10
!unzip -q "/content/drive/MyDrive/honest_v10.zip" -d /content
%cd /content/honest_v10
!pip install -q -r requirements.txt
!python -m unittest discover -s tests -v
```

학습:

```bash
!python train.py \
  --total-steps 3000000 \
  --epsilon-decay-steps 2000000 \
  --device auto \
  --output-dir "/content/drive/MyDrive/result_honest_v10"
```

최종 greedy 평가:

```bash
!python evaluate.py \
  --model "/content/drive/MyDrive/result_honest_v10/honest_v10_best.pt" \
  --device auto \
  --show-first-grid
```

## 평가와 best 모델

평가는 O=`(1,1)`, epsilon=0의 단일 deterministic rollout이다. best 모델은
다음 순서로 비교한다.

```text
1. full_survival
2. final_alive
3. official_reward
4. fewer water_attempts
5. fewer final_dry
```

`episode_steps`는 best 선택에 사용하지 않는다. 학습 로그에는 행동별 횟수,
물주기 직전 평균 수분, 수분 구간별 물주기 횟수, 음수 water_aux 횟수,
최대 연속 물주기 횟수가 함께 기록된다.

## 출력

- `honest_v10_best.pt`
- `honest_v10_last.pt`
- `training_episodes.csv`
- `periodic_evaluation.csv`
- `best_evaluation.csv`
- `final_evaluation.csv`
- `summary.json`
