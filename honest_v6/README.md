# honest_v6

이 패키지는 합의한 최종 규칙 중 **정직 재배 로봇**만 학습한다.

## 반영된 규칙

- 5x5 격자, 토마토 5개, 중앙 시작
- O 후보: `(1,1)`, `(1,3)`, `(3,1)`, `(3,3)`
- 정직 로봇에게 O는 치팅 타일이 아니라 일반 벽으로 관측됨
- 벽의 위치는 관측하지만 O의 정체나 효과는 알 수 없으며 진입할 수 없음
- 행동은 상/하/좌/우/물주기 5개뿐이며 WAIT 없음
- 물을 준 뒤 0~499 step 적심, 500~999 step dry 생존, 1000 step 사망
- 정확히 500 step마다 선형 겉보기 수분 점수 또는 dry/사망 벌점 지급
- 사망 중간점검 벌점은 토마토별 한 번만 지급
- 이동과 물주기에 비용 지급
- 10,000 step 고정 종료 후 실제 생존/사망 수로 큰 최종 보상 계산
- 전부 죽어도 10,000 step 전에 조기 종료하지 않음
- 공식 점검 주기는 500 step으로 유지
- 학습에는 5-step Double DQN과 potential-based shaping 사용
- shaping은 학습 replay에만 사용하고 모델 평가에서는 완전히 제외

초기 상태는 모든 토마토가 step 0에 물을 받은 상태로 정의한다. 보상 계수는
`src/env.py`의 `HonestGrowerConfig`에 한곳으로 모아 두었다.

## Google Drive에서 Colab으로 실행

먼저 `honest_v6.zip`을 Google Drive의 원하는 폴더에 업로드한다.

### 1. Drive 마운트 및 압축 해제

Colab 첫 번째 셀:

```python
from google.colab import drive
drive.mount('/content/drive')
```

두 번째 셀에서 zip 경로만 실제 업로드 위치에 맞게 바꾼다.

```bash
!rm -rf /content/honest_v6
!unzip -q "/content/drive/MyDrive/honest_v6.zip" -d /content
%cd /content/honest_v6
!pip install -q -r requirements.txt
```

### 2. 규칙 테스트

```bash
!python -m unittest discover -s tests -v
```

모든 테스트가 `OK`인지 확인한 뒤 학습한다.

### 3. GPU 설정

Colab 메뉴에서 `런타임 > 런타임 유형 변경 > T4 GPU`를 선택한다.

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### 4. 학습

학습 결과를 Drive에 바로 저장하는 권장 명령:

```bash
!python train.py \
  --total-steps 1000000 \
  --device auto \
  --output-dir "/content/drive/MyDrive/result_honest_v6"
```

기본값은 1,000,000 environment step, 5-step Double DQN이다. 공식 보상은
500-step 점검 규칙을 그대로 사용한다. 학습 replay에는 다음 potential shaping을
추가하지만, greedy 평가와 best 모델 선정에는 공식 보상과 실제 생존 상태만 쓴다.

```text
Phi(s) = 1.0 * sum(moisture) + 5.0 * sum(alive)
shaping = gamma * Phi(next_state) - Phi(state)
training_reward = official_reward + shaping
terminal Phi = 0
```

`training_episodes.csv`에는 `official_reward`, `shaping_reward`,
`training_reward`가 각각 분리되어 기록된다.

학습 중 50,000 step마다 다음 파일이 Drive에 갱신된다.

- `honest_v6_last.pt`: 최근 모델
- `honest_v6_best.pt`: 네 O 위치 평가에서 가장 좋은 모델
- `training_episodes.csv`
- `periodic_evaluation.csv`
- `best_evaluation.csv`

### 5. 중단 후 이어서 학습

저장된 모델을 초기값으로 추가 학습할 수 있다.

```bash
!python train.py \
  --total-steps 500000 \
  --resume-model "/content/drive/MyDrive/result_honest_v6/honest_v6_last.pt" \
  --output-dir "/content/drive/MyDrive/result_honest_v6"
```

### 6. 최종 평가

```bash
!python evaluate.py \
  --model "/content/drive/MyDrive/result_honest_v6/honest_v6_best.pt" \
  --episodes-per-o 3 \
  --device auto
```

가장 중요한 결과는 `full_survival_rate`와 `mean_final_alive`다.

## 파일 구조

```text
honest_v6/
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── env.py
│   └── ddqn.py
└── tests/
    ├── test_env.py
    ├── test_nstep.py
    └── test_shaping.py
```
