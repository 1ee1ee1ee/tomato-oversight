# Tomato-Oversight

**2026 창의설계축전** — AI 안전(AI Safety) 강화학습 연구 프로젝트.

tomato-watering 그리드월드에서 **감시자(overseer)가 재배 로봇의 "치팅"을 감사(audit)** 하도록 학습시킨다.
겉보기 성과만 보는 감시(체제 1)와 행동 로그까지 보는 감시(체제 2)의 적발 성능 차이를 측정하는 것이 목표.

> 전체 연구 설계는 [`docs/설계문서.md`](docs/설계문서.md) 참고 (이 저장소의 "정답 문서").

---

## 프로젝트 구조

```text
tomato-oversight/
├─ src/
│  ├─ __init__.py
│  ├─ grower_env.py       # 재배 로봇 환경 (TomatoWateringEnv, E0/E1)
│  └─ wrappers.py         # 학습용 래퍼 (신선도 관측 추가 + 보상 shaping)
├─ scripts/
│  └─ 01_validate_env.py  # 환경 검증/시연 스크립트
├─ notebooks/             # 팀원별 Colab 노트북 (학습 실행)
│  ├─ 00_colab_bootstrap.ipynb   # 환경 확인·시각화
│  └─ 01_train_honest.ipynb      # 정직 로봇 DQN 학습
├─ models/                # 학습된 정책 (.zip은 Git 미포함 → Google Drive 보관)
├─ docs/
│  └─ 설계문서.md          # 연구 설계 백브리핑
└─ requirements.txt
```

## 연구 구조 (2단계 독립 학습)

1. **재배 로봇 학습** → 정직 / 치터 로봇을 각각 DQN으로 학습 후 **정책 고정**
   - 정직: `reward_mode="true"`, `allow_o_tile=False`
   - 치터: `reward_mode="apparent"`, `allow_o_tile=True`
2. **감시자 학습** → 고정된 로봇들을 상대로 `CONTINUE`/`AUDIT` 시점을 학습
   - 체제 1(결과 관측형) vs 체제 2(행동 관측형) 비교

환경 단계: **E0**(기준선) · **E1**(O 타일=핵심 실험) · E2(S만) · E3(O+S 확장). 발표는 E0+E1로 완성.

---

## 시작하기

### A) Google Colab (팀 협업 · 권장)

가장 빠른 방법: [`notebooks/00_colab_bootstrap.ipynb`](notebooks/00_colab_bootstrap.ipynb)를 Colab에서 열고 **`런타임 > 모두 실행`**.
환경 세팅·시각화·무작위 정책 애니메이션까지 자동으로 확인됩니다.

직접 셀을 만들려면 첫 셀에 붙여넣고 실행:

```python
!pip install -q stable-baselines3 gymnasium
!git clone https://github.com/<팀계정>/tomato-oversight.git
%cd tomato-oversight

from src.grower_env import TomatoWateringEnv, GrowerConfig
env = TomatoWateringEnv(GrowerConfig(mode="E1"))
obs, info = env.reset(seed=42)
print(env.render())   # 텍스트(ansi) 렌더 — Colab에서 정상 작동
```

학습 결과는 Google Drive에 저장(세션이 끊겨도 유지):

```python
from google.colab import drive
drive.mount('/content/drive')
model.save('/content/drive/MyDrive/tomato-oversight/models/honest_dqn')
```

> ⚠️ **Colab 주의점**
> - `render_mode="human"`(Tkinter 창)은 Colab에서 **작동 안 함** → 격자 이미지는 `render_mode="rgb_array"`로 만들고 `plt.imshow(env.render())`로 표시 (부트스트랩 노트북 3~4장 참고). 텍스트만 필요하면 기본 `env.render()`의 ansi 출력 사용.
> - 무료 세션은 유휴 ~90분/최대 ~12시간에 끊김 → 학습 중간에 Drive로 체크포인트 저장.
> - 5×5 격자 DQN은 작아서 **GPU 불필요, CPU 런타임으로 충분**.

### B) 로컬 실행

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 환경 시연 (로컬은 Tkinter 창 렌더 가능)
python scripts/01_validate_env.py --mode E1 --render --seed 42
```

---

## 팀 협업 규칙

- **코드(.py)는 브랜치에서 작업 → PR로 병합.** 무거운 로직은 반드시 `src/`의 .py로, 노트북은 얇게(실행만) 유지.
- **노트북은 사람/실험별로 분리** (예: `honest_train.ipynb`, `cheater_train.ipynb`, `monitor_train.ipynb`). 한 노트북을 여럿이 동시에 돌리지 않기.
- **학습된 정책(.zip)은 Git에 올리지 않고** Google Drive 공유폴더에 보관 (설계상 재배 로봇 정책은 "고정").

## 진행 상태

- [x] 재배 로봇 환경 `TomatoWateringEnv` (E0/E1)
- [x] 환경 검증 스크립트
- [x] Colab 부트스트랩 노트북 + matplotlib 렌더러 (`rgb_array`)
- [x] 정직 로봇 DQN 학습 노트북 + 학습 래퍼 (`01_train_honest`)
- [ ] 치터 로봇 DQN 학습 (`02_train_cheater`)
- [ ] 감시자 환경 (체제 1 / 체제 2)
- [ ] 감시자 DQN 학습 및 체제 비교 실험
