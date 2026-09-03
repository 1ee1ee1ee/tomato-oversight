# 실내 온보드 자율 드론

GPS도 클라우드도 쓰지 않는다. 기체 안에서 인식하고, 기체 안에서 판단하고,
기체 안에서 막는다.

```
python3 -m drone.main --fast      # 하드웨어 없이 즉시 완주 (설치할 것 없음)
python3 -m unittest discover -s drone/tests -t .
```

실행하면 이륙 → 탐색 회전 → 대상 접근 → 정지 관찰 → 복귀 → 착륙까지
전 과정이 돌아가고 `flight_log.jsonl`에 매 틱이 기록된다.

## 지금 상태

| 계층 | 상태 |
|---|---|
| 판단(policy) · 안전(guard) · 시뮬레이션 | **동작 확인. 테스트 47개** |
| MAVLink 연동 (`link.py`) | 작성 완료, **실기체 미검증** |
| OAK-D / ONNX 인식 백엔드 | 작성 완료, **실기체 미검증** |

실기체에 올리기 전 반드시 아래 "실기체 투입 순서"를 따를 것.

---

## 구조

아래로 갈수록 빠르고, 단순하고, 믿을 수 있다.

```
policy.py    판단   0.5~10Hz    상태기계. 다음에 뭘 할지 정한다.
    ↓ 제안
guard.py     검증   매 틱       결정론적 if 문. 여기를 통과해야 나간다.
    ↓ 승인된 명령
link.py      통신   10~20Hz     MAVLink
    ↓
ArduPilot    제어   400Hz~1kHz  자세·위치. 이 코드는 여기 손대지 않는다.
```

**신경망은 인식에서만 일한다.** 판단을 상태기계로 둔 건 성능 때문이 아니라,
실내에는 "왜 그렇게 움직였는지 모르겠다"를 흡수할 여유 공간이 없기 때문이다.
학습 정책이나 소형 VLM으로 바꾸려면 `Policy.step()` 하나만 교체하면 된다.
Guard는 그대로 두면 된다 — 그게 Guard의 존재 이유다.

### Guard 설계 원칙

Guard는 **인식 모델의 보고가 아니라 센서 원본만** 본다.

모델에게 "지금 위험한가?"를 묻고 그 대답으로 안전을 판정하면, 모델이 틀렸을 때
그 사실을 알아낼 방법이 사라진다. 그래서 전방 차단은 ToF 거리계 값으로만
판단하고, 검출 결과는 쓰지 않는다. 이 원칙은 `test_guard.py`의
`test_guard_ignores_model_output_and_trusts_rangefinder`로 고정해 두었다.

Guard가 막는 것:

| 검사 | 동작 |
|---|---|
| 텔레메트리 두절 | HOLD |
| 배터리 20% 이하 | LAND (다른 모든 판단을 덮어씀) |
| 배터리 35% 이하 | 복귀 단계로 전환 요청 |
| **옵티컬 플로우 품질 저하** | **HOLD — 실내 최대 실패 원인** |
| 인식 두절 | 전진 성분만 제거 (후진·측면은 허용) |
| 전방 ToF < 1.0m | 전진 차단 |
| 지오펜스 경계 | 바깥 성분만 제거, 안쪽·접선은 통과 |
| 고도 0.8~2.0m 이탈 | 해당 방향 수직속도만 0 |
| 속도 초과 | 클램프 |

경계에서 명령 전체를 막지 않고 **성분만 깎는 것**이 핵심이다. 전부 막으면
기체가 벽에 갇힌다.

정상 비행에서 Guard 개입은 **0회**여야 한다. 상시 켜진 경고는 경고가 아니다.
`FullMission.test_mission_completes_without_guard_intervention`이 이걸 강제한다.

---

## 실내가 어려운 이유: GPS가 없다

야외 자율비행의 90%는 GPS가 공짜로 해결해준다. 실내에서는 그게 통째로 사라진다.
해법은 두 가지고, 가격 차이가 10배다.

| 방식 | 부품 | 가격 | 난이도 |
|---|---|---|---|
| **옵티컬 플로우 + 하방 거리계** | Matek 3901-L0X | **5~7만원** | 낮음 — ArduPilot이 기본 지원 |
| VIO / SLAM | RealSense T265(단종) 또는 자체 구현 | 50만원+ | 연구 과제 |

**옵티컬 플로우로 가세요.** 마우스 센서와 같은 원리로 바닥 움직임을 읽는
3g짜리 부품 하나가 실내 자율의 문턱을 넘겨줍니다. ArduPilot EKF3가 이걸
위치 소스로 그대로 받아줍니다.

### 대신 이게 망가지는 조건을 알아야 한다

옵티컬 플로우는 **바닥 무늬가 없거나, 어둡거나, 너무 높으면** 품질이 떨어지고
그러면 위치 추정이 통째로 무너집니다.

- 단색 장판·흰 타일 위: 무늬가 없어 추적 실패 → **바닥에 신문지나 매트를 깔 것**
- 조도 부족: 실내등을 다 켤 것
- 3901-L0X의 ToF는 **2m가 한계** → 순항 고도를 1.3m로 잡은 이유. 천장이 높은
  공간이면 TFmini-S(12m)를 따로 달 것

`Guard`가 `flow_quality < 50`에서 즉시 HOLD를 거는 것이 이 실패에 대한 방어다.

---

## 부품표 (실내용, 총 138~192만원)

야외용 X500급 프레임을 실내에 들이면 안 된다. **프롭가드는 옵션이 아니라 필수.**

| 항목 | 부품 | 가격 |
|---|---|---|
| 프레임 | 프롭가드 일체형 3~4인치 시네훕 (또는 250급 + 가드) | 6~12만 |
| 모터·ESC | 1806~2004급 ×4 + 4in1 ESC 30A | 12~18만 |
| 비행 컨트롤러 | Holybro Pixhawk 6C Mini (ArduPilot) | 25~30만 |
| **옵티컬 플로우** | **Matek 3901-L0X (플로우+ToF 일체, 3g)** | **5~7만** |
| 하방 거리계 | TFmini-S — 천장 높은 공간용 (선택) | 0~5만 |
| **온보드 AI** | **Raspberry Pi 5 8GB + Hailo-8L (13 TOPS)** | 28~35만 |
| 인식 카메라 | OAK-D Lite (스테레오 + 카메라 내 추론) | 20~25만 |
| 측방 회피 | VL53L1X ToF ×2~3 | 3~5만 |
| 조종기·수신기 | RadioMaster Pocket ELRS | 10~15만 |
| 배터리 | 4S 1500mAh ×3 | 9~15만 |
| 충전기 | ISDT Q6 / HOTA D6 + 세이프티백 | 8~12만 |
| 전원 | **Pi 전용 5V 5A BEC** | 2~3만 |
| 예비 부품 | 프롭, 가드, 암 | 5~10만 |
| | **합계** | **138~192만** |

### 부품 선택 이유

**OAK-D Lite** — 추론이 카메라 안에서 끝나고 `(라벨, 확신도, X, Y, Z[m])`가
바로 나온다. Pi의 CPU를 거의 안 쓰고, 스테레오 실측이라 단안처럼 거리를
틀리지 않는다. 실내 온보드 자율에 가장 잘 맞는 선택.

**Pi 5 + Hailo-8L** — Jetson Orin Nano Super는 2026년 7월 가격 인상으로
$399가 되어 국내 70~85만원이다. 온보드 VLM을 돌릴 게 아니면 과하다.

**Pi 전용 BEC** — 자작 드론 실패 원인 1위가 온보드 컴퓨터 브라운아웃이다.
FC의 보조 5V 출력으로 Pi를 돌리면 언젠가 반드시 떨어진다.

### 무게와 법규

이 구성은 900g~1.2kg다.

- **2kg 이하** → 비영리 목적이면 기체 신고 대상 아님
- **250g 초과** → 조종자 증명 필요 (4종, 온라인 교육 이수)
- **실내 비행** → 공역이 아니므로 비행승인·공역 제한을 받지 않는다.
  실내를 고른 것이 규제 부담을 가장 크게 줄여준다.

---

## 배선

```
                    ┌─────────────────┐
   OAK-D Lite ──USB─┤                 │
                    │  Raspberry Pi 5 │
   VL53L1X ×3 ──I2C─┤   + Hailo-8L    │
                    └────────┬────────┘
                             │ UART (Telem2, 921600)
                             │ MAVLink2
                    ┌────────┴────────┐
   Matek 3901-L0X ──┤  Pixhawk 6C Mini │── ESC ── 모터 ×4
   (UART, MSP)      └─────────────────┘
                             │
                        5V 5A BEC ──→ Pi (별도 전원!)
```

---

## ArduPilot 설정

### SITL로 먼저 검증

```bash
sim_vehicle.py -v ArduCopter -f quad \
    --add-param-file=drone/sim/indoor_noGPS.parm --console --map

# 다른 터미널에서
python3 -m drone.main --link mavlink --perception mock
```

`drone/sim/indoor_noGPS.parm`은 GPS를 끄고 옵티컬 플로우를 위치 소스로
쓰도록 EKF3를 바꾼다. **이 파일은 SITL 전용이다** — `FLOW_TYPE`,
`RNGFND1_TYPE` 값이 시뮬레이터용이다.

### 실기체 파라미터

EKF3 소스 설정은 파일과 동일하다.

```
AHRS_EKF_TYPE  3      EK3_SRC1_POSXY  0     # GPS 없음
EK3_ENABLE     1      EK3_SRC1_VELXY  5     # OpticalFlow
EK2_ENABLE     0      EK3_SRC1_POSZ   2     # RangeFinder
                      EK3_SRC1_YAW    1     # Compass
```

센서 타입 값(`FLOW_TYPE`, `RNGFND1_TYPE`, `SERIALx_PROTOCOL`)은 **펌웨어
버전마다 enum이 달라지므로 여기 적힌 숫자를 믿지 말고 Mission Planner의
드롭다운에서 고를 것.** Matek 3901-L0X는 MSP 프로토콜을 쓰므로 연결한
시리얼 포트의 프로토콜을 MSP로, `FLOW_TYPE`과 `RNGFND1_TYPE`을 각각 MSP
계열로 선택하면 된다.

### 이륙 전 필수 확인

1. **플로우 방향 검증** — 기체를 손에 들고 앞뒤로 움직이며 Mission Planner의
   `FLOW_QUALITY`가 150 이상 나오는지, `OPTICAL_FLOW` 값의 부호가 움직임
   방향과 맞는지 확인. 부호가 반대면 기체가 발산한다.
2. **거리계 값** — 지면에서 1m 들었을 때 100cm 근처가 나오는지.
3. **EKF 상태** — Mission Planner의 EKF 창에서 velocity/position variance가
   빨간색이 아닌지.

---

## 실기체 투입 순서

**순서를 지키세요.** 뒤집으면 "코드가 이상한 건지 기체가 이상한 건지"
구분이 안 돼서 디버깅이 불가능해집니다.

| 단계 | 내용 | 프롭 |
|---|---|---|
| 1 | 시뮬레이션 완주 (`--fast`) | — |
| 2 | SITL + `--link mavlink` 로 MAVLink 흐름 확인 | — |
| 3 | 실기체에 `--link mavlink` 연결, **프롭 제거 상태**로 명령 로그만 확인 | ❌ |
| 4 | 수동 비행으로 호버 튜닝, `FLOW_QUALITY` 실측 | ✅ |
| 5 | Loiter 모드로 실내 위치 유지 확인 (자율 코드 없이) | ✅ |
| 6 | `--perception mock` 으로 자율 루프 첫 비행 | ✅ |
| 7 | `--perception oakd` 로 실제 인식 투입 | ✅ |

**5단계를 통과하지 못하면 6단계로 가지 마세요.** Loiter가 안 되는 기체에
자율 코드를 얹으면 그 다음부터는 아무것도 진단할 수 없습니다.

비행 중에는 항상 조종기를 손에 들고, **Stabilize/AltHold로 즉시 낚아챌 수
있는 스위치**를 확보해 두세요. 자율 코드는 반드시 실패합니다. 문제는
실패할 때 사람이 잡을 수 있느냐입니다.

---

## 설정 바꾸기

숫자는 전부 `config.py` 한 곳에 있다.

```python
Limits(max_radius_m=4.0, min_alt_m=0.8, max_alt_m=2.0,
       max_speed_ms=0.6, min_front_m=1.0, min_flow_quality=50)

Mission(target_label="person", stop_distance_m=1.5,
        cruise_alt_m=1.3, inspect_seconds=4.0)
```

`max_radius_m`은 **방 크기의 절반보다 작게** 잡으세요. 지오펜스는 벽이 아니라
경고선입니다.

---

## 다음 단계

- **온보드 VLM** — Jetson Orin Nano 8GB에서 Qwen2.5-VL-3B, VILA-1.5-3B급이
  돌아간다. `Policy.step()`을 교체하면 "빨간 상자를 찾아서 그 앞에 서"
  같은 자연어 임무를 받을 수 있다. 추론이 1~3초라 판단 계층에만 쓸 것.
- **학습 정책** — SITL로 데이터를 모아 접근 정책을 학습시킬 수 있다.
  Guard는 그대로 두면 sim-to-real 실패가 추락으로 이어지지 않는다.
- **다중 대상** — 현재는 확신도 최고 검출 하나만 추적한다.

## 파일

```
drone/
  state.py        자료형. 의존성 없음.
  config.py       모든 조정값.
  guard.py        결정론적 안전 검증기.  ← 가장 중요한 파일
  policy.py       판단 상태기계.
  perception.py   mock / OAK-D / ONNX 백엔드.
  link.py         MAVLink + 시뮬레이터.
  main.py         실행 루프.
  sim/            SITL 파라미터.
  tests/          47개.
```
