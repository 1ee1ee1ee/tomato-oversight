# 실내 온보드 자율 드론

GPS도 클라우드도 쓰지 않는다. 기체 안에서 인식하고, 기체 안에서 판단하고,
기체 안에서 막는다.

```
python3 -m drone.main --fast      # 하드웨어 없이 즉시 완주 (설치할 것 없음)
python3 -m drone.main --order "의자 몇 개 있는지 세줘" --fast
python3 -m unittest discover -s drone/tests -t .
```

실행하면 이륙 → 탐색 회전 → 대상 접근 → 정지 관찰 → 복귀 → 착륙까지
전 과정이 돌아가고 `flight_log.jsonl`에 매 틱이 기록된다.

## 지금 상태

| 계층 | 상태 |
|---|---|
| 판단(policy) · 안전(guard) · 명령 컴파일러 · 시뮬레이션 | **동작 확인. 테스트 108개** |
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

### 장애물 감지는 별도 센서 없이 OAK-D 뎁스로 한다

ToF 거리계(VL53L1X 등)를 따로 달지 않는다. OAK-D Lite 가 이미 스테레오
뎁스맵을 만들고 있으므로, 그 중앙 영역의 하위 5퍼센타일을 전방 거리로 쓴다.

```
스테레오 뎁스맵 ──▶ 중앙 50%×40% 크롭 ──▶ 유효범위(0.2~8m) 필터
                    (가장자리는 프롭)        │
                                            ▼
                        유효 픽셀 400개 미만 ──▶ forward_blind = True
                        그 외 ──▶ 하위 5퍼센타일 = front_m
```

설계상 중요한 두 가지:

- **검출 결과가 아니라 뎁스 원본을 쓴다.** 모델이 아무것도 못 알아봐도
  벽은 벽이다. Guard 원칙(아래)과 같은 이유다.
- **`front_m=None`(센서 없음)과 `forward_blind=True`(눈 감음)를 구분한다.**
  전자는 검사를 건너뛰고, 후자는 전진을 막는다. 무늬 없는 흰 벽에 가까이
  붙으면 스테레오가 통째로 실패하는데, 그때가 바로 멈춰야 할 순간이다.
  거리를 0 으로 지어내지도, 그냥 통과시키지도 않는다.

**OAK-D 는 전방만 본다** (화각 약 70°). 옆과 뒤는 VL53L1X ToF 세 개가 맡는다
(`rangefinders.py`). 카메라와 거리계는 서로 다른 하드웨어이고 고장도 따로
나므로, 한쪽을 다른 쪽 클래스에 넣지 않고 `WithRangefinders` 로 합성한다.

**후방까지 다는 이유**: Policy 는 대상에 너무 가까우면 물러서고, 전방 뎁스가
안 잡힐 때도 후진으로 빠져나온다. 뒤에 눈이 없으면 그 탈출로가 곧 충돌이 된다.

```bash
python3 -m drone.main --perception oakd --rangefinders vl53l1x --link mavlink
```

**주소 충돌 주의**: VL53L1X 는 전원을 넣으면 전부 같은 주소(0x29)로 올라온다.
XSHUT 핀으로 하나씩만 깨우면서 주소를 재할당해야 세 개를 동시에 쓸 수 있다.
`VL53L1XArray` 가 이걸 처리한다.

### 자연어 명령은 이륙 전에 딱 한 번 읽는다

```bash
python3 -m drone.main --order "페트병 3개 세줘" --fast
```

```
policy.py 위에 한 층 더:

  "페트병 3개 세줘"
        ↓  compiler.py  — 지상에서, 이륙 전에, 단 한 번
  MissionSpec(behavior=COUNT, target=bottle, n=3, stop=1.5m, alt=1.3m)
        ↓  mission_spec.validate()  — Guard 의 한계와 대조
  ✅ 통과 → 비행 시작        ❌ 거부 → 프로펠러도 안 돈다
        ↓
  이후 비행은 전부 결정론적 상태기계
```

**명령을 비행 중에 해석하지 않는 것이 요점이다.** 언어 모델을 제어 루프
안에 넣으면 모델이 틀렸을 때 그 결과가 곧바로 기체 움직임이 된다. 대신
지상에서 한 번 컴파일하고, 잘못된 명령은 이륙 전에 거른다.

검증이 잡아내는 대표적인 모순:

```
$ python3 -m drone.main --order "사람 앞 30cm 까지 가" --dry-run
  정지 거리: 0.3m
  ❌ 정지 거리 0.3m 가 전방 회피 여유 1.0m 보다 짧다
     — Guard 가 전진을 계속 막게 된다
```

이걸 못 걸러내면 기체는 "가만히 떠 있기만 하다가 임무 시간이 끝나는"
형태로 실패하고, 그때는 원인을 찾기가 훨씬 어렵다.

#### 임무 유형 (화이트리스트)

| behavior | 명령 예시 | 동작 |
|---|---|---|
| `approach_inspect` | "사람 찾아서 앞에 서줘" | 찾아가 1.5m 앞 정지, 관찰, 복귀 |
| `count` | "의자 몇 개 있는지 세줘" | 여러 개를 찾아 세고 복귀 (같은 개체 중복 계수 방지) |
| `follow` | "나 따라다녀" | 거리를 유지하며 계속 추종 (배터리/시간이 종료 결정) |
| `patrol` | "한 바퀴 돌면서 살펴봐" | 다가가지 않고 회전하며 관측만 기록 |

#### 컴파일러 백엔드 두 가지

| 백엔드 | 방식 | 의존성 | 쓸 곳 |
|---|---|---|---|
| `rules` (기본) | 한국어/영어 키워드 파싱 | **없음** | 오프라인, 결정론적, 테스트 |
| `claude` | Claude API + 구조화 출력 | `anthropic`, 네트워크 | 표현이 자유로운 명령 |

`claude` 백엔드는 **지상에서** 도므로 1~3초 지연이 문제가 되지 않는다.
구조화 출력(`output_config.format`)을 쓰기 때문에 스키마를 어긴 응답이
나올 수 없고, 나온 값은 다시 `validate()` 를 통과해야 한다.

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
| 전방 거리 < 1.0m | 전진 차단 |
| **전방 뎁스 측정 불가** | **전진 차단** (후진·회전·착륙은 허용) |
| 좌/우 0.5m 이내 | 해당 방향 횡이동만 차단 |
| **후방 0.5m 이내** | **후진 차단** (회전으로는 탈출 가능) |
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
| **옵티컬 플로우 + 하방 거리계** | MicoAir MTF-02P | **약 3.2만원** ($22.90) | 낮음 — ArduPilot이 기본 지원 |
| 같은 방식, 저조도 대응 | Holybro H-Flow | 약 24만원 (€154.95) | 낮음 (CAN 연결) |
| VIO / SLAM | RealSense T265(단종) 또는 자체 구현 | 50만원+ | 연구 과제 |

> **Matek 3901-L0X는 단종(EOL)됐다.** ArduPilot 지원 목록에서도 빠졌다.
> 후속으로는 MicoAir MTF-02P를 쓴다 — 더 싸고, 측정 범위가 2m에서 6m로 늘었다.

**옵티컬 플로우로 가세요.** 마우스 센서와 같은 원리로 바닥 움직임을 읽는
3g짜리 부품 하나가 실내 자율의 문턱을 넘겨줍니다. ArduPilot EKF3가 이걸
위치 소스로 그대로 받아줍니다.

### 대신 이게 망가지는 조건을 알아야 한다

옵티컬 플로우는 **바닥 무늬가 없거나, 어둡거나, 너무 높으면** 품질이 떨어지고
그러면 위치 추정이 통째로 무너집니다.

- 단색 장판·흰 타일 위: 무늬가 없어 추적 실패 → **바닥에 신문지나 매트를 깔 것**
- 조도 부족: 실내등을 다 켤 것
- MTF-02P의 ToF는 6m까지 닿는다(구형 3901-L0X는 2m였다). 일반적인 방 천장
  높이에서는 여유가 충분하므로 하방 거리계를 따로 달 필요가 없다. 순항 고도
  1.3m·상한 2.0m는 센서 한계가 아니라 **방 크기에 맞춘 값**이다 — 체육관처럼
  천장이 높은 공간이면 `config.py`에서 올려도 된다

`Guard`가 `flow_quality < 50`에서 즉시 HOLD를 거는 것이 이 실패에 대한 방어다.

---

## 부품표 (실내용, 총 120~165만원)

야외용 X500급 프레임을 실내에 들이면 안 된다. **프롭가드는 옵션이 아니라 필수.**

| 항목 | 부품 | 가격 |
|---|---|---|
| 프레임 | 프롭가드 일체형 3~4인치 시네훕 (또는 250급 + 가드) | 6~12만 |
| 모터·ESC | 1806~2004급 ×4 + 4in1 ESC 30A | 12~18만 |
| 비행 컨트롤러 | Holybro Pixhawk 6C Mini (ArduPilot) | 25~30만 |
| **옵티컬 플로우** | **MicoAir MTF-02P (플로우 + 6m ToF, 1g)** | **3~5만** |
| **온보드 컴퓨터** | **Raspberry Pi 5 4GB** (NPU 불필요 — 아래 참조) | 13~18만 |
| 인식 카메라 | OAK-D Lite (스테레오 + 카메라 내 1.4 TOPS 추론) | 20~25만 |
| **측·후방 거리계** | **VL53L1X ToF ×3** (좌·우·후) | **4.5만** |
| 조종기·수신기 | RadioMaster Pocket ELRS | 10~15만 |
| 배터리 | 4S 1500mAh ×3 | 9~15만 |
| 충전기 | ISDT Q6 / HOTA D6 + 세이프티백 | 8~12만 |
| 전원 | **Pi 전용 5V 5A BEC** | 2~3만 |
| 예비 부품 | 프롭, 가드, 암 | 5~10만 |
| | **합계** | **120~165만** |

### 부품 선택 이유

**OAK-D Lite** — 추론이 카메라 안에서 끝나고 `(라벨, 확신도, X, Y, Z[m])`가
바로 나온다. Pi의 CPU를 거의 안 쓰고, 스테레오 실측이라 단안처럼 거리를
틀리지 않는다. 실내 온보드 자율에 가장 잘 맞는 선택.

**호스트 NPU는 사지 말 것** — OAK-D Lite가 카메라 안에서 1.4 TOPS로 추론을
끝내므로, 호스트에 Hailo-8L 같은 가속기를 또 다는 건 중복이다. Pi는 우리
Python 루프와 MAVLink만 돌리면 되므로 4GB로 충분하다. 약 15만원과 55g을
아낀다. 호스트에서 별도 모델(소형 VLM 등)을 돌리게 되면 그때 추가하면 된다.
Jetson Orin Nano Super는 2026년 7월 인상으로 $399(국내 70~85만원)라 과하다.

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
  OAK-D Lite ──USB3──┐
                     ├── Raspberry Pi 5 4GB ──USB──┐
  VL53L1X ×3 ──I2C───┘   (좌·우·후 XSHUT: GPIO      │
                          17 / 27 / 22)             │  MAVLink2
                                                    │  /dev/ttyACM0
                                    ┌───────────────┴──┐
              MicoAir MTF-02P ──────┤  FC (ArduPilot)  │── 4in1 ESC ── 모터 ×4
              UART, MAVLink1,       └──────────────────┘
              115200, SERIAL4(GPS2)          │
                                    5V 5A BEC ──→ Pi (별도 전원!)
                                        └── GND 를 FC 와 공통 ★필수
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

MicoAir MTF-02P를 SERIAL1에 연결했을 때의 값 (ArduPilot 공식 문서 기준):

```
SERIAL1_PROTOCOL  1      # MAVLink1 — MSP 아님
SERIAL1_BAUD      115    # 115200
SERIAL1_OPTIONS   1024   # 펌웨어 4.5.0 이상에서만
FLOW_TYPE         5      # MAVLink
RNGFND1_TYPE      10     # MAVLink
RNGFND1_ORIENT    25     # 하방
RNGFND1_MIN       0.01
RNGFND1_MAX       8
```

`RNGFND1_MIN`/`MAX`는 **미터** 단위다. 구형 펌웨어에서는 `RNGFND1_MIN_CM`/
`MAX_CM`(센티미터)였으므로, Mission Planner에 어느 쪽 이름이 보이는지 확인할 것.
`SERIAL1`이 아닌 포트에 연결했다면 위 `SERIAL1_*`을 해당 번호로 바꾼다.

### 전원 — FC 로 Pi 를 돌릴 수 없다

FC 내장 BEC 는 보통 **2A** 이고 Raspberry Pi 5 는 피크 **5A** 를 먹는다.
여기에 OAK-D 가 0.5~1A 를 더 쓴다. 이륙 순간 모터가 전류를 당길 때 Pi 가
리셋되므로 **별도 5V 5A BEC 가 선택이 아니라 필수**다. GND 를 FC 와 공통으로
묶지 않으면 UART 가 간헐적으로 끊긴다.

### FC 연결 — USB 를 먼저 쓴다

기본값이 `/dev/ttyACM0:921600` 인 이유다. FC 의 USB 를 Pi 에 꽂으면 배선 실수
여지가 없고 레벨 문제도 없다. UART 직결(Pi 8/10번 핀 ↔ FC TX/RX 교차, GND 공통)
은 가볍고 안정적이지만, Pi 5 는 `/boot/firmware/config.txt` 에 `enable_uart=1`
을 넣고 `/boot/cmdline.txt` 에서 `console=serial0,115200` 을 지워야 한다.
SITL 은 `--mavlink-url udp:127.0.0.1:14550` 으로 바꿔 쓴다.

### 이륙 전 필수 확인

1. **플로우 방향 검증** — 기체를 손에 들고 앞뒤로 움직이며 Mission Planner의
   `FLOW_QUALITY`가 150 이상 나오는지, `OPTICAL_FLOW` 값의 부호가 움직임
   방향과 맞는지 확인. 부호가 반대면 기체가 발산한다.
2. **거리계 값** — 지면에서 1m 들었을 때 Mission Planner의 `sonarrange`가
   1.0 근처를 가리키는지.
3. **EKF 상태** — Mission Planner의 EKF 창에서 velocity/position variance가
   빨간색이 아닌지.
4. **거리계 방향** — 센서를 하나씩 손으로 가리며 좌·우·후 중 어느 값이
   변하는지 확인. 바꿔 꽂으면 Guard 가 반대쪽을 막는다.

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

- **개방형 어휘 검출** — 지금은 모델이 학습한 클래스(MobileNet-SSD 20종)만
  찾는다. YOLOE-26 같은 개방형 어휘 모델을 올리면 재학습 없이 텍스트
  프롬프트만으로 "빨간 소화기" 같은 대상을 찾을 수 있다. `compiler.py` 의
  `LABELS` 사전이 필요 없어지고, 한국어를 영어로 옮겨 그대로 넣으면 된다.
  **명령 컴파일러의 짝이 되는 조각이므로 다음 우선순위는 이것이다.**
- **온보드 VLM** — Jetson Orin Nano 8GB에서 Qwen2.5-VL-3B, VILA-1.5-3B급이
  돌아간다. 컴파일러의 `claude` 백엔드를 온보드 VLM으로 바꾸면 네트워크
  없이도 자유로운 명령을 받을 수 있다. 추론이 1~3초라 지상 컴파일 단계에만
  쓸 것 — 비행 루프에는 넣지 않는다.
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
  mission_spec.py 컴파일된 임무 + 이륙 전 검증.
  compiler.py     자연어 → MissionSpec (rules / claude 백엔드).
  main.py         실행 루프.
  sim/            SITL 파라미터.
  rangefinders.py VL53L1X ToF 배열 (좌·우·후방).
  tests/          108개.
```
