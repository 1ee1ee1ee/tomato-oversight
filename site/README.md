# 공개 재생 페이지

한국어 명령을 이륙 전에 검증하고, 통과한 것만 시뮬레이터로 비행시킨 로그를
브라우저에서 재생합니다. 서버가 없습니다 — 정적 파일 두 개입니다.

| 파일 | 손으로 고치나 | 무엇 |
|---|---|---|
| `orders.txt` | **예** | 페이지에 실릴 자연어 명령 목록 |
| `index.html` | 예 | 페이지 자체 (외부 스크립트 없음) |
| `flight_data.json` | 아니오 — 생성물 | 컴파일 결과 + 비행 프레임 |
| `build.py` | 가끔 | 위 JSON 을 만드는 스크립트 |

## 같이 고치는 법

**명령을 하나 추가하고 싶을 때** — 코드를 몰라도 됩니다.

1. GitHub 에서 [`site/orders.txt`](https://github.com/1ee1ee1ee/tomato-oversight/edit/main/site/orders.txt) 를 엽니다.
2. 한국어 명령을 한 줄 씁니다. 예: `의자 3개 세줘`
3. "Commit changes…" → 새 브랜치에 커밋하고 Pull Request 를 엽니다.
4. CI 가 그 명령을 컴파일하고, 통과하면 시뮬레이터로 끝까지 비행시킵니다.
5. 병합되면 공개 페이지가 자동으로 새로 배포됩니다.

CI 가 `python3 site/build.py --check` 로 **재생 데이터가 최신인지** 확인하므로,
`orders.txt` 만 고친 PR 은 빨간불이 뜹니다. 로컬에서 이렇게 맞춰주세요.

```bash
python3 site/build.py            # flight_data.json 을 다시 만든다
git add site/flight_data.json site/orders.txt
```

Python 을 돌릴 수 없다면 PR 에 그렇게 적어두세요. 대신 돌려서 커밋을 얹어드립니다.

**통과·거부는 사람이 정하지 않습니다.** 페이지의 "통과"/"거부" 딱지는
`drone/mission_spec.py` 의 `validate()` 가 내린 판정을 그대로 옮긴 것입니다.
그래서 말이 안 되는 명령을 넣어도 페이지가 좋게 포장해주지 않습니다 —
거부된 채로, 거부 이유와 함께 실립니다. 그게 이 페이지의 요점입니다.

**페이지 모양을 고치고 싶을 때** — [`site/index.html`](https://github.com/1ee1ee1ee/tomato-oversight/edit/main/site/index.html)
한 파일이 전부입니다. CSS·JS 가 안에 들어 있고 CDN 을 쓰지 않습니다
(글꼴만 Google Fonts). 밝은 테마와 어두운 테마 토큰이 `:root` 에 함께 있으니
색을 바꿀 때는 양쪽을 같이 봐주세요.

## 로컬에서 보기

`index.html` 을 두 번 눌러 여는 방식(`file://`)으로는 브라우저가
`flight_data.json` 요청을 막습니다. 서버를 하나 띄우세요.

```bash
python3 site/build.py
python3 -m http.server -d site 8000
# http://localhost:8000
```

## 파일 하나로 뽑기

발표용 USB, 메일 첨부, fetch 를 막는 샌드박스처럼 서버를 띄울 수 없는 자리를
위해 데이터까지 박아 넣은 단일 HTML 을 만들 수 있습니다. 두 번 눌러 열립니다.

```bash
python3 site/build.py --single 리플레이.html
python3 site/build.py --single 리플레이.html --bare   # 문서 껍데기 없이
```

배포되는 사이트는 이 경로를 쓰지 않습니다 — 사이트는 `index.html` + JSON 둘입니다.

## 배포

`.github/workflows/pages.yml` 이 `main` 푸시마다 단위 테스트 → 데이터 최신 확인
→ GitHub Pages 배포까지 합니다. 손으로 올리는 단계는 없습니다.

저장소 설정에서 Pages 를 켜는 단계도 없습니다. 배포 잡의
`actions/configure-pages` 가 사이트가 없으면 만들고, 이미 있으면
아무것도 하지 않습니다.

> 조직 정책 등으로 자동 생성이 막히면 배포 잡이 여기서 실패합니다.
> 그때만 **Settings → Pages → Source** 를 **GitHub Actions** 로 직접 바꿔주세요.

## 이 페이지와 관제 대시보드의 차이

| | 이 페이지 (`site/`) | 관제 대시보드 (`drone/dashboard/`) |
|---|---|---|
| 데이터 | 미리 기록된 로그 재생 | 지금 나는 기체의 실시간 상태 |
| 명령 입력 | 없음 (읽기 전용) | 있음 — 입력하면 **이륙합니다** |
| 공개 | 공개 URL 로 배포 | 기본 루프백 전용, 밖으로 열려면 토큰 필수 |

관제 대시보드는 공개하지 마세요. `POST /api/order` 는 실제로 프로펠러를 돌립니다.
