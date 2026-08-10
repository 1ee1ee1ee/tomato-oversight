# result_honest_v14 — Colab 1M run1 결과 (2026-08-10)

honest_v14(상관 리셋 + 두 길이 커리큘럼 + sync 평가/선정) 1M step, T4 37분.
체크포인트에 `obs_semantics: "life"` **태그 정상 포함** (v13 zip과 달리 로딩 안전).

## 판정: ⚠️ 합격 게이트 미통과 — 부분 개선

best = **900k**, 최종 10판 재평가:

| 지표 | v13 (1M) | v14 초안(로컬, 전부 드릴) | **v14 run1 (커리큘럼)** | 목표 |
|---|---:|---:|---:|---:|
| 표준 full_survival | **0.8** | 0.2 | 0.3 | ≥0.8 ❌ |
| 표준 mean_alive | 4.6 | 3.5 | 3.5 | — |
| 표준 official | +1116 | +187 | +400 | — |
| **sync 구조 alive** | 3.4 | **4.4** | 3.9 | ≥4.5 ❌ |
| sync 완전구조율 | ~0 | 0.7 | 0.4 | — |

- **sync 구조는 v13 대비 개선**(3.4→3.9)됐지만 초안(4.4)보다 후퇴.
- **표준 유지력은 여전히 v13에 크게 못 미침**(0.3 vs 0.8) — 커리큘럼이 초안의
  퇴화(0.2)를 소폭 회복하는 데 그침.
- 학습 곡선이 500k 이후 진동(600k: 0.8/4.8 ↔ 700k: 0.4/3.0), 선정 시점
  (900k: 0.6/4.4)과 재평가(0.3/3.5)의 격차도 큼 — **정책이 불안정**하고
  실패 꼬리(alive 1~2 에피소드)가 남아 있음.

## 원인 가설 (다음 라운드 입력)

두 과제(장기 유지 + 위기 구조)를 같은 1M 예산·같은 네트워크로 배우게 하자
서로 간섭하며 진동. v13은 단일 과제라 말미에 안정 수렴한 것과 대조.

## 다음 선택지 (팀 결정 대기)

1. **재학습 (a)**: drill 비율 0.5→0.3 + 총 1.5M(+LR 감쇠 완화) — 표준 비중 회복.
2. **재학습 없이 역할 분담 (b)**: 정직 에피소드·배신 K이전 = **v13**(표준 0.8),
   적응형 치터의 구조 두뇌 = **v14**(sync 3.9) — 적응형의 정직 구간은 짧아
   장기 유지력이 덜 중요하므로 현재 자산으로 즉시 진행 가능.
3. (a)+(b) 병행: (b)로 감시자 작업 먼저 진행, (a)는 백그라운드 개선.

## 파일

- `run1/summary.json` — 설정·최종 지표 전체
- `run1/periodic_evaluation.csv` — 100k마다 표준+sync 학습 곡선
- `run1/best_evaluation.csv` / `best_sync_evaluation.csv` — 선정 시점(900k) 5판
- `run1/final_evaluation.csv` / `final_sync_evaluation.csv` — 최종 10판 상세

**`.pt` 모델·training_episodes.csv는 Drive에** (`MyDrive/result_honest_v14/run1/`,
justin040531@gmail.com). repo에 넣으려면 Colab에서:

```python
!cp "/content/drive/MyDrive/result_honest_v14/run1/honest_v14_best.pt" /content/tomato-oversight/result_honest_v14/run1/
!cd /content/tomato-oversight && git add result_honest_v14 && git commit -m "add v14 best.pt" && git push
```
