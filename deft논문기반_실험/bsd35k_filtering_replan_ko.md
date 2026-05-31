# BSD35k Augmentation 재계획 (v4_35k_baseline모델.ipynb 기반)

작성 기준 파일:
- **`notebooks/v4_35k_baseline모델.ipynb`** (실제 *증강* 실험: BSD10k train80 + v4 필터 BSD35k → 고정 BSD10k 20% holdout 평가)
- `notebooks/clean_label,v2,v3,v4.ipynb` (v4 = `confidence_filter_v4` 계열 score 생성기)
- `baseline_confidnce_train/outputs/517_mlp_classification` (같은 holdout의 BSD10k-only baseline 프록시)

> 앞선 초안은 *교체* 실험(`BSD35kTrain_BSD10kTest`)을 근거로 썼는데, 그건 BSD10k를 통째로 BSD35k로 바꾼 worst-case였다. 사용자 지적대로 **실제 증강 실험은 `v4_35k_baseline모델.ipynb`** 이고, 이 문서는 그 결과로 전면 재작성한 것이다.

---

## 0. 한 줄 결론

> **현재 방식(CLAP 임베딩 + DCASE baseline + v4 전역 confidence 필터)으로는 BSD35k를 더해도 baseline을 절대 못 넘는다.** v4 필터를 강하게 걸수록 좋아지지만(단조 증가), 최강(ge4)도 baseline보다 H-Acc -0.44다. 전역 필터는 "데이터를 안 넣는 극한"인 baseline에 *아래에서 수렴*할 뿐 넘지 못한다.
>
> baseline을 **넘으려면** 전역 threshold가 아니라 **class-selective 증강**이 필요하다: 증강이 *도움 되는 클래스만* 35k를 넣고, *해치는 클래스(특히 fx-n, fx-m)는 빼는* 것이다. per-class delta가 그 결정 신호이며, v4 confidence score가 아니다.

---

## 1. 결정적 비교 (같은 2192 holdout, 같은 모델)

`517_mlp pred_ge_2`(retained 100% = 필터 없음)가 곧 **BSD10k-only baseline**이고, holdout(final_test) 2192개가 v4_35k 실험과 **완전히 동일**(overlap 2192/2192)하다.

| setup | train 수 | accuracy | H-Acc | macro-acc | top-acc |
|---|---:|---:|---:|---:|---:|
| **BSD10k-only baseline** | 8,764 | **80.05%** | **79.24%** | **74.00%** | 88.93% |
| v4_ge4 (+7,633) | 16,397 | 79.28% | 78.80% | 73.45% | 88.34% |
| v4_ge3 (+15,566) | 24,330 | 78.74% | 77.99% | 72.62% | 87.73% |
| v4_ge2 (+23,851) | 32,615 | 77.38% | 76.92% | 71.13% | 87.00% |
| v4_all (+31,464) | 40,228 | 77.35% | 76.24% | 70.37% | 87.02% |

관찰:

1. **단조성**: 적게(엄격하게) 넣을수록 좋다. ge4 > ge3 > ge2 > all. → v4 confidence score는 *증강 regime에서는* 분명히 유효한 정렬 신호다. (앞서 *교체* 실험에서 "score⊥downstream(-0.07)"이라고 했던 건 교체 regime에서만 맞는 말이고, 증강에는 적용되지 않는다. 정정.)
2. **천장**: 그래도 ge4(-0.44 H-Acc)가 baseline을 못 넘는다. 전역 필터의 극한(아무것도 안 넣음)이 baseline이므로, 전역 필터만으로는 *원리상* baseline 초과 불가.

---

## 2. baseline을 넘으려면: per-class delta가 답

best 증강(ge4)과 baseline의 **클래스별 recall 차이**(5-fold 평균, 동일 holdout):

**35k가 해치는 클래스 (delta < 0):**

| class | baseline | v4_ge4 | Δge4 | v4_all | n_35k |
|---|---:|---:|---:|---:|---:|
| fx-n | 92.3 | 81.4 | **-10.9** | 62.4(-29.9) | 659 |
| fx-m | 77.9 | 72.9 | **-5.0** | 66.1 | 1542 |
| ss-i | 36.6 | 32.7 | -3.9 | 34.1 | 383 |
| is-e | 68.4 | 64.5 | -3.9 | 69.4 | 1360 |
| ss-u | 67.1 | 63.9 | -3.2 | 73.6 | 2573 |
| m-sp | 88.3 | 85.4 | -2.9 | 85.5 | 964 |
| sp-p | 81.1 | 78.6 | -2.5 | 76.7 | 69 |
| m-si | 81.7 | 80.4 | -1.3 | 77.0 | 1344 |
| fx-a | 43.5 | 42.4 | -1.1 | 31.2 | 730 |
| sp-s,is-k,is-p | | | -0.3~-0.4 | | |

**35k가 돕는 클래스 (delta > 0):**

| class | baseline | v4_ge4 | Δge4 | n_35k |
|---|---:|---:|---:|---:|
| m-m | 57.2 | 63.4 | **+6.2** | 2044 |
| fx-el | 70.5 | 76.2 | **+5.7** | 2000 |
| ss-n | 70.5 | 72.8 | +2.3 | 4831 |
| fx-v | 60.0 | 62.0 | +2.0 | 1852 |
| ss-s | 82.3 | 84.2 | +1.9 | 596 |
| fx-o | 84.1 | 85.8 | +1.7 | 4534 |
| fx-h | 86.4 | 87.8 | +1.4 | 1423 |
| fx-ex | 47.0 | 48.3 | +1.3 | 839 |
| is-w, is-s | | | +0.2~0.3 | |

해석:

- 순손실은 **fx-n(-10.9), fx-m(-5.0)** 두 클래스가 지배한다. 특히 fx-n은 35k 659개가 BSD10k의 fx-n과 음향/라벨이 달라서 ge4로도 회복 안 됨 → **class-specific 오염**의 전형.
- 반대로 m-m(+6.2), fx-el(+5.7) 등 10개 클래스는 35k가 순이득.
- 따라서 **"해치는 클래스는 빼고 돕는 클래스만 넣으면"** macro/H-Acc가 baseline을 넘을 여지가 실재한다. (거친 추정: 손실 클래스를 baseline으로 복원 + 이득 클래스 유지 → macro 73.45 → 75+ 가능.)

---

## 3. 재계획 (목표: BSD35k 증강으로 baseline 초과)

### 3.1 결정 신호
- (X) v4 confidence score 절대값 — 전역 정렬엔 좋지만 어느 클래스를 빼야 하는지는 못 알려줌.
- (O) **per-class 증강 delta** (이번 §2 표) — 어떤 클래스가 35k로 득/실인지 직접 측정값.

### 3.2 비교할 전략 (모두 같은 8764 train + 2192 holdout, v4 score 재사용·재학습 X)

정책 임계: `Δge4 < -1.0 → DROP`, `-1.0 ≤ Δge4 < +1.0 → ge4`, `Δge4 ≥ +1.0 → helped`.
(ge2/ge3/ge4 = `predicted_confidence_score = 1+4·v4_filter_score ≥ 2/3/4`, 기존 노트북과 동일 규칙.)

| 전략 | 정의 | 추가 샘플 수 |
|---|---|---:|
| B0 baseline | BSD10k train80만 (노트북에서 직접 재학습) | 0 |
| A_ge4 (참고) | + v4_ge4 전체 (이미 학습됨: H-Acc 78.80) | 7,633 |
| **C1 class-selective** | Δ<-1 클래스 **DROP**, 나머지(중립·이득) 전부 **ge4** | **5,306** |
| **C2 class-selective (volume)** | Δ<-1 클래스 **DROP**, 이득(Δ≥+1) 클래스는 **ge2**(데이터 최대), 중립 클래스는 **ge4** | **15,426** |

**DROP된 9개 클래스**: fx-n(-10.9), fx-m(-5.0), is-e(-3.9), ss-i(-3.9), ss-u(-3.2), m-sp(-2.9), sp-p(-2.5), m-si(-1.3), fx-a(-1.1). 두 subset 모두 14개 클래스만 추가.

가설: **C1/C2가 B0(79.24%)를 넘는 첫 전략**이 될 수 있다. fx-n·fx-m 제거만으로 macro +0.4 이상 회복, 이득 클래스(m-m +6.2, fx-el +5.7) 추가로 추가 상승 기대.

### 3.3 검증
동일 고정 holdout(2192)에서 5-fold train/val. 1차 판정 지표 = H-Acc(주), macro-acc, accuracy. **B0(79.24%) 초과**가 성공 기준.

---

## 4. 산출물 (생성 완료)

### 4.1 실행 노트북 (사용자가 실행)
- **`notebooks/bsd35k_class_selective_aug.ipynb`** — `v4_35k_baseline모델.ipynb`의 machinery(`confidence_baseline_common`, 고정 holdout, `BaseClassifier`) 그대로 재사용. `SEED=1821` 고정(기존 holdout 재현). 셀 구성:
  1. setup → 2. 고정 80/20 holdout → 3. BSD35k v4 로드 → 4. **per-class delta 계산 + policy 자동 산출**(기존 confusion matrix에서) → 5. C1/C2 subset 빌드 + CSV 저장 → 6. **학습**(`baseline_b0`, `C1`, `C2`; 기존 run 있으면 자동 skip) → 7. **전략 비교표**(기존 v4_ge4·v4_all 병기, B0 초과 여부 자동 판정) → 8. per-class recall 변화표.
  - `RUN_DATASET_LABELS`로 학습 대상 조절 가능. 노트북이 subset CSV·delta CSV·비교표·per-class 변화표를 모두 자동 저장.

### 4.2 사전 생성된 CSV (`outputs/bsd35k_class_selective/`, GPU 불필요)
- `reports/per_class_aug_delta.csv` — §2 표(클래스별 baseline/ge4/delta + policy)
- `reports/bsd35k_class_retained_analysis.csv` — 클래스별 정책·C1/C2 retained 수/비율
- `predictions/bsd35k_subset_C1_classselective.csv` (5,306) / `..._C2_...csv` (15,426)
- `generate_subsets.py` — 재현 스크립트

> 노트북(§4.1)은 학습 시점에 위 CSV를 자기 출력 폴더(`baseline_confidnce_train/outputs/v4_35k_class_selective/`)에 **다시 생성**하므로, 둘은 독립적으로 동일 결과를 낸다.

### 4.3 실행 비용
B0·C1·C2 = 3개 dataset × 5-fold × ~100ep. RTX 3060에서 대략 1~2시간(C2가 데이터 많아 가장 김). A_ge4·v4_all은 이미 학습되어 비교표에 그대로 병기됨(재학습 불필요).

---

## 5. 판정 기준 & 다음 한 수
1. 노트북 §7 비교표에서 **C1 또는 C2의 H-Acc > B0(79.24%)** 이면 성공.
2. 성공 시: 해당 class-selective 규칙(DROP 9-class + ge4/ge2 정책)을 **BSD35k 전체 deployment 규칙**으로 확정.
3. 미달 시 후속: (a) DROP 경계를 Δ<-0.5로 강화, (b) 이득 클래스 threshold 튜닝(ge2↔ge3), (c) fx-n/fx-m의 BSD35k 라벨을 직접 점검(class-specific 오염 원인 규명), (d) sample-weight 방식(트레이너 수정 필요).
