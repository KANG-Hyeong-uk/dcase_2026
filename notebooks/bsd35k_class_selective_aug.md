# bsd35k_class_selective_aug.ipynb — 실험 문서 (60/20/20, val·test 순수 10k)

## 한 줄 요약

BSD10k를 **train 60% / val 20% / test 20%**로 나누고, **35k는 train fold에만** 섞은 채(val·test는 순수 BSD10k) BSD35k를 **클래스-선택적으로(CP/CP1)** 증강해 baseline(H-Acc 79.32%)을 넘는지 검증했다. **결과: CP·CP1 모두 baseline 미달.** 게다가 모델선택을 순수 10k val로 정직하게 고쳐도 결과가 바뀌지 않아, "val 오염이 원인"이라는 가설도 기각됐다.

---

## 1. 두 노트북의 핵심 차이

### DCASE2026_Task1_VSCode.ipynb (원본 베이스라인)

- **목적**: DCASE 2026 Task 1 공식 베이스라인(HATR) 재현.
- **데이터**: BSD10k 전체(10,956개)만 사용 (35k 증강 없음).
- **분할**: StratifiedKFold(5)로 train/test를 나누고, fold마다 train에서 StratifiedShuffleSplit(0.2)로 val 분리. test가 fold마다 달라지는 rotating fold.
- **평가**: 5 fold의 (서로 다른) test에서 평균.

### notebooks/bsd35k_class_selective_aug.ipynb (본 실험)

- **목적**: BSD35k를 클래스별로 선별 증강해 baseline 초과가 가능한지 검증.
- **데이터**: BSD10k 60/20/20 분할. **35k는 train에만** 추가. val·test는 순수 10k.
- **분할**: test 20%(2192) 고정 holdout. train_pool 80%(8764)에 StratifiedKFold(4) → train 60%(6573) / val 20%(2191). **모든 전략이 동일한 test 2192에서 평가** → 직접 비교.
- **증강**: per-class threshold 분석으로 CP(14클래스, 10,121개) / CP1(12클래스, 9,335개).
- **목표**: CP 또는 CP1의 H-Acc > baseline_b0(79.32%).

---

## 2. 데이터셋 분할 구조 (60/20/20)

```
BSD10k 전체 (10,956개)
        │  cbc.make_fixed_holdout(seed=1821, test_size=0.2)
 ┌──────┴──────────────────────────────────────┐
 │ train_pool 8,764 (80%)                      │ final_test 2,192 (20%) ← 고정, 순수 10k
 │   StratifiedKFold(4) → train 60% / val 20%  │   (모든 전략 공통, 35k 절대 미포함)
 │                                             │
 │  fold k:                                    │
 │   ├ train 6,573 (60%, 순수 10k) + 35k 추가  │ ← 35k는 여기에만
 │   │     baseline_b0: +0                      │
 │   │     CP:          +10,121 (14 classes)    │
 │   │     CP1:         +9,335  (12 classes)    │
 │   └ val   2,191 (20%, 순수 10k, 35k 없음)   │ ← 모델선택(early stop)도 순수 10k 기준
 │                                             │
 ▼                                             │
 학습 → 최종 평가 ──────────────────────────────┘  (항상 동일 test 2,192)
```

핵심 설계점: **35k는 train fold에만** 들어가고 **val·test는 순수 BSD10k**다. 이전(폐기) 버전은 (10k+35k) 합친 것을 KFold해서 val에도 35k가 섞였는데, 이번엔 KFold를 train_pool(순수 10k)에만 적용해 **모델선택이 순수 10k 분포 기준**으로 이뤄지도록 고쳤다.

---

## 3. 실험 조건 비교 표

| 항목 | DCASE2026_Task1_VSCode.ipynb | bsd35k_class_selective_aug.ipynb |
|---|---|---|
| **사용 데이터** | BSD10k 전체 (10,956) | BSD10k + BSD35k 선별(train만) |
| **train** | fold별 ~7,011 (10k만) | 6,573 (60% 10k) + 35k 추가분 |
| **val** | fold별 ~1,753 (10k) | 2,191 (20% 순수 10k, 35k 없음) |
| **test** | fold마다 다른 ~2,192 (rotating) | 고정 2,192 (20% 순수 10k, 전략 공통) |
| **분할** | KFold(5) + ShuffleSplit(val 0.2) | holdout(test 20%) + KFold(4)로 train/val 60/20 |
| **SEED** | set_seed() (42 추정) | 1821 |
| **모드** | both, audio | ('both',) |
| **folds** | 5 | 4 (=60/20 train/val) |
| **NUM_EPOCHS / BATCH / lr / patience / hidden** | 100 / 64 / 0.001 / 5 / 128 | 동일 |
| **35k 위치** | 없음 | **train fold에만** (val·test 미포함) |
| **결과 저장** | model_output/{mode}/fold_{k}/ | outputs/v4_35k_602020_pureval/ |

---

## 4. CP/CP1 전략 도출 (per-class threshold 분석)

35k를 클래스별로 어떻게 추가할지 결정하는 과정. **이 분석 자체는 가설 생성 단계**이고, 진짜 증거는 §6 학습 결과다.

### 4.1 confidence score
BSD35k v4 필터 스코어(outputs/confidence_filter_v4/.../BSD35k-CS_filter_predictions_v4.csv):
```
predicted_confidence_score = 1 + 4 * v4_filter_score
ge2/ge3/ge4 = predicted_confidence_score >= 2/3/4
```

### 4.2 클래스별 delta
각 클래스 c: `d_geX[c] = recall_geX[c] - recall_base[c]`, `best_thr = argmax(d_ge2,d_ge3,d_ge4)`, `best_delta = max(...)`.
- recall_base: 517_mlp_classification/pred_ge_2 (BSD10k-only) 의 fold 평균 대각선.
- recall_geX: v4_35k_baseline_model/v4_geX 의 fold 평균 recall.
- ⚠️ **이 delta는 "모든 클래스를 동시에 같은 threshold로 증강한 전역 run"에서 측정**한 값이다 (§7.4에서 이게 결함의 핵심).

### 4.3 전략
- **CP**: best_delta > 0 → best_thr로 ADD, 아니면 DROP. → 14 ADD(10,121), 9 DROP.
- **CP1**: best_delta ≥ +1.0(노이즈 마진)만 ADD. → 12 ADD(9,335), 11 DROP. CP 대비 is-w·sp-s 추가 DROP.

| threshold | 클래스 (CP) |
|---|---|
| ge2 추가 | ss-n, ss-u, ss-i, is-s, is-w |
| ge3 추가 | fx-v, is-e, sp-s |
| ge4 추가 | m-m, fx-el, fx-o, fx-h, ss-s, fx-ex |
| DROP | fx-n, fx-m, m-sp, sp-p, m-si, fx-a, is-p, sp-c, is-k |

---

## 5. 봐야 할 결과물

OUTPUT_ROOT = `baseline_confidnce_train/outputs/v4_35k_602020_pureval/`

| 파일 | 내용 |
|---|---|
| summary_results.csv | 전략×fold 메트릭 (raw) |
| aug_strategy_comparison.csv | **핵심 판정** — baseline_b0 vs CP vs CP1 (H-Acc 정렬) |
| per_class_recall_change.csv | 클래스별 recall 변화 (CP/CP1 − baseline) |
| cm_foldavg_{strategy}.csv / plots/cm_foldavg_3models.png | fold 평균 confusion matrix |
| per_class_threshold_analysis.csv | CP/CP1 선택 근거 |

성공 기준: **CP 또는 CP1의 hierarchical_accuracy_mean > baseline_b0(79.32%)**.

---

# ===== 실측 결과 (60/20/20, pure-10k val, 4-fold) =====

## 6. 결론 먼저: CP·CP1 모두 baseline 미달 (실험 실패)

| 전략 | 추가 35k | accuracy | **H-Acc** | macro acc | top acc | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| **baseline_b0** (35k 미추가) | 0 | 80.33 | **79.32** | 74.01 | 89.05 | — (천장) |
| **CP** (클래스선택) | 10,121 | 77.49 | **77.07** | 71.52 | 86.92 | **−2.25** |
| **CP1** (클래스선택, 보수) | 9,335 | 78.15 | **76.96** | 71.67 | 87.15 | **−2.36** |

(출처: aug_strategy_comparison.csv, 4-fold 평균, 동일 test 2,192)

> 참고: 이전 프로토콜의 전역 필터(v4_ge4 78.8 / ge3 78.0 / ge2 76.9 / all 76.2)도 전부 baseline 미달이었다. 즉 전역이든 선택이든 **35k를 같은 가중치로 추가하면 baseline이 천장**이다.

## 7. 진단 — 왜 실패했나 (confusion matrix 기반)

### 7.1 추가한 클래스는 올랐다 (가설의 절반은 맞음)
| 클래스(ADD) | baseline | CP | Δ |
|---|---:|---:|---:|
| ss-n | 70.83 | 75.32 | **+4.49** (CP1 +10.26) |
| fx-ex | 42.50 | 51.25 | +8.75 |
| sp-c* | 41.43 | 47.86 | +6.43 (*CP에선 DROP인데 부수 상승) |
| m-m | 56.25 | 58.20 | +1.95 |
| fx-el | 74.40 | 76.19 | +1.79 |
| is-e | 70.56 | 72.18 | +1.62 |

### 7.2 그런데 추가 안 한(DROP) 이웃 클래스가 붕괴 (가설이 놓친 부분)
| 클래스(DROP) | baseline | CP | Δ |
|---|---:|---:|---:|
| **fx-n** | 91.48 | 70.27 | **−21.21** (CP1 −28.98) |
| **fx-a** | 42.65 | 22.06 | **−20.59** |
| m-si | 80.63 | 71.30 | −9.33 |
| fx-m | 79.46 | 70.54 | −8.92 |
| m-sp | 87.23 | 81.57 | −5.66 |
| sp-p | 80.90 | 75.35 | −5.55 |

ss/fx-ex에서 얻은 +4~+9를, fx-n/fx-a/m-si의 −21/−21/−9가 **압도**했다 → macro 74.0 → 71.5.

### 7.3 결정적 증거 — 경계 침식(boundary stealing)
무너진 fx-n 샘플이 **어디로 갔는지** 추적:

| 오분류 경로 | baseline | CP |
|---|---:|---:|
| fx-n → **ss-n** | 0.034 | **0.197** |

(cm_foldavg_*.csv) ss-n에 35k를 **가장 많이**(ge2, 최저 threshold, 2538개) 부어넣자 ss-n 결정영역이 팽창해 **음향적 이웃 fx-n을 흡수**했다. ss-n의 +4.5(CP)/+10.3(CP1) 이득은 fx-n에서 빼앗아 온 것이다 (CP1이 ss-n을 더 강하게 밀어 fx-n이 더 붕괴: −28.98).

→ **augmentation은 클래스를 독립적으로 개선하지 않는다.** 클래스 C에 데이터를 더하면 C의 prior·경계가 커지고, 그 대가로 이웃(특히 DROP 클래스)의 recall을 깎는다.

### 7.4 per-class delta가 빗나간 이유
§4의 best_delta는 **전역 run**(모든 클래스 동시 팽창, 이웃 효과 상쇄)에서 측정됐다. CP는 **일부만**(ss-n만 ge2 flood, fx-n은 0) 팽창시켜 그 상쇄가 사라지고 한 클래스가 이웃을 일방적으로 잡아먹는다. **전역에서 잰 delta는 선택 조건으로 이전되지 않는다** — 이것이 가설의 근본 결함이다.

### 7.5 ★ 핵심 신규 발견: "val 오염"은 원인이 아니었다
이번 버전은 모델선택(early stop)을 **순수 10k val**로 고쳤다. 그 효과:

| | val accuracy | test accuracy | gap |
|---|---:|---:|---:|
| baseline_b0 | ~79.7 | 80.3 | ≈0 |
| **CP (순수 val, 이번)** | **~77.6** | **77.5** | **≈0** (정직) |
| (참고) CP (오염 val, 이전) | ~84.5 | ~78 | ~7 (거짓) |

val을 순수 10k로 바꾸자 val이 test와 일치(정직)해졌다. **그런데 그 정직한 모델 실력이 77.5%로 baseline 79.3% 미달.** 즉 이전의 높은 val(84%)은 35k 분포에 대한 거짓 신호였을 뿐, 모델선택을 고쳐도 35k-증강 모델의 진짜 실력은 baseline 아래다.

→ **val 오염은 증상이지 원인이 아니다.** 진짜 원인은 §7.3의 경계 침식(test 단계 현상)이라 val을 고쳐도 사라지지 않는다. (실제로 순수 val CP1은 76.96으로 오염 val 때(77.53)보다 오히려 낮다.)



## 8. 한 줄 결론

> 60/20/20 + 순수 10k val로 깨끗하게 다시 해도 CP·CP1은 baseline(79.32)을 못 넘었다(77.07 / 76.96). 원인은 "ss-n에 35k를 몰아주면 음향 이웃 fx-n을 잡아먹는 경계 침식"(fx-n recall 91.5→70.3, fx-n→ss-n 0.03→0.20)이며, **val 오염을 고쳐도(val≈test로 정직해져도) 모델 실력 자체가 baseline 미달**이라 사라지지 않는다. 같은 가중치 row 추가 프레임은 폐기하고, down-weight·prior 보정(P1)으로 마지막 검증 후 안 되면 BSD10k 내부 최적화로 전환한다.
