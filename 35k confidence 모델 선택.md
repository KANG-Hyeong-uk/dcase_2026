# 35k confidence 모델 선택

## 0. 결론 요약

BSD35k confidence filtering 목적에서 가장 추천하는 모델은 **v4 Ensemble: `Rank average: binary filter probability + 5-class expected confidence score`**이다.

단, 여기서 중요한 점은 **v4의 F1-optimal threshold 0.18을 그대로 쓰는 것이 최종 추천은 아니라는 것**이다. 0.18은 BSD35k-CS의 83.11%를 retain하는 관대한 threshold라서 pseudo-label cleaning 목적에는 너무 느슨하다. 실제 BSD35k에서 "신뢰도 높은 샘플만 안전하게 추출"하려면 다음 운용이 더 적합하다.

| 목적 | 추천 score | threshold | BSD35k retained ratio | OOF expected precision | 해석 |
|---|---:|---:|---:|---:|---|
| 균형형 high-confidence subset | v4 rank average | 0.5 | 49.47% | 0.8040 | 절반 정도를 남기되 noisy sample을 꽤 줄이는 운용 |
| 추천 기본값 | v4 rank average | 0.7 | 29.23% | 0.8813 | false high-confidence 위험을 낮춘 안정적 필터 |
| 매우 엄격한 pseudo-label seed | v4 rank average | 0.8 | 19.29% | 0.9284 | precision 중심의 clean subset 구축 |
| 거의 확실한 seed만 사용 | v4 rank average | 0.9 | 9.56% | 0.9616 | recall을 희생하고 high-confidence seed만 확보 |

따라서 최종 판단은:

> **BSD35k에서 confidence label 없이 high-quality sample을 추출하는 주 모델은 v4 ensemble을 사용하고, 기본 deployment threshold는 0.7, 더 깨끗한 seed가 필요하면 0.8을 사용한다.**

이 결론은 validation accuracy가 아니라 **AUC-PR, threshold별 precision/retain trade-off, 3/4 confusion 구조, false high-confidence 위험, calibration 안정성, BSD35k 적용 분포**를 기준으로 한 것이다.

---

## 1. 문제 재정의: 이 실험은 5-class accuracy 문제가 아니다

원래 label은 confidence 1-5이지만 실제 목적은 BSD35k에서 다음을 수행하는 것이다.

- high-confidence sample만 추출
- noisy 또는 애매한 sample 제거
- pseudo-label training에 넣을 clean subset 구성
- confidence label이 없는 환경에서 threshold 기반 filtering 수행

따라서 평가 기준은 단순히 `predicted class == true class`가 아니다. 특히 BSD10k의 confidence 분포는 다음과 같이 심하게 불균형하다.

| confidence | 비율 |
|---:|---:|
| 1 | 약 1% |
| 2 | 약 6.9% |
| 3 | 약 30.1% |
| 4 | 약 55% |
| 5 | 약 7% |

이 구조에서는 confidence 4만 반복 예측해도 accuracy가 약 55%까지 나온다. 실제로 majority baseline은 `accuracy=0.5518`, `QWK=0.0000`이다. 따라서 accuracy가 0.57-0.58이라고 해서 모델이 confidence 구조를 잘 배운 것은 아니다. 특히 다음 질문이 더 중요하다.

- confidence 3과 4를 구분할 수 있는가?
- confidence 4 majority collapse가 아닌가?
- confidence 5를 별도로 분리할 수 있는가?
- confidence 1/2를 완전히 무시하지 않는가?
- high-confidence라고 뽑은 샘플 중 false positive가 얼마나 되는가?
- threshold를 올렸을 때 precision이 안정적으로 상승하는가?
- BSD35k에서 retain ratio가 비정상적으로 치우치지 않는가?

---

## 2. 모델별 핵심 성능 비교

| 모델 | 대표 결과 | 장점 | 핵심 한계 |
|---|---:|---|---|
| Simple Classification, 1-layer | acc 0.5757, macro F1 0.2834 | 가장 단순하고 해석 쉬움 | confidence 4로 강한 collapse, class 1/5 분리 약함 |
| Simple Regression, 1-layer | MAE 0.5769, rounded acc 0.5123 | ordinal score를 직접 생성 | MAE가 majority-class proxy보다도 불리, 평균 근처 collapse |
| Deep/5-class Classification v2 | MAE 0.5087, acc 0.5809, QWK 0.3375, macro F1 0.3062 | 5-class score와 ranking signal 보유 | class 1 recall 0, class 5 recall 0.139, pred 4 과다 |
| Binary Filter v3 | F1 0.7822, AUC-PR 0.8260, raw ECE 0.0304 | 실제 filtering task와 label 정의가 일치 | ranking 성능은 5-class expected score보다 낮음 |
| Ensemble v4 | F1 0.7853, AUC-PR 0.8421 | 가장 좋은 ranking/filtering score, threshold 운용 우수 | F1 개선폭은 작고, score는 확률이 아니라 rank 기반임 |

v4의 강점은 "5-class를 더 정확히 맞힌다"가 아니다. **binary model의 task alignment와 5-class expected score의 ordinal ranking signal을 결합해, high-confidence sample 순위를 더 안정적으로 만든다**는 점이다.

---

## 3. Confusion matrix 분석

### 3.1 Simple Classification Model

1-layer classification model의 best setting은 `C_audio_text_class_dropout_0.5`이며 `accuracy=0.5757`, `macro F1=0.2834`이다.

Confusion matrix는 label을 confidence 1-5로 복원하면 다음과 같다.

| true \ pred | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 10 | 11 | 0 |
| 2 | 0 | 11 | 39 | 100 | 0 |
| 3 | 0 | 0 | 180 | 476 | 0 |
| 4 | 0 | 4 | 142 | 1049 | 15 |
| 5 | 0 | 0 | 3 | 130 | 22 |

Prediction distribution:

| predicted confidence | 비율 |
|---:|---:|
| 2 | 0.68% |
| 3 | 17.06% |
| 4 | 80.57% |
| 5 | 1.69% |

해석:

- class 1 recall은 0이다.
- class 2 recall은 0.0733으로 사실상 collapse이다.
- class 5 recall은 0.1419로 낮다.
- 전체 예측의 80.57%가 confidence 4이다.
- confidence 3 샘플의 476/656이 4로 올라간다.

즉 simple classification은 accuracy만 보면 majority baseline보다 조금 좋아 보이지만, 실제로는 **confidence 4 majority collapse가 강하다**. BSD35k filtering에서 이 모델을 쓰면 confidence 3 샘플을 high-confidence로 과하게 통과시킬 위험이 크다.

### 3.2 Simple Regression Model

1-layer regression best setting은 `C_audio_text_class_dropout_0.5`이며 `MAE=0.5769`, `rounded accuracy=0.5123`이다. Regression output을 1-5로 clip 후 반올림한 confusion matrix는 다음과 같다.

| true \ pred | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 7 | 8 | 0 |
| 2 | 0 | 1 | 118 | 47 | 0 |
| 3 | 0 | 2 | 364 | 290 | 0 |
| 4 | 0 | 4 | 426 | 749 | 9 |
| 5 | 0 | 0 | 19 | 139 | 9 |

Row recall:

| class | recall |
|---:|---:|
| 1 | 0.0000 |
| 2 | 0.0060 |
| 3 | 0.5549 |
| 4 | 0.6305 |
| 5 | 0.0539 |

해석:

- regression은 ordinal structure를 반영할 수 있다는 이론적 장점이 있지만, 이 실험에서는 평균 근처인 3/4로 강하게 수렴한다.
- pred 3 비율은 42.61%, pred 4 비율은 56.25%이며 pred 5는 0.82%뿐이다.
- class 1/2/5 분리 능력은 classification보다도 불안정하다.
- high-confidence를 `rounded pred >= 4`로 보면 precision은 0.7242, recall은 0.6686이지만, 이 값은 "연속 score가 좋은 ranking signal"이라는 증거라기보다 3/4 경계에 눌린 결과에 가깝다.

특히 MAE 0.5769는 v2의 majority class 4 baseline MAE 0.5360보다 나쁘다. 따라서 현재 simple regression은 BSD35k threshold filtering에 주 모델로 쓰기 어렵다.

### 3.3 Deep / v2 5-class Classification Model

v2 best는 `stage2_baseline_emd`이고 결과는 다음과 같다.

| metric | value |
|---|---:|
| MAE | 0.5087 |
| Spearman | 0.4781 |
| Accuracy | 0.5809 |
| QWK | 0.3375 |
| Macro F1 | 0.3062 |

Confusion matrix:

| true \ pred | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 13 | 54 | 39 | 0 |
| 2 | 0 | 91 | 356 | 302 | 0 |
| 3 | 0 | 64 | 1203 | 2011 | 2 |
| 4 | 0 | 44 | 960 | 4962 | 79 |
| 5 | 0 | 2 | 37 | 629 | 108 |

Row recall:

| class | recall |
|---:|---:|
| 1 | 0.0000 |
| 2 | 0.1215 |
| 3 | 0.3668 |
| 4 | 0.8208 |
| 5 | 0.1392 |

Prediction distribution:

| predicted confidence | 비율 |
|---:|---:|
| 1 | 0.00% |
| 2 | 1.95% |
| 3 | 23.82% |
| 4 | 72.50% |
| 5 | 1.73% |

해석:

- simple model보다는 MAE, QWK, Spearman이 개선되었다.
- 그러나 여전히 class 1은 완전히 무시된다.
- class 5는 support가 7% 정도 있음에도 recall 0.1392에 그친다.
- confidence 3의 2013/3280이 4 또는 5로 올라간다.
- confidence 5의 629/776이 4로 내려간다. 즉 5를 "최상위 high-confidence"로 분리하는 능력은 약하다.
- 전체 예측의 72.50%가 confidence 4라서 majority 4 bias가 여전히 강하다.

하지만 v2를 완전히 버리면 안 된다. 이유는 `predicted class`는 불안정해도 **expected confidence score와 P4+P5 ranking signal은 꽤 강하기 때문**이다. 실제 binary filtering으로 변환했을 때:

| 5-class score 사용 방식 | precision | recall | F1 | AUC-PR |
|---|---:|---:|---:|---:|
| expected score >= 4.0 | 0.9318 | 0.2365 | 0.3772 | 0.8374 |
| P4+P5 >= 0.5 | 0.7230 | 0.8251 | 0.7707 | 0.8305 |

`expected score >= 4.0`은 recall이 너무 낮지만 precision 0.9318이다. 즉 v2는 5-class hard label classifier로는 부족하지만, **엄격한 high-confidence ranking score로는 가치가 있다**.

### 3.4 Binary Confidence Filter v3

v3는 confidence 1/2/3을 0, confidence 4/5를 1로 재정의한 모델이다. 실제 목적이 "좋은 샘플 vs 아닌 샘플"에 가깝기 때문에 task definition은 가장 잘 맞는다.

| setting | threshold | precision | recall | F1 | AUC-PR |
|---|---:|---:|---:|---:|---:|
| F1-optimal | 0.400 | 0.6904 | 0.9021 | 0.7822 | 0.8260 |
| default | 0.500 | 0.7343 | 0.8131 | 0.7717 | 0.8260 |
| precision-oriented recall>=0.7 | 0.585 | 0.7723 | 0.7000 | 0.7344 | 0.8260 |
| strict | 0.700 | 0.8237 | 0.5419 | 0.6537 | 0.8260 |
| very strict | 0.800 | 0.8668 | 0.3939 | 0.5417 | 0.8260 |

Calibration:

| calibration | ECE |
|---|---:|
| raw probability | 0.0304 |
| Platt scaling | 0.0234 |
| isotonic | 0.0000 diagnostic only |

해석:

- Binary target은 실제 filtering 목적과 직접 일치한다.
- raw ECE 0.0304는 나쁘지 않고, Platt scaling으로 0.0234까지 개선된다.
- 그러나 AUC-PR 0.8260은 5-class expected score의 0.8374보다 낮다.
- 즉 binary model은 threshold classification에는 강하지만, sample ranking signal은 5-class expected score보다 약하다.

BSD35k-CS 적용 시:

| threshold | retained ratio | expected precision |
|---:|---:|---:|
| 0.5 | 62.96% | 0.7343 |
| 0.585 | 43.59% | 0.7723 |
| 0.7 | 22.35% | 0.8237 |
| 0.8 | 9.53% | 0.8668 |
| 0.9 | 2.13% | 0.9084 |

v3 단독은 나쁜 모델이 아니다. 오히려 "binary filter만 쓸 것인가?"라는 질문에는 가장 자연스러운 후보이다. 하지만 BSD35k에서 precision을 높이기 위해 threshold를 올리면 retain ratio가 너무 빠르게 줄어든다. 예를 들어 threshold 0.8에서 9.53%만 남는다.

### 3.5 Ensemble v4

v4는 v3 binary probability와 v2 5-class expected score를 결합한다. 가장 좋은 방식은 다음이다.

```text
rank_average_binary_score
= average(rank(binary_mlp_prob), rank(fiveclass_expected_score))
```

핵심 결과:

| method | threshold | precision | recall | F1 | AUC-PR | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|
| Binary MLP v3 | 0.400 | 0.6904 | 0.9021 | 0.7822 | 0.8260 | 0.7461 |
| 5-class expected score | 3.140 | 0.6655 | 0.9490 | 0.7823 | 0.8374 | 0.7523 |
| Rank average: binary + P45 | 0.205 | 0.6932 | 0.9029 | 0.7843 | 0.8390 | 0.7577 |
| **Rank average: binary + expected score** | **0.180** | **0.6859** | **0.9183** | **0.7853** | **0.8421** | **0.7593** |
| OOF logistic stacker | 0.365 | 0.6773 | 0.9317 | 0.7844 | 0.8404 | 0.7573 |

v4의 F1 개선은 작다.

```text
v3 F1 = 0.7822
v4 F1 = 0.7853
gain   = +0.0031
```

하지만 filtering 목적에서는 F1보다 ranking과 threshold behavior가 중요하다. 여기서 v4의 개선은 더 의미 있다.

```text
v3 AUC-PR = 0.8260
v4 AUC-PR = 0.8421
gain      = +0.0161
```

즉 v4는 "분류 정확도"를 크게 올렸다기보다, **샘플을 high-confidence 가능성 순으로 정렬하는 능력**을 개선했다. BSD35k처럼 label이 없는 데이터셋에서는 이 차이가 더 중요하다.

---

## 4. False high-confidence 위험 분석

실제 filtering에서 가장 위험한 오류는 low/mid-confidence sample을 high-confidence로 통과시키는 것이다. 즉 false positive high-confidence가 핵심 위험이다.

v3와 v4의 OOF binary confusion count를 비교하면 다음과 같다.

### v3 Binary MLP

| threshold | TP | FP | TN | FN | precision | recall | retained |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | 6153 | 2759 | 1376 | 668 | 0.6904 | 0.9021 | 0.8134 |
| 0.5 | 5546 | 2007 | 2128 | 1275 | 0.7343 | 0.8131 | 0.6894 |
| 0.7 | 3696 | 791 | 3344 | 3125 | 0.8237 | 0.5419 | 0.4095 |
| 0.8 | 2687 | 413 | 3722 | 4134 | 0.8668 | 0.3939 | 0.2829 |
| 0.9 | 1517 | 153 | 3982 | 5304 | 0.9084 | 0.2224 | 0.1524 |

### v4 Rank Average

| threshold | TP | FP | TN | FN | precision | recall | retained |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.18 | 6264 | 2869 | 1266 | 557 | 0.6859 | 0.9183 | 0.8336 |
| 0.5 | 4389 | 1070 | 3065 | 2432 | 0.8040 | 0.6435 | 0.4983 |
| 0.7 | 2791 | 376 | 3759 | 4030 | 0.8813 | 0.4092 | 0.2891 |
| 0.8 | 1933 | 149 | 3986 | 4888 | 0.9284 | 0.2834 | 0.1900 |
| 0.9 | 876 | 35 | 4100 | 5945 | 0.9616 | 0.1284 | 0.0832 |

중요한 차이:

- v3 threshold 0.7: FP 791, precision 0.8237
- v4 threshold 0.7: FP 376, precision 0.8813
- v3 threshold 0.8: FP 413, precision 0.8668
- v4 threshold 0.8: FP 149, precision 0.9284

v4는 threshold를 올렸을 때 false high-confidence가 더 빠르게 줄어든다. pseudo-label quality filtering에서는 이 특성이 매우 중요하다.

---

## 5. Calibration / Reliability 분석

### 5.1 Classification softmax의 문제

5-class classification은 softmax probability를 제공하지만, severe imbalance 상황에서는 softmax confidence가 곧 reliability를 의미하지 않는다. v2의 confusion matrix를 보면 class 4 예측이 72.50%에 달한다. 이는 모델이 "confidence 4라는 의미적 상태"를 정확히 배웠다기보다, label prior와 3/4 경계의 애매함에 눌린 결과일 수 있다.

특히 BSD35k 예측 분포는 다음과 같다.

| predicted class | n | rate |
|---:|---:|---:|
| 2 | 122 | 0.39% |
| 3 | 10,337 | 32.85% |
| 4 | 20,985 | 66.70% |
| 5 | 20 | 0.06% |

이 분포는 위험 신호다.

- confidence 1은 아예 예측되지 않는다.
- confidence 5는 BSD35k 31,464개 중 20개뿐이다.
- 대부분을 3/4로 나누며, 특히 4가 66.70%이다.

따라서 5-class hard label을 그대로 BSD35k confidence category로 쓰는 것은 권장하지 않는다. 다만 expected score는 hard label보다 부드러운 ordinal ranking signal이므로 filtering score의 일부로는 유용하다.

### 5.2 Regression score의 안정성

Regression은 이론적으로 ordinal structure를 자연스럽게 반영하고 threshold filtering에 유리할 수 있다. 그러나 현재 simple regression은 다음 문제가 있다.

- MAE 0.5769로 v2 5-class MAE 0.5087보다 나쁘다.
- majority class 4 baseline MAE 0.5360보다도 불리하다.
- rounded output이 3/4 근처에 몰린다.
- class 2 recall 0.0060, class 5 recall 0.0539로 tail 분리 실패가 심하다.

즉 현재 regression score는 "연속값이라서 threshold에 안정적"이라고 보기 어렵다. 오히려 평균 회귀 현상이 강해서, BSD35k에서 진짜 high-confidence tail을 안정적으로 구분하기 어렵다.

### 5.3 Binary classifier의 calibration

Binary v3는 task 자체가 filtering과 일치하며 calibration도 나쁘지 않다.

- raw ECE: 0.0304
- Platt scaling ECE: 0.0234

따라서 v3 probability는 해석 가능한 high-confidence probability로 쓸 수 있다. 하지만 ranking 성능은 v2 expected score보다 낮다. 이 때문에 v3 단독보다 v4 결합이 더 적합하다.

### 5.4 Ensemble이 reliability를 개선하는 방식

v4 rank average는 확률 calibration을 직접 개선하는 방식은 아니다. 대신 다음 두 신호의 약점을 상호 보완한다.

- binary model: filtering target과 직접 맞지만 ranking AUC-PR이 상대적으로 낮음
- 5-class expected score: 5-class hard prediction은 불안정하지만 ordinal ranking signal이 강함

Rank average는 score scale에 덜 민감하다. BSD35k처럼 distribution shift가 있을 수 있는 데이터에서 raw probability 평균보다 rank 기반 결합이 더 안전할 가능성이 있다. 특히 모델별 calibration scale이 다를 때 rank fusion은 과도한 softmax confidence에 덜 끌려간다.

---

## 6. Regression vs Classification trade-off

### Regression

장점:

- confidence 1-5가 ordinal variable이라는 점을 자연스럽게 반영한다.
- continuous score를 만들 수 있어 ranking과 threshold filtering에 이론적으로 적합하다.
- `score >= t` 형태의 inference가 단순하다.

단점:

- severe imbalance에서는 평균값 근처로 collapse되기 쉽다.
- 현재 실험에서는 3/4 근처로 수렴했다.
- tail class인 1/2/5를 분리하지 못한다.
- MAE가 majority baseline보다도 나빠 실제 score reliability가 부족하다.

판단:

> 현재 simple regression은 이론적 장점에도 불구하고 BSD35k filtering 주 모델로 부적합하다. regression을 쓰려면 ordinal-aware loss, pairwise ranking loss, quantile/regression calibration, class-balanced sampling 등이 추가로 필요하다.

### Classification

장점:

- confusion matrix 해석이 쉽다.
- class separation을 직접 학습한다.
- P4+P5, expected score, margin P45-P3 같은 다양한 confidence score를 만들 수 있다.

단점:

- imbalance에서 majority class 4 collapse가 강하다.
- class 1과 class 5를 잘 분리하지 못한다.
- softmax probability가 실제 reliability와 다를 수 있다.

판단:

> 5-class classification을 hard label predictor로 쓰는 것은 위험하지만, expected score와 P4+P5는 filtering score의 구성 요소로 가치가 있다.

### Binary filtering

장점:

- 실제 목적과 label definition이 직접 일치한다.
- high-confidence precision/recall을 직접 최적화할 수 있다.
- calibration 분석이 쉽다.

단점:

- confidence 4와 5를 구분하지 않는다.
- confidence 3과 4의 미세한 ordinal 경계를 버린다.
- v3 단독 AUC-PR은 5-class expected score보다 낮다.

판단:

> binary filter는 반드시 포함해야 하지만, 단독 사용보다는 5-class expected score와 ensemble하는 것이 더 안정적이다.

---

## 7. BSD35k 실제 filtering 관점의 최종 선택

### 7.1 왜 v4가 가장 적합한가?

v4를 선택하는 이유는 다음과 같다.

1. **Filtering 목적과 맞는 binary signal을 포함한다.**  
   v3 binary probability는 `confidence >= 4`라는 실제 high-confidence filtering 목표를 직접 학습했다.

2. **Ordinal ranking signal을 보존한다.**  
   v2 5-class model은 hard class로는 불안정하지만 expected score의 AUC-PR이 0.8374로 강하다.

3. **AUC-PR이 가장 높다.**  
   v4 rank average의 AUC-PR은 0.8421로, v3 binary 0.8260보다 높다. unlabeled BSD35k에서는 "정렬 능력"이 매우 중요하다.

4. **strict threshold에서 false high-confidence를 더 잘 줄인다.**  
   threshold 0.7 기준 v4 precision은 0.8813이고 v3는 0.8237이다. threshold 0.8 기준 v4는 0.9284, v3는 0.8668이다.

5. **BSD35k retain ratio가 더 실용적이다.**  
   v4 threshold 0.7은 BSD35k의 29.23%를 retain하면서 expected precision 0.8813을 제공한다. v3 threshold 0.7은 22.35% retain, precision 0.8237이다. 즉 v4는 더 많이 남기면서도 더 깨끗하다.

6. **majority class 4 hard prediction에 덜 의존한다.**  
   v4는 5-class predicted class를 그대로 쓰지 않고 expected score와 binary probability의 rank를 결합한다. 따라서 class 4 collapse의 영향을 줄인다.

### 7.2 최종 추천 inference pipeline

BSD35k에 대해 다음 pipeline을 추천한다.

```text
Input sample
  -> CLAP audio/text embedding + class/meta features
  -> v3 binary filter model
       binary_mlp_prob = P(confidence >= 4)
  -> v2 5-class confidence model
       prob_confidence_1..5
       expected_score = sum(c * P(c))
  -> rank fusion
       rank_binary = percentile_rank(binary_mlp_prob)
       rank_expected = percentile_rank(expected_score)
       final_filter_score = (rank_binary + rank_expected) / 2
  -> threshold
       final_filter_score >= T -> retain
       final_filter_score <  T -> drop or low-priority pool
```

추천 threshold:

| threshold | 사용 상황 |
|---:|---|
| 0.5 | 데이터 양이 중요하고 noise를 중간 정도만 줄이면 되는 경우 |
| 0.7 | 기본 추천값. quality와 retained size 균형이 가장 좋음 |
| 0.8 | downstream training seed를 깨끗하게 만들고 싶은 경우 |
| 0.9 | pseudo-label teacher 초기 seed처럼 precision이 recall보다 훨씬 중요한 경우 |

### 7.3 3단계 dataset construction 제안

단일 threshold보다 더 안정적인 운용은 다음과 같은 tiering이다.

| tier | 조건 | 용도 |
|---|---|---|
| Gold | v4 score >= 0.8 | pseudo-label seed, teacher training, high-trust subset |
| Silver | 0.7 <= v4 score < 0.8 | 일반 training에 사용하되 낮은 weight 적용 |
| Bronze | 0.5 <= v4 score < 0.7 | 보조 데이터, augmentation 또는 semi-supervised 후보 |
| Drop / Review | v4 score < 0.5 | noisy 가능성이 높으므로 제외 또는 낮은 우선순위 |

이 방식은 BSD35k 전체를 binary retain/drop으로만 자르지 않고, downstream 학습에서 sample weighting을 가능하게 한다.

예시:

```text
weight = 1.0  if score >= 0.8
weight = 0.7  if 0.7 <= score < 0.8
weight = 0.3  if 0.5 <= score < 0.7
weight = 0.0  if score < 0.5
```

---

## 8. 모델별 최종 판정

### Simple Classification

판정: **비추천**

이유:

- confidence 4 예측이 80.57%로 과도하다.
- class 1 recall 0, class 5 recall 0.1419이다.
- confidence 3을 4로 올리는 오류가 많아 false high-confidence 위험이 크다.
- validation accuracy 0.5757은 majority baseline 대비 충분한 근거가 아니다.

### Simple Regression

판정: **비추천**

이유:

- MAE 0.5769로 v2와 차이가 크며 majority class 4 MAE proxy보다도 나쁘다.
- 3/4 평균값 근처 collapse가 강하다.
- class 2, 5 recall이 거의 없다.
- continuous score라는 장점이 실제로는 threshold 안정성으로 이어지지 않았다.

### Deep / v2 5-class Classification

판정: **단독 사용 비추천, ensemble 구성 요소로 추천**

이유:

- hard class prediction은 confidence 4 과다 예측과 tail class collapse가 크다.
- 그러나 expected score와 P4+P5는 high-confidence ranking signal로 유효하다.
- `expected score >= 4.0`은 precision 0.9318로 매우 엄격한 필터 역할이 가능하다.

### Binary Confidence Filter v3

판정: **단독으로도 사용 가능하지만 최종 추천은 아님**

이유:

- 실제 목적과 label 구조가 맞다.
- calibration도 raw ECE 0.0304로 준수하다.
- 하지만 AUC-PR이 v4보다 낮고, strict threshold에서 retained ratio와 precision trade-off가 v4보다 불리하다.

### Ensemble v4

판정: **최종 추천**

이유:

- AUC-PR 최고: 0.8421
- threshold 0.7에서 expected precision 0.8813
- threshold 0.8에서 expected precision 0.9284
- BSD35k retained ratio가 실용적이다.
- binary task alignment와 ordinal expected score를 동시에 활용한다.
- hard 5-class confusion의 class 4 collapse에 직접 종속되지 않는다.

---

## 9. 논문식 최종 결론

본 실험에서 가장 높은 단순 validation accuracy를 가진 모델을 선택하는 것은 부적절하다. BSD10k confidence label은 confidence 4가 약 55%를 차지하는 long-tail distribution이며, majority class 4 예측만으로도 약 55% accuracy가 가능하다. 따라서 모델 선택 기준은 accuracy가 아니라 high-confidence filtering에서의 precision, threshold stability, minority behavior, 3/4 boundary confusion, calibration reliability가 되어야 한다.

Simple classification과 simple regression은 모두 confidence 3/4 근처로 collapse되는 경향이 강하다. Simple classification은 confidence 4를 80% 이상 예측하며, regression은 continuous score를 제공함에도 평균값 근처 수렴으로 인해 class 1/2/5를 거의 분리하지 못한다. 따라서 두 모델은 BSD35k filtering 주 모델로 부적합하다.

v2 5-class classification은 MAE, QWK, Spearman 측면에서 단순 모델보다 개선되었지만, confusion matrix상 class 1 recall 0, class 5 recall 0.1392이며 predicted class 4 비율이 72.50%이다. 그러므로 5-class hard prediction을 그대로 confidence category로 쓰는 것은 위험하다. 그러나 expected confidence score는 high-confidence ranking signal로 유의미하며, binary conversion에서 AUC-PR 0.8374를 보인다.

v3 binary filter는 실제 목적과 가장 직접적으로 일치하고 calibration도 준수하다. 하지만 단독 ranking AUC-PR은 0.8260으로 v2 expected score보다 낮다. 이 둘의 장점을 결합한 v4 rank-average ensemble은 AUC-PR 0.8421로 가장 높고, strict threshold에서 false high-confidence를 가장 안정적으로 줄인다.

따라서 BSD35k confidence label이 없는 실제 deployment 환경에서는 **v4 rank-average ensemble을 최종 filtering score로 사용**하는 것이 가장 합리적이다. 기본 threshold는 0.7을 추천하며, 이 경우 BSD35k-CS에서 약 29.23%를 retain하고 OOF 기준 expected precision은 0.8813이다. 더 엄격한 pseudo-label seed가 필요하면 threshold 0.8을 사용해 약 19.29% retain, expected precision 0.9284를 목표로 하는 것이 좋다.

최종적으로 이 문제의 답은 "가장 accuracy가 높은 모델"이 아니라:

> **binary confidence filter의 task alignment와 5-class expected score의 ordinal ranking을 결합한 v4 ensemble이, BSD35k pseudo-label filtering / dataset quality filtering 목적에 가장 안정적인 선택이다.**

---

## 10. 추가 정정: downstream DCASE baseline 실험 기준에서는 v4가 최선이 아니다

위 결론은 **BSD10k confidence label 기준으로 high-confidence filtering score 자체를 평가한 결론**이다. 그러나 `baseline_confidnce_train/03_v4_rank_average_baseline_train.ipynb`의 downstream 결과를 함께 보면, 최종 DCASE baseline 학습 성능 관점에서는 결론을 더 조심스럽게 수정해야 한다.

Downstream 실험의 평균 결과는 다음과 같다.

| experiment | filter | retained ratio | accuracy | macro accuracy | hierarchical accuracy | hierarchical F1 |
|---|---|---:|---:|---:|---:|---:|
| v2_5class | pred_ge_2 | 0.9997 | 80.8212 | 75.1055 | 80.0664 | 79.1033 |
| 517_regression | pred_ge_3 | 0.9305 | 80.1551 | 74.0085 | 79.5320 | 78.4292 |
| 517_mlp_classification | pred_ge_3 | 0.9948 | 80.3923 | 74.1297 | 79.4763 | 78.6020 |
| v3_binary | pred_ge_2 | 0.9242 | 80.0000 | 73.6287 | 79.2928 | 78.0566 |
| v4_rank_average | pred_ge_2 | 0.7618 | 77.4635 | 70.5533 | 76.8046 | 75.2310 |
| v4_rank_average | pred_ge_3 | 0.4990 | 71.3139 | 64.2862 | 71.5445 | 69.7921 |
| v4_rank_average | pred_ge_4 | 0.2385 | 59.0237 | 47.9873 | 58.9915 | 63.5123 |

이 결과는 중요한 반례다. v4는 confidence filtering score 자체의 AUC-PR은 가장 좋았지만, downstream DCASE baseline 학습에서는 5개 실험 중 하위권이다. 특히 v4는 같은 `pred_ge_2/3/4` 형식으로 변환했을 때 train pool을 훨씬 더 많이 제거한다.

| filter | v4 retained ratio | 비교되는 상위 실험 retained ratio |
|---|---:|---:|
| pred_ge_2 | 0.7618 | v2 pred_ge_2: 0.9997 |
| pred_ge_3 | 0.4990 | regression pred_ge_3: 0.9305 |
| pred_ge_4 | 0.2385 | simple classification pred_ge_4: 0.8193 |

따라서 downstream 성능 하락은 단순히 "v4 score가 나쁘다"라기보다, **DCASE baseline classifier가 데이터 양 감소에 매우 민감하고, v4의 rank-based filtering이 너무 공격적으로 train distribution을 줄였기 때문**으로 해석하는 것이 맞다.

### 수정된 결론

목적에 따라 최종 선택을 분리해야 한다.

| 최종 목적 | 추천 모델 |
|---|---|
| BSD35k에서 confidence가 높은 subset을 보수적으로 고르기 | v4 rank-average ensemble |
| DCASE downstream classifier 성능을 최대로 유지하기 | v2_5class `pred_ge_2` 또는 regression/simple `pred_ge_3` |
| pseudo-label seed처럼 precision이 매우 중요한 작은 clean set 만들기 | v4 threshold 0.7-0.8 |
| training data를 많이 유지하면서 약한 noise만 제거하기 | v2_5class `pred_ge_2`, v3_binary `pred_ge_2` |

즉, **v4는 "clean subset selector"로는 강하지만, "downstream DCASE baseline 성능을 가장 높이는 train-pool filter"로는 현재 증거상 최선이 아니다.**

### 추가로 해야 할 실험

이 문제는 BSD35k에 confidence label이 없기 때문에 BSD35k에서 직접 confidence-filter 성능을 검증할 수 없다. 따라서 모델 선택은 두 단계로 해야 한다.

1. **BSD10k intrinsic confidence validation**  
   confidence label이 있는 BSD10k에서 AUC-PR, precision@retention, calibration, confusion matrix를 평가한다.

2. **BSD10k downstream proxy validation**  
   같은 BSD10k train pool을 filtering하고, held-out final_test에서 DCASE class/hierarchical 성능을 평가한다.

그 뒤 BSD35k에는 선택된 filter를 적용한다. BSD35k 자체에서는 confidence label이 없으므로 직접 검증이 아니라, pseudo-label training 후 downstream validation 성능으로 간접 검증해야 한다.

현재 downstream 결과까지 포함하면 실용적 선택은 다음에 더 가깝다.

> **BSD35k에서 무조건 많이 버리는 high-confidence filtering을 할 것이 아니라, 우선 v2_5class `pred_ge_2` 또는 v3_binary `pred_ge_2`처럼 retained ratio가 높은 약한 필터를 기본값으로 두고, v4는 별도의 high-precision seed subset 생성용으로 사용하는 것이 더 안전하다.**
