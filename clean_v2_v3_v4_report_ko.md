# Clean Label v2/v3/v4 실험 보고서

## 1. 실험 목적

이번 실험은 기존 confidence pipeline에서 입력 feature만 clean하게 바꾸었을 때 성능이 개선되는지 확인하기 위한 비교 실험이다.

변경한 입력:

```text
audio embedding 512 + text embedding 512 + class one-hot 23 = 1047 dims
```

제거한 입력:

```text
top_class one-hot
metadata length features
```

비교 원칙:

- v2, v3, v4의 학습 구조, seed, fold, optimizer, scheduler, threshold grid, stacking 방식은 기존 실험과 동일하게 유지한다.
- 추가된 것은 train/validation loss 시각화와 row-normalized confusion matrix뿐이다.
- 따라서 비교 포인트는 입력 feature 변경의 효과다.

## 2. v2 5-class 결과

| Model | Feature dim | MAE | Accuracy | QWK | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Original v2 EMD | 1056 | 0.5087 | 0.5809 | 0.3375 | 0.3062 |
| Clean v2 EMD | 1047 | 0.5118 | 0.5771 | 0.3243 | 0.3036 |
| Original v2 Ordinal Smoothing | 1056 | 0.5142 | 0.5800 | 0.3433 | 0.3139 |
| Clean v2 Ordinal Smoothing | 1047 | 0.5181 | 0.5740 | 0.3079 | 0.3062 |
| Original v2 CE | 1056 | 0.5145 | 0.5768 | 0.3299 | 0.3075 |
| Clean v2 CE | 1047 | 0.5169 | 0.5750 | 0.3048 | 0.3108 |

v2 해석:

- clean input은 MAE, Accuracy, QWK 기준으로 original보다 낮다.
- Macro F1은 CE에서 clean이 소폭 높지만, EMD와 ordinal smoothing에서는 떨어진다.
- 5-class ordinal confidence 예측에서는 top_class/meta feature가 노이즈라기보다 약한 calibration signal로 작동한 것으로 보인다.

## 3. v3 Binary 결과

Target:

```text
high confidence = confidence >= 4
```

| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC-PR | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original v3 F1-optimal | 0.400 | 0.6872 | 0.6904 | 0.9021 | 0.7822 | 0.8260 | 0.7461 |
| Clean v3 F1-optimal | 0.395 | 0.6899 | 0.6946 | 0.8959 | 0.7825 | 0.8190 | 0.7387 |
| Original v3 @ 0.5 | 0.500 | 0.7004 | 0.7343 | 0.8131 | 0.7717 | 0.8260 | 0.7461 |
| Clean v3 @ 0.5 | 0.500 | 0.6933 | 0.7270 | 0.8126 | 0.7674 | 0.8190 | 0.7387 |

v3 해석:

- F1-optimal threshold에서는 clean v3가 F1을 아주 조금 높인다.
- 그러나 AUC-PR과 AUC-ROC는 original v3가 더 높다.
- filtering score로는 단일 threshold 성능보다 ranking 품질이 중요하므로, clean v3가 더 낫다고 보기 어렵다.

## 4. v4 Score Stacking 결과

| Model | Best Method | Threshold | Precision | Recall | F1 | AUC-PR | AUC-ROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Original v4 | Rank average: binary + expected score | 0.180 | 0.6859 | 0.9183 | 0.7853 | 0.8421 | 0.7593 |
| Clean v4 | OOF logistic stacker | 0.360 | 0.6729 | 0.9355 | 0.7828 | 0.8354 | 0.7535 |

v4 해석:

- clean v4는 recall을 더 높이지만 precision과 AUC-PR이 낮다.
- expected F1도 original v4가 더 높다.
- confidence filtering의 목적이 BSD35k pseudo-label 품질 확보라면 precision/ranking 품질이 중요하므로 original v4가 더 적합하다.

## 5. BSD35k-CS Filtering Scenario

Original v4:

| Threshold | Retained samples | Retained ratio | Expected precision | Expected recall | Expected F1 |
|---:|---:|---:|---:|---:|---:|
| 0.180 | 26150 | 0.8311 | 0.6859 | 0.9183 | 0.7853 |
| 0.500 | 15566 | 0.4947 | 0.8040 | 0.6435 | 0.7148 |
| 0.600 | 12357 | 0.3927 | 0.8409 | 0.5285 | 0.6491 |
| 0.700 | 9197 | 0.2923 | 0.8813 | 0.4092 | 0.5589 |
| 0.800 | 6069 | 0.1929 | 0.9284 | 0.2834 | 0.4342 |
| 0.900 | 3008 | 0.0956 | 0.9616 | 0.1284 | 0.2266 |

Clean v4:

| Threshold | Retained samples | Retained ratio | Expected precision | Expected recall | Expected F1 |
|---:|---:|---:|---:|---:|---:|
| 0.360 | 27927 | 0.8876 | 0.6729 | 0.9355 | 0.7828 |
| 0.500 | 19823 | 0.6300 | 0.7224 | 0.8320 | 0.7733 |
| 0.600 | 12646 | 0.4019 | 0.7742 | 0.7149 | 0.7433 |
| 0.700 | 6131 | 0.1949 | 0.8297 | 0.5508 | 0.6621 |
| 0.800 | 1678 | 0.0533 | 0.9046 | 0.3140 | 0.4662 |
| 0.900 | 446 | 0.0142 | 0.9452 | 0.1340 | 0.2347 |

Scenario 해석:

- clean v4는 best threshold에서 더 많은 sample을 남긴다.
- 그러나 expected precision이 original보다 낮다.
- strict threshold 구간에서는 clean v4가 retained sample을 훨씬 적게 남긴다.
- downstream pseudo-label 학습에서는 이전 실험상 quality가 quantity보다 중요했으므로, original v4가 더 안정적인 선택이다.

## 6. 결론

clean input 실험의 결론은 다음과 같다.

```text
top_class와 metadata length를 제거한 clean input은 모델을 더 단순하게 만들지만,
confidence prediction/filtering 성능을 개선하지 못했다.
```

최종 판단:

- v2 5-class: original 우세
- v3 binary: F1은 거의 동률, ranking 품질은 original 우세
- v4 stacking: original 우세
- BSD35k filtering: original v4가 더 안정적

따라서 다음 실험은 clean input을 기본값으로 밀기보다, original v4 score를 유지하면서 scalar/tabular model을 비교하는 방향이 적절하다.

## 7. 다음 실험: Random Forest vs XGBoost

### 7.1 후보 모델 관점

Random Forest:

- bagging 기반이라 튜닝이 비교적 쉽고 안정적이다.
- feature importance 해석이 쉽다.
- 그러나 1000차원 이상의 dense embedding에는 약하다.
- class imbalance와 ordinal confidence boundary를 세밀하게 잡는 데 한계가 있다.
- probability calibration이 거칠어 threshold filtering score로 쓰기에는 불리할 수 있다.

XGBoost:

- boosting 기반이라 약한 scalar signal을 누적해서 잘 잡는다.
- class imbalance, nonlinear rule, feature interaction을 Random Forest보다 잘 다룰 가능성이 높다.
- learning rate, depth, regularization으로 overfit 제어가 가능하다.
- ranking/AUC-PR 중심 score model에 더 적합하다.
- 단, raw 1047-dim embedding 전체를 넣으면 과적합과 잡음 문제가 생길 수 있다.

### 7.2 Confidence 분류에 더 맞는 쪽

confidence filtering 목적에는 XGBoost가 Random Forest보다 더 적합할 가능성이 높다.

이유:

- 목표가 hard class prediction보다 high-confidence score ranking에 가깝다.
- v4에서 이미 scalar score 조합이 효과적이었다.
- XGBoost는 `binary_mlp_prob`, `fiveclass_score`, `P4+P5`, margin, entropy, class one-hot 같은 scalar/tabular feature interaction을 잘 학습한다.
- Random Forest는 baseline으로는 좋지만, precision-recall threshold curve를 정교하게 만드는 데는 XGBoost보다 약할 가능성이 크다.

### 7.3 권장 실험 설계

Raw embedding을 바로 tree에 넣는 실험은 우선순위를 낮춘다.

권장 입력:

```text
v2/v3/v4 scalar scores
+ agreement features
+ class one-hot 23
+ optional prototype/consistency scalar
```

권장 비교:

| Experiment | Model | Input | Target | Primary metric |
|---|---|---|---|---|
| T1 | RandomForestClassifier | scalar + class one-hot | confidence >= 4 | AUC-PR |
| T2 | XGBClassifier | scalar + class one-hot | confidence >= 4 | AUC-PR |
| T3 | RandomForestRegressor | scalar + class one-hot | confidence 1-5 | MAE/QWK |
| T4 | XGBRegressor | scalar + class one-hot | confidence 1-5 | MAE/QWK |

추천 우선순위:

1. XGBClassifier for binary high-confidence filtering
2. XGBRegressor for ordinal score
3. RandomForestClassifier as sanity baseline
4. RandomForestRegressor as lower-priority baseline

### 7.4 성공 기준

XGBoost/Random Forest 실험은 original v4를 이겨야 의미가 있다.

비교 기준:

- AUC-PR > 0.8421
- F1-optimal F1 > 0.7853
- precision-optimal @ recall>=0.7 precision > 0.7801
- BSD35k scenario에서 threshold 0.5/0.6 구간의 expected precision 개선

이 기준을 넘지 못하면 tree model은 최종 필터로 쓰기보다 diagnostic model 또는 ensemble 후보로만 둔다.

