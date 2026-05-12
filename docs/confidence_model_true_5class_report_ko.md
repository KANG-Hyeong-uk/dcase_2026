# BSD10k/BSD35k True Confidence 1-5 분류기 보고서

작성일: 2026-05-10

## 먼저 정정

이전의 `confidence >= 4` 실험은 정확히 말하면 confidence 분류기가 아니라 high-confidence binary classifier였다.

```text
confidence >= 4 여부를 분류하는 모델
```

하지만 원래 목표는 BSD10k의 `confidence` 값 자체를 예측하는 confidence classifier다.

따라서 이번 실험에서는 target을 다시 원래대로 두었다.

```text
target = confidence class 1, 2, 3, 4, 5
```

즉 이번 보고서의 모델은 진짜 confidence 1-5 분류기다.

## 데이터와 입력

학습 데이터:

```text
BSD10k usable rows: 10,956
```

confidence 분포:

| confidence | n | rate |
|---:|---:|---:|
| 1 | 106 | 0.97% |
| 2 | 749 | 6.84% |
| 3 | 3,280 | 29.94% |
| 4 | 6,045 | 55.18% |
| 5 | 776 | 7.08% |

입력 feature:

```text
CLAP audio embedding
CLAP text embedding
class one-hot
top-class one-hot
metadata numeric
```

metadata numeric:

```text
title_chars
tag_count
description_chars
has_description
```

## 평가 지표

confidence는 1-5 순서가 있는 label이다.

따라서 exact accuracy만 보면 부족하다. 예를 들어 true 4를 pred 3으로 맞힌 것과 pred 1로 맞힌 것은 같은 오답이 아니다.

이번 보고서에서는 다음 지표를 같이 본다.

```text
MAE: expected confidence score와 true confidence 사이의 평균 절대 오차
Accuracy: argmax class가 정확히 맞은 비율
Macro F1: minority confidence class까지 고려한 F1
Quadratic weighted kappa: ordinal label에서 예측 순서가 얼마나 맞는지 보는 지표
```

최종 best 선택 기준은 `MAE`다.

이유:

```text
BSD35k-CS에 적용할 때 hard class 하나만 쓰기보다 predicted_confidence_score를 sample weighting/filtering에 쓰는 것이 더 자연스럽기 때문이다.
```

## 실험한 모델

실험 스크립트:

```text
confidence_model_true_5class_experiments.py
```

비교한 모델:

```text
softmax_ce_no_weights
softmax_ce_weighted
multitask_softmax_reg
ordinal_cumulative
two_tower_softmax
two_tower_multitask
two_tower_ordinal
ensemble_avg_softmax_ordinal_tower
ensemble_avg_best_ordinal_towers
stacked_class_logistic
stacked_score_ridge
```

## 전체 결과

| experiment | model type | MAE | Accuracy | Macro F1 | QWK |
|---|---|---:|---:|---:|---:|
| ensemble_avg_best_ordinal_towers | ensemble_average | 0.5088 | 0.5888 | 0.3339 | 0.3768 |
| stacked_score_ridge | score_stacking | 0.5093 | 0.5806 | 0.2674 | 0.3761 |
| ensemble_avg_softmax_ordinal_tower | ensemble_average | 0.5094 | 0.5882 | 0.3354 | 0.3801 |
| two_tower_softmax | two_tower_softmax | 0.5120 | 0.5736 | 0.3244 | 0.3658 |
| stacked_class_logistic | ensemble_stacking | 0.5126 | 0.5942 | 0.3218 | 0.3685 |
| two_tower_ordinal | two_tower_ordinal | 0.5151 | 0.5779 | 0.3338 | 0.3839 |
| softmax_ce_no_weights | softmax | 0.5161 | 0.5759 | 0.3347 | 0.3758 |
| ordinal_cumulative | ordinal | 0.5166 | 0.5821 | 0.3286 | 0.3648 |
| multitask_softmax_reg | multitask_softmax_reg | 0.5285 | 0.5741 | 0.3335 | 0.3685 |
| two_tower_multitask | two_tower_multitask | 0.5304 | 0.5800 | 0.3260 | 0.3561 |
| softmax_ce_weighted | softmax | 0.5628 | 0.5124 | 0.3581 | 0.4161 |

## 최종 Best 모델

최종 best:

```text
ensemble_avg_best_ordinal_towers
```

구성:

```text
ordinal_cumulative
two_tower_multitask
two_tower_ordinal
```

방법:

```text
세 모델의 confidence class probability를 평균 ensemble
최종 class = argmax(mean probability)
최종 score = sum(P(confidence=k) * k), k=1..5
```

성능:

| metric | value |
|---|---:|
| MAE | 0.5088 |
| Accuracy | 0.5888 |
| Macro F1 | 0.3339 |
| Quadratic weighted kappa | 0.3768 |

기존 `confidence_model_mlp_clean.ipynb` 결과와 비교:

| model | MAE | Accuracy | Macro F1 |
|---|---:|---:|---:|
| previous clean MLP | 0.5549 | 0.5155 | 0.3499 |
| true 5-class best | 0.5088 | 0.5888 | 0.3339 |
| improvement | -0.0461 | +0.0733 | -0.0160 |

해석:

```text
새 모델은 expected confidence score MAE와 exact accuracy를 크게 개선했다.
다만 macro F1은 약간 낮아졌다.
confidence 1, 2, 5처럼 데이터가 적은 class를 더 적극적으로 맞히려면 class weighting이 도움이 되지만,
그 경우 MAE와 accuracy가 크게 나빠진다.
```

## Confusion Matrix

최종 best 모델의 BSD10k OOF confusion matrix:

| true \ pred | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| true 1 | 0 | 9 | 61 | 36 | 0 |
| true 2 | 0 | 106 | 386 | 255 | 2 |
| true 3 | 0 | 67 | 1,422 | 1,780 | 11 |
| true 4 | 0 | 49 | 1,093 | 4,764 | 139 |
| true 5 | 0 | 1 | 46 | 570 | 159 |

예측 class 분포:

| predicted confidence | n | rate |
|---:|---:|---:|
| 2 | 232 | 2.12% |
| 3 | 3,008 | 27.46% |
| 4 | 7,405 | 67.59% |
| 5 | 311 | 2.84% |

해석:

```text
모델은 confidence 3과 4를 가장 많이 예측한다.
confidence 1은 매우 적은 class라 거의 직접 예측하지 못한다.
이는 데이터 분포가 confidence 4에 강하게 치우쳐 있고, confidence 1 sample이 106개뿐이기 때문이다.
```

## BSD35k-CS 적용 결과

최종 BSD35k-CS 예측 파일:

```text
outputs/confidence_model_true_5class/predictions/BSD35k-CS_predicted_true_5class_ensemble_avg_best_ordinal_towers.csv
```

이 파일은 BSD35k-CS 전체 sample에 대해 confidence 1-5를 예측한 결과다.

핵심 컬럼:

```text
predicted_confidence_class
predicted_confidence_score
prob_confidence_1
prob_confidence_2
prob_confidence_3
prob_confidence_4
prob_confidence_5
```

BSD35k-CS 예측 요약:

| item | value |
|---|---:|
| rows | 31,464 |
| mean predicted confidence score | 3.4776 |
| min predicted score | 2.5613 |
| median predicted score | 3.4584 |
| max predicted score | 4.5068 |

예측 class 분포:

| predicted confidence | n |
|---:|---:|
| 2 | 22 |
| 3 | 10,988 |
| 4 | 20,438 |
| 5 | 16 |

평균 class probability:

| class | mean probability |
|---:|---:|
| confidence 1 | 0.0269 |
| confidence 2 | 0.0629 |
| confidence 3 | 0.3605 |
| confidence 4 | 0.5050 |
| confidence 5 | 0.0447 |

해석:

```text
BSD35k-CS는 모델 기준으로 대부분 confidence 3 또는 4로 분류된다.
hard class만 보면 confidence 4가 가장 많지만, score 평균은 3.48이다.
따라서 sample weighting에는 hard class보다 predicted_confidence_score를 쓰는 것이 더 안정적이다.
```

## 최종 사용 추천

BSD35k-CS를 confidence classifier로 분류한 결과를 쓰려면:

```text
predicted_confidence_class
```

를 사용한다.

BSD35k-CS sample weighting이나 filtering에 쓰려면:

```text
predicted_confidence_score
```

를 사용하는 것이 더 좋다.

예:

```text
confidence score >= 4.0       엄격한 high-confidence subset
confidence score >= 3.5       비교적 신뢰 가능한 subset
confidence score as weight    soft weighting
```

## 이번 정정의 핵심

이전 binary classifier:

```text
confidence >= 4인지 아닌지만 예측
```

이번 true confidence classifier:

```text
confidence 1, 2, 3, 4, 5 전체를 예측
```

따라서 사용해야 할 최종 파일은 binary 결과 파일이 아니라 아래 파일이다.

```text
outputs/confidence_model_true_5class/predictions/BSD35k-CS_predicted_true_5class_ensemble_avg_best_ordinal_towers.csv
```

## 한 줄 결론

네가 원한 confidence 분류기는 `ensemble_avg_best_ordinal_towers`가 현재 best다.

BSD10k OOF 기준 MAE 0.5088, accuracy 0.5888이며, BSD35k-CS 전체에 대해 confidence 1-5 class와 expected confidence score를 모두 예측했다.
