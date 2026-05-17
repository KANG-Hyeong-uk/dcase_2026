# confidence분류기 모델 5개 분석

작성일: 2026-05-17

## 1. 목표

confidence 라벨은 모델의 라벨링 신뢰도를 의미하며, 분포는 다음과 같다.

| confidence | 비율 |
|---:|---:|
| 1 | 1.0% |
| 2 | 6.9% |
| 3 | 30.1% |
| 4 | 55.0% |
| 5 | 7.0% |

현재 목적은 confidence 1~5를 완벽하게 맞히는 것이 아니라, 학습 데이터 필터링에 사용할 수 있도록 다음 기준을 잘 분류하는 것이다.

- `confidence >= 3`
- `confidence >= 4`

특히 `confidence >= 4`는 실제 high-confidence 데이터만 고르는 목적에 가깝고, `confidence >= 3`은 low-confidence인 1, 2를 제거하는 목적에 가깝다.

## 2. 분석 대상 모델 5개

이번 분석에서 비교한 confidence 분류 모델은 총 5개다.

| 번호 | 모델 | 성격 | 주요 출력 |
|---:|---|---|---|
| 1 | v2 5-class confidence model | confidence 1~5 분류 | predicted class, expected score, P1~P5 |
| 2 | v3 binary MLP | `confidence >= 4` 직접 이진 분류 | high-confidence probability |
| 3 | v4 rank average | v3 binary score와 v2 expected score 조합 | rank average filtering score |
| 4 | 5-17 MLP classification | confidence 5-class MLP | predicted class |
| 5 | 5-17 regression | confidence score regression | predicted continuous score |

## 3. 기존 실험 결과 요약

### 3.1 v2 5-class confidence model

v2는 confidence 1~5를 직접 분류하는 모델이다. 전체 5-class 성능은 완벽하지 않지만, expected score와 class probability를 활용하면 filtering score로 사용할 수 있다.

주요 결과:

| 지표 | 값 |
|---|---:|
| MAE | 0.5087 |
| accuracy | 0.5809 |
| macro F1 | 0.3062 |
| quadratic weighted kappa | 0.3375 |

`confidence >= 4` 관점에서는 다음과 같다.

| 방식 | precision | recall | F1 | retain |
|---|---:|---:|---:|---:|
| predicted class >= 4 | 0.7105 | 0.8471 | 0.7728 | 74.2% |
| expected score best-F1 | 0.6649 | 0.9506 | 0.7825 | 89.0% |
| P4+P5 best-F1 | 0.6748 | 0.9295 | 0.7819 | 85.8% |

해석:

- v2는 5-class 자체 accuracy보다 score/ranking 용도로 가치가 있다.
- `confidence >= 3` 필터에는 expected score 또는 P3+P4+P5가 유용하다.
- `confidence >= 4` 필터에는 단독 사용보다 v3와 결합했을 때 더 좋다.

### 3.2 v3 binary MLP

v3는 처음부터 다음 binary target으로 학습한 모델이다.

```text
target = 1 if confidence >= 4 else 0
```

즉, confidence 4, 5를 high-confidence로 보고 1, 2, 3을 low-confidence로 보는 모델이다.

`confidence >= 4` 기준 주요 결과:

| threshold | precision | recall | F1 | retain |
|---:|---:|---:|---:|---:|
| 0.4 | 0.6904 | 0.9021 | 0.7822 | 81.3% |
| 0.5 | 0.7343 | 0.8131 | 0.7717 | 68.9% |
| 0.585 | 0.7723 | 0.7000 | 0.7344 | 56.4% |
| 0.7 | 0.8237 | 0.5419 | 0.6537 | 41.0% |
| 0.8 | 0.8668 | 0.3939 | 0.5417 | 28.3% |
| 0.9 | 0.9084 | 0.2224 | 0.3573 | 15.2% |

해석:

- `confidence >= 4` 목적에는 v3가 직접적으로 잘 맞는다.
- threshold를 높이면 precision은 올라가지만 recall이 빠르게 떨어진다.
- v3 단독으로 쓸 경우 `0.5`는 균형형, `0.585` 이상은 precision 중심, `0.7` 이상은 strict filtering이다.

### 3.3 v4 rank average

v4는 새 audio/text 모델을 다시 학습한 것이 아니라, 기존 score를 조합한 meta filtering 방식이다.

사용한 핵심 score:

```text
rank_average_binary_score
= rank(v3 binary_mlp_prob)와 rank(v2 fiveclass expected score)의 평균
```

`confidence >= 4` 기준 주요 결과:

| threshold | precision | recall | F1 | retain |
|---:|---:|---:|---:|---:|
| 0.18 | 0.6859 | 0.9183 | 0.7853 | 83.4% |
| 0.5 | 0.8040 | 0.6435 | 0.7148 | 49.8% |
| 0.6 | 0.8409 | 0.5285 | 0.6491 | 39.1% |
| 0.7 | 0.8813 | 0.4092 | 0.5589 | 28.9% |
| 0.8 | 0.9284 | 0.2834 | 0.4342 | 19.0% |
| 0.9 | 0.9616 | 0.1284 | 0.2266 | 8.3% |

해석:

- `confidence >= 4` 목적에서는 v4가 가장 추천된다.
- F1 기준으로는 v3보다 약간 좋고, AUC-PR/ranking 관점에서는 더 분명히 좋다.
- 실제 filtering에서는 v4 threshold를 조절하는 방식이 가장 실용적이다.
- 기본 시작점은 threshold `0.5`, 더 깨끗한 subset이 필요하면 `0.7`을 추천한다.

### 3.4 5-17 MLP classification

5-17 MLP classification은 confidence 5-class를 직접 분류하는 실험이다.

best 결과:

| 모델 | accuracy | MAE | macro F1 |
|---|---:|---:|---:|
| C_audio_text_class_dropout_0.5 | 0.5757 | 0.4877 | 0.2834 |
| D_audio_text_class23_dropout_0.5 | 0.5780 | 0.4868 | 0.2786 |

binary filtering 관점:

| 목적 | 모델 | precision | recall | F1 | retain |
|---|---|---:|---:|---:|---:|
| original confidence >= 3 | MLP C | 0.9265 | 0.9980 | 0.9609 | 99.3% |
| original confidence >= 3 | MLP D | 0.9265 | 0.9980 | 0.9609 | 99.3% |
| original confidence >= 4 | MLP C | 0.6744 | 0.8908 | 0.7677 | 82.3% |
| original confidence >= 4 | MLP D | 0.6759 | 0.8908 | 0.7686 | 82.1% |

해석:

- `confidence >= 3`에서는 높은 F1이 나오지만, 이는 positive 비율이 너무 높기 때문에 생기는 착시가 있다.
- `confidence >= 4`에서는 v3/v4보다 낮다.
- 단독 best 모델로 쓰기보다는 5-class score 후보 중 하나로 볼 수 있다.

### 3.5 5-17 regression

5-17 regression은 confidence를 continuous score로 예측하는 실험이다.

best 결과:

| 모델 | MAE | MSE | accuracy |
|---|---:|---:|---:|
| C_audio_text_class_dropout_0.5 | 0.5769 | 0.5310 | 0.5123 |

해석:

- 기존 v2 5-class 모델보다 MAE와 accuracy가 낮다.
- threshold filtering에 바로 쓰기에는 근거가 약하다.
- 현재 기준에서는 메인 confidence 필터 모델로 추천하지 않는다.

## 4. 목적별 추천 모델

### 4.1 confidence >= 3 목적

`confidence >= 3`은 전체 데이터의 약 92.1%가 positive다. 따라서 이 기준은 high-confidence 분류라기보다 confidence 1, 2를 제거하는 low-confidence filtering 문제에 가깝다.

추천 방식:

1. v2 5-class/ordinal 모델 사용
2. expected score 또는 P3+P4+P5 사용
3. threshold로 confidence 1, 2만 제거

추천 모델:

```text
v2 5-class confidence model
또는
5-class/ordinal 모델의 P3+P4+P5 score
```

주의할 점:

- `confidence >= 3` binary classifier를 직접 학습하면 대부분을 positive로 예측해도 성능이 좋아 보일 수 있다.
- 이 경우 F1만 보면 안 되고, 1/2 제거율과 3/4/5 보존율을 같이 봐야 한다.

### 4.2 confidence >= 4 목적

`confidence >= 4`는 confidence 3을 걸러야 하므로 실제로 의미 있는 high-confidence filtering 문제다.

추천 모델:

```text
v4 rank average: binary_mlp_prob + fiveclass expected score
```

추천 threshold:

| 목적 | v4 threshold | 예상 성향 |
|---|---:|---|
| 많이 살리기 | 0.18 | recall 높음, retain 약 83% |
| 균형형 | 0.5 | precision 약 0.80, retain 약 50% |
| 고품질 subset | 0.7 | precision 약 0.88, retain 약 29% |
| 매우 엄격 | 0.9 | precision 약 0.96, retain 약 8% |

최초 실험에서는 `threshold = 0.5`를 기본값으로 추천한다.

## 5. 다음 실험 계획

다음 실험의 목적은 confidence filtering 모델이 실제 DCASE baseline 학습 성능을 개선하는지 확인하는 것이다.

실험 절차는 다음과 같다.

1. BSD10k 전체 데이터에서 먼저 20%를 고정 평가용 holdout set으로 분리한다.
2. 남은 80% 데이터에 대해 confidence filtering을 적용한다.
3. filtering된 데이터로 동일한 baseline 모델을 학습한다.
4. 모든 실험은 처음에 분리한 동일한 20% holdout set에서 평가한다.
5. confidence model 5개와 filtering 기준 3개를 조합하여 총 15회 실험한다.

중요한 점:

- holdout 20%는 어떤 filtering에도 사용하지 않는다.
- 모든 실험은 같은 holdout set으로 비교한다.
- baseline 모델 구조, epoch, optimizer, seed, augmentation 조건은 가능한 동일하게 유지한다.
- 비교 대상은 confidence 필터링 방식만 달라야 한다.

## 6. 사용할 confidence 모델 5개

후속 실험에서 사용할 confidence filtering 모델은 다음 5개다.

| 번호 | confidence 모델 | filtering score |
|---:|---|---|
| 1 | v2 5-class confidence model | expected score 또는 class probability |
| 2 | v3 binary MLP | predicted high-confidence probability |
| 3 | v4 rank average | rank_average_binary_score |
| 4 | 5-17 MLP classification | predicted confidence class 또는 class probability |
| 5 | 5-17 regression | predicted confidence score |

## 7. 적용할 데이터 분류 기준 3개

각 confidence 모델마다 다음 3가지 filtering 기준으로 데이터를 나눈다.

| 기준 | 사용할 데이터 | 의미 |
|---|---|---|
| confidence >= 2 | 2, 3, 4, 5 사용 | confidence 1만 제거 |
| confidence >= 3 | 3, 4, 5 사용 | low-confidence 1, 2 제거 |
| confidence >= 4 | 4, 5 사용 | high-confidence 데이터만 사용 |

## 8. 총 15회 실험 매트릭스

| 실험 번호 | confidence 모델 | filtering 기준 |
|---:|---|---|
| 1 | v2 | confidence >= 2 |
| 2 | v2 | confidence >= 3 |
| 3 | v2 | confidence >= 4 |
| 4 | v3 | confidence >= 2 |
| 5 | v3 | confidence >= 3 |
| 6 | v3 | confidence >= 4 |
| 7 | v4 | confidence >= 2 |
| 8 | v4 | confidence >= 3 |
| 9 | v4 | confidence >= 4 |
| 10 | 5-17 MLP classification | confidence >= 2 |
| 11 | 5-17 MLP classification | confidence >= 3 |
| 12 | 5-17 MLP classification | confidence >= 4 |
| 13 | 5-17 regression | confidence >= 2 |
| 14 | 5-17 regression | confidence >= 3 |
| 15 | 5-17 regression | confidence >= 4 |

## 9. 각 모델별 filtering 기준 구현 방법

### 9.1 v2

v2는 5-class 모델이므로 다음 방식을 사용할 수 있다.

```text
confidence >= 2: expected_score >= threshold_for_2 또는 P2+P3+P4+P5
confidence >= 3: expected_score >= threshold_for_3 또는 P3+P4+P5
confidence >= 4: expected_score >= threshold_for_4 또는 P4+P5
```

우선 실험에서는 단순하고 해석 가능한 expected score 기준을 사용한다.

### 9.2 v3

v3는 원래 `confidence >= 4` 전용 binary model이다.

따라서 `confidence >= 4` 실험에는 가장 자연스럽게 사용할 수 있다.

```text
confidence >= 4: predicted_high_confidence_prob >= threshold
```

단, v3는 `confidence >= 2`, `confidence >= 3`을 직접 학습한 모델이 아니므로 이 두 기준에서는 보조 비교군으로만 해석한다.

### 9.3 v4

v4는 v3 binary score와 v2 expected score를 조합한 filtering score다.

```text
rank_average_binary_score >= threshold
```

`confidence >= 4` 목적에서는 가장 중요한 실험 대상이다.

초기 threshold는 다음처럼 둔다.

```text
confidence >= 4: threshold = 0.5
```

15회 기본 실험이 끝난 후 v4 threshold sweep을 별도로 수행한다.

### 9.4 5-17 MLP classification

5-17 MLP classification은 predicted class를 기준으로 filtering한다.

```text
confidence >= 2: predicted_class >= 2
confidence >= 3: predicted_class >= 3
confidence >= 4: predicted_class >= 4
```

단, 내부 라벨이 0~4로 저장된 경우 원래 confidence 기준과 1 차이가 날 수 있으므로 반드시 원래 confidence 1~5로 복원한 뒤 filtering한다.

### 9.5 5-17 regression

5-17 regression은 predicted continuous score에 threshold를 적용한다.

```text
confidence >= 2: predicted_score >= 2.0
confidence >= 3: predicted_score >= 3.0
confidence >= 4: predicted_score >= 4.0
```

다만 기존 결과상 성능이 약하므로 메인 후보보다는 비교용 모델로 본다.

## 10. 평가 지표

각 실험마다 다음 값을 기록한다.

### 10.1 filtering 단계 지표

| 지표 | 의미 |
|---|---|
| retained samples | filtering 후 남은 학습 샘플 수 |
| retained ratio | 전체 train 후보 중 유지 비율 |
| predicted confidence distribution | filtering 후 confidence 분포 |
| confidence threshold | 사용한 threshold |

### 10.2 baseline 학습 성능 지표

DCASE baseline 모델 학습 후 동일 holdout 20%에서 다음 지표를 비교한다.

| 지표 | 의미 |
|---|---|
| validation/test loss | holdout loss |
| accuracy 또는 task metric | baseline task 성능 |
| macro F1 | class imbalance 대응 지표 |
| per-class metric | 특정 class 성능 하락 확인 |
| confusion matrix | filtering으로 인한 class bias 확인 |

## 11. v4 threshold 추가 실험 계획

15회 기본 실험이 끝나면, 가장 유력한 v4에 대해서 threshold를 조절한 추가 실험을 수행한다.

우선 후보 threshold:

| threshold | 목적 |
|---:|---|
| 0.18 | F1-optimal, 많이 살리는 설정 |
| 0.5 | 균형형 기본 설정 |
| 0.6 | precision 강화 |
| 0.7 | 고품질 subset |
| 0.8 | strict filtering |
| 0.9 | very strict filtering |

추가 실험에서는 특히 다음 질문을 확인한다.

1. v4 threshold를 높여 precision을 올리면 DCASE baseline 성능도 좋아지는가?
2. 너무 많이 버리면 데이터 수 부족으로 성능이 떨어지는가?
3. 최적 threshold는 confidence filtering F1 기준과 downstream baseline 성능 기준에서 같은가?
4. `confidence >= 4`로 매우 깨끗한 데이터만 쓰는 것이 좋은가, 아니면 `confidence >= 3`으로 더 많은 데이터를 쓰는 것이 좋은가?

## 12. 예상 결론 가설

현재 confidence 모델 성능만 기준으로 보면 다음 가설이 가장 그럴듯하다.

### 가설 1

`confidence >= 4` 필터링에서는 v4 rank average가 가장 좋은 downstream 성능을 낼 가능성이 높다.

이유:

- v3 binary MLP보다 ranking 성능이 좋다.
- v2 5-class expected score의 ordinal 정보를 함께 사용한다.
- threshold 조절로 retain ratio와 precision을 유연하게 바꿀 수 있다.

### 가설 2

`confidence >= 3` 필터링은 성능 향상 폭이 크지 않을 수 있다.

이유:

- 원래 데이터의 약 92%가 confidence 3 이상이다.
- 사실상 confidence 1, 2만 제거하는 실험이다.
- 데이터 품질 향상보다 데이터 수 유지 효과가 더 클 수 있다.

### 가설 3

`confidence >= 4`만 사용하면 데이터 품질은 좋아지지만, 데이터 수가 줄어 downstream 성능이 떨어질 수 있다.

이유:

- confidence 4, 5는 약 62% 정도다.
- threshold를 높이면 retained ratio가 50%, 30%, 10% 수준까지 줄어든다.
- baseline 모델이 충분한 데이터량을 필요로 한다면 너무 strict한 filtering은 손해일 수 있다.

## 13. 최종 추천 실행 순서

우선순위는 다음과 같다.

1. holdout 20%를 먼저 고정 분리한다.
2. full 80% train baseline을 가능하면 control로 1회 학습한다.
3. 5개 confidence 모델 x 3개 filtering 기준으로 15회 실험한다.
4. 각 실험의 retained ratio와 downstream 성능을 같이 기록한다.
5. 15회 결과에서 v4가 유리하면 v4 threshold sweep을 수행한다.
6. 최종 선택은 confidence 모델 자체 F1이 아니라 downstream baseline holdout 성능으로 결정한다.

현재 기준의 가장 유력한 후보는 다음과 같다.

```text
confidence >= 3 목적:
v2 5-class expected score 또는 P3+P4+P5

confidence >= 4 목적:
v4 rank_average_binary_score

v4 기본 threshold:
0.5

v4 strict threshold:
0.7
```
