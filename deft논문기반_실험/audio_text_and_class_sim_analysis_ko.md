# 01/02 Similarity 실험 결과 분석

분석 대상:

- `01_audio_text_sim_experiment.ipynb`
- `02_audio_class_sim_experiment.ipynb`

두 실험은 모두 모델 학습 없이 CLAP cosine similarity 하나를 confidence와 비교한다.

## 1. 실험 목적 정리

### 01 audio_text_sim

```text
audio_text_sim = cosine(audio_emb, text_emb)
```

질문:

```text
오디오와 title/tags/description 기반 text embedding이 잘 맞을수록 confidence가 높은가?
```

### 02 audio_class_sim

```text
audio_class_sim = cosine(audio_emb, assigned_class_emb)
```

질문:

```text
오디오와 metadata class embedding이 잘 맞을수록 confidence가 높은가?
```

두 실험 모두 같은 방식으로 평가했다.

- confidence별 Mean ± Std
- confidence별 Boxplot
- similarity를 5개 분위 구간으로 나눠 predicted confidence 1~5로 매핑
- 실제 confidence와 비교한 count table
- Accuracy / Precision / Recall / F1
- classification report

## 2. 데이터 분포 전제

두 실험 모두 사용 row 수는 10956개다.

confidence 분포는 다음과 같다.

```text
confidence 1: 106
confidence 2: 749
confidence 3: 3280
confidence 4: 6045
confidence 5: 776
```

중요한 점:

- confidence 4가 압도적으로 많다.
- confidence 1은 106개뿐이다.
- 따라서 similarity를 강제로 5구간으로 나눠 predicted confidence 1~5로 보는 평가는 매우 거칠다.
- exact 5-class accuracy보다, 높은 similarity 구간에 confidence 4/5 또는 5가 더 많이 모이는지를 함께 봐야 한다.

## 3. Confidence별 평균 similarity

### 01 audio_text_sim

| confidence | n | mean | std | median |
|---:|---:|---:|---:|---:|
| 1 | 106 | 0.4602 | 0.1525 | 0.4949 |
| 2 | 749 | 0.4694 | 0.1294 | 0.4912 |
| 3 | 3280 | 0.4707 | 0.1323 | 0.4931 |
| 4 | 6045 | 0.4912 | 0.1211 | 0.5124 |
| 5 | 776 | 0.5385 | 0.1018 | 0.5599 |

해석:

- confidence 1~3은 평균이 거의 비슷하다.
- confidence 4에서 조금 올라간다.
- confidence 5에서 확실히 올라간다.
- 즉 `audio_text_sim`은 confidence 5를 어느 정도 구분하는 신호가 있다.
- 하지만 confidence 1/2/3/4를 세밀하게 나누는 신호는 약하다.

### 02 audio_class_sim

| confidence | n | mean | std | median |
|---:|---:|---:|---:|---:|
| 1 | 106 | 0.0377 | 0.1317 | 0.0374 |
| 2 | 749 | 0.0661 | 0.1331 | 0.0668 |
| 3 | 3280 | 0.0744 | 0.1320 | 0.0741 |
| 4 | 6045 | 0.0780 | 0.1306 | 0.0750 |
| 5 | 776 | 0.1331 | 0.1353 | 0.1142 |

해석:

- confidence가 높아질수록 평균이 단조 증가한다.
- confidence 5가 특히 높다.
- confidence 5 - confidence 1 평균 차이는 `0.0953`으로, audio_text_sim의 `0.0783`보다 크다.
- confidence 5 - confidence 4 평균 차이도 `0.0550`으로, audio_text_sim의 약 `0.0473`보다 크다.
- 평균 기준으로는 `audio_class_sim`이 `audio_text_sim`보다 confidence ordering이 조금 더 명확하다.

## 4. Similarity 5-bin exact classification 성능

### Metric 비교

| feature | accuracy | macro precision | macro recall | macro F1 | weighted F1 |
|---|---:|---:|---:|---:|---:|
| audio_text_sim | 0.2110 | 0.2110 | 0.2398 | 0.1678 | 0.2542 |
| audio_class_sim | 0.2107 | 0.2107 | 0.2460 | 0.1677 | 0.2536 |

해석:

- 둘 다 exact 5-class classifier로는 성능이 낮다.
- accuracy는 약 21% 수준이다.
- macro F1도 약 0.168 수준이다.
- 둘 사이의 차이는 거의 없다.

중요:

```text
similarity를 단순히 5등분해서 confidence 1~5를 정확히 맞추는 방식은 잘 안 된다.
```

즉 이 feature들은 5-class confidence classifier가 아니라 ranking/filtering 보조 feature로 보는 것이 맞다.

## 5. 5-bin table 해석

5-bin table은 다음 방식이다.

```text
similarity 하위 20% -> predicted confidence 1
similarity 다음 20% -> predicted confidence 2
...
similarity 상위 20% -> predicted confidence 5
```

좋은 exact classifier라면 대각선이 커야 한다.

하지만 데이터가 confidence 4에 몰려 있어서 모든 bin에서 confidence 4가 많다.

따라서 더 중요한 것은:

```text
Q1에서 Q5로 갈수록 confidence 5 또는 confidence 4/5 비율이 증가하는가?
```

## 6. High-confidence enrichment

### 01 audio_text_sim

구간별 confidence 비율:

| predicted bin | conf1 | conf2 | conf3 | conf4 | conf5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.19% | 7.89% | 36.72% | 51.23% | 2.97% |
| 2 | 0.91% | 8.03% | 31.31% | 55.13% | 4.61% |
| 3 | 1.14% | 7.21% | 28.02% | 56.60% | 7.03% |
| 4 | 0.78% | 5.29% | 27.93% | 56.73% | 9.27% |
| 5 | 0.82% | 5.75% | 25.70% | 56.18% | 11.55% |

핵심 변화:

```text
confidence 5 rate:
Q1 2.97% -> Q5 11.55%
gain: +8.58%p

confidence 4/5 rate:
Q1 54.20% -> Q5 67.73%
gain: +13.53%p
```

해석:

- audio_text_sim이 높을수록 confidence 5 비율이 뚜렷하게 증가한다.
- confidence 4/5 비율도 의미 있게 증가한다.
- 따라서 audio_text_sim은 high-confidence ranking 보조 feature로 의미가 있다.
- 하지만 Q5에서도 confidence 3이 25.70%나 있고, confidence 4가 계속 대부분이라 단독 필터로는 약하다.

### 02 audio_class_sim

구간별 confidence 비율:

| predicted bin | conf1 | conf2 | conf3 | conf4 | conf5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.41% | 7.66% | 32.48% | 55.02% | 3.42% |
| 2 | 1.10% | 7.03% | 29.21% | 57.05% | 5.61% |
| 3 | 0.87% | 6.30% | 29.76% | 54.77% | 8.31% |
| 4 | 0.91% | 7.03% | 30.40% | 55.36% | 6.30% |
| 5 | 0.55% | 6.16% | 27.84% | 53.67% | 11.78% |

핵심 변화:

```text
confidence 5 rate:
Q1 3.42% -> Q5 11.78%
gain: +8.35%p

confidence 4/5 rate:
Q1 58.44% -> Q5 65.45%
gain: +7.01%p
```

해석:

- audio_class_sim도 confidence 5 비율은 잘 올린다.
- 하지만 confidence 4/5 전체 비율 증가는 audio_text_sim보다 작다.
- Q4에서 confidence 5 비율이 오히려 Q3보다 낮아지는 등, bin별 흐름이 아주 매끄럽지는 않다.

## 7. Classification report 해석

### audio_text_sim

```text
accuracy: 0.21
macro F1: 0.17
weighted F1: 0.25
```

confidence 5:

```text
precision: 0.12
recall: 0.33
F1: 0.17
```

해석:

- similarity 상위 bin을 confidence 5로 보면 실제 confidence 5 중 약 33%를 잡는다.
- 하지만 precision은 12%라서, confidence 5라고 예측한 것 대부분은 실제 4 또는 3이다.
- confidence 5를 “깨끗하게” 뽑는 기준으로는 약하다.

### audio_class_sim

```text
accuracy: 0.21
macro F1: 0.17
weighted F1: 0.25
```

confidence 5:

```text
precision: 0.12
recall: 0.33
F1: 0.17
```

해석:

- audio_text_sim과 거의 같다.
- confidence 5 recall은 비슷하지만 precision이 낮다.
- 단독으로 confidence 5를 선별하기에는 부족하다.

## 8. 01 vs 02 비교

### 평균 분포 기준

`audio_class_sim`이 조금 더 좋다.

```text
audio_text_sim confidence 5 - 1 mean gap: 약 0.0783
audio_class_sim confidence 5 - 1 mean gap: 약 0.0953
```

즉 confidence별 평균 separation은 class similarity 쪽이 더 크다.

### high-confidence enrichment 기준

`audio_text_sim`이 조금 더 좋다.

```text
audio_text_sim Q5-Q1 confidence 4/5 gain: +13.53%p
audio_class_sim Q5-Q1 confidence 4/5 gain: +7.01%p
```

confidence 5 gain은 거의 비슷하다.

```text
audio_text_sim Q5-Q1 confidence 5 gain: +8.58%p
audio_class_sim Q5-Q1 confidence 5 gain: +8.35%p
```

### 5-class metric 기준

둘 다 거의 같다.

```text
audio_text_sim accuracy: 0.2110, macro F1: 0.1678
audio_class_sim accuracy: 0.2107, macro F1: 0.1677
```

## 9. 최종 결론

### audio_text_sim

장점:

- similarity가 높아질수록 confidence 5 비율이 증가한다.
- Q1 -> Q5에서 confidence 4/5 비율 증가폭이 크다.
- high-confidence ranking 보조 feature로 의미가 있다.

한계:

- confidence 1~5 exact classification은 약하다.
- confidence 3/4/5가 많이 섞인다.
- Q5에서도 confidence 3이 25% 이상 존재한다.

판정:

```text
confidence ranking/filtering 보조 feature로 사용 가능
단독 confidence classifier로는 부적합
```

### audio_class_sim

장점:

- confidence별 평균 similarity가 단조 증가한다.
- confidence 5와 confidence 1/4의 평균 차이가 audio_text_sim보다 크다.
- confidence 5 비율은 상위 similarity bin에서 뚜렷하게 증가한다.

한계:

- high-confidence 4/5 enrichment는 audio_text_sim보다 약하다.
- exact 5-class metric은 audio_text_sim과 거의 동일하게 낮다.

판정:

```text
confidence 평균 ordering 보조 feature로 사용 가능
단독 confidence classifier로는 부적합
```

## 10. 현재까지 feature 우선순위

지금까지 01/02/03을 함께 보면:

```text
1순위: audio_assigned_label_margin
2순위: audio_text_sim
3순위: audio_class_sim
4순위: text_assigned_label_margin
보류: text_top1_top2_margin
제외 후보: audio_top1_top2_margin
```

다만 목적에 따라 달라진다.

### confidence 5를 더 많이 포함하는 high-score group을 만들고 싶다면

```text
audio_text_sim
audio_class_sim
audio_assigned_label_margin
```

셋 모두 후보가 된다.

### metadata label이 다른 class보다 잘 맞는지를 보고 싶다면

```text
audio_assigned_label_margin
```

이 가장 직접적인 feature다.

### confidence 1~5를 정확히 분류하고 싶다면

```text
세 feature 모두 단독으로는 부족하다.
```

따라서 다음 단계는 이 feature들을 단독으로 쓰기보다 조합해서 보는 것이 맞다.
