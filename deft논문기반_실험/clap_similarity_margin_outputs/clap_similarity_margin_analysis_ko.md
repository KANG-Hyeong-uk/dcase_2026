# CLAP similarity margin 실험 결과 분석

## 1. 이 실험이 보려는 것

이 실험은 baseline 모델이 아니라 CLAP embedding만 사용한다.

각 sample에 대해 audio/text embedding을 모든 class embedding과 비교해서 다음 margin들을 계산했다.

```text
top1_top2_margin = top1_sim - top2_sim
assigned_label_margin = assigned_class_sim - best_other_class_sim
```

핵심 질문은 두 가지다.

```text
1. CLAP이 한 class를 명확하게 고르면 confidence가 높은가?
2. metadata의 assigned class가 다른 class보다 잘 맞으면 confidence가 높은가?
```

## 2. 먼저 봐야 할 중요한 전제

BSD10k confidence 분포는 매우 불균형하다.

```text
confidence 1: 106
confidence 2: 749
confidence 3: 3280
confidence 4: 6045
confidence 5: 776
```

confidence 4가 전체의 절반 이상이다. 반면 confidence 1은 100개 정도뿐이다.

따라서 margin을 5개 분위 구간으로 나눈 뒤 `predicted confidence 1~5`처럼 보는 표는, 정확한 5-class 분류 성능보다는 다음을 보기 위한 표로 해석해야 한다.

```text
margin이 높은 구간으로 갈수록
confidence 4/5 또는 confidence 5가 더 많이 모이는가?
```

## 3. Top1이 metadata class와 일치하는가?

| feature source | top1 assigned match rate | assigned rank mean | assigned rank median |
|---|---:|---:|---:|
| audio | 0.1836 | 7.3815 | 5.0 |
| text | 0.1832 | 7.2912 | 5.0 |

해석:

- audio 기준 top1 class가 metadata class와 일치하는 비율은 약 18.36%다.
- text 기준도 약 18.32%다.
- assigned class의 median rank는 5위다.

즉, CLAP class embedding 공간에서 metadata class가 항상 top1으로 잘 잡히지는 않는다.

이 때문에 `assigned_label_margin` 평균이 대부분 음수로 나온다.

## 4. Confidence별 mean margin 분석

### audio_top1_top2_margin

| confidence | mean |
|---:|---:|
| 1 | 0.0523 |
| 2 | 0.0500 |
| 3 | 0.0500 |
| 4 | 0.0478 |
| 5 | 0.0468 |

해석:

- confidence가 높아질수록 오히려 margin 평균이 조금 감소한다.
- `top1과 top2 차이가 크다`는 것이 confidence가 높다는 뜻으로 이어지지 않는다.
- audio top1/top2 margin은 confidence feature로 부적합하다.

### audio_assigned_label_margin

| confidence | mean |
|---:|---:|
| 1 | -0.1606 |
| 2 | -0.1354 |
| 3 | -0.1168 |
| 4 | -0.0976 |
| 5 | -0.0589 |

해석:

- confidence가 높아질수록 assigned label margin이 단조 증가한다.
- 값은 여전히 음수지만, confidence 5로 갈수록 덜 음수다.
- 즉 metadata class가 다른 class보다 압도적으로 top1은 아니더라도, high-confidence sample일수록 상대적으로 더 가까워진다.
- 네 가지 margin 중 가장 해석이 좋다.

### text_top1_top2_margin

| confidence | mean |
|---:|---:|
| 1 | 0.0464 |
| 2 | 0.0528 |
| 3 | 0.0539 |
| 4 | 0.0525 |
| 5 | 0.0680 |

해석:

- confidence 5에서 top1/top2 margin이 확실히 커진다.
- 하지만 1~4 구간은 단조 증가하지 않는다.
- confidence 5를 잡는 보조 신호는 될 수 있지만, 전체 confidence ordering feature로는 약하다.

### text_assigned_label_margin

| confidence | mean |
|---:|---:|
| 1 | -0.1349 |
| 2 | -0.1273 |
| 3 | -0.1124 |
| 4 | -0.1011 |
| 5 | -0.0733 |

해석:

- confidence가 높아질수록 assigned label margin이 단조 증가한다.
- audio assigned margin과 같은 방향이다.
- 다만 confidence 1과 5의 차이는 audio assigned margin보다 작다.

## 5. 5-bin classification metric 해석

| feature | accuracy | macro F1 | weighted F1 |
|---|---:|---:|---:|
| audio_top1_top2_margin | 0.1907 | 0.1434 | 0.2380 |
| audio_assigned_label_margin | 0.2090 | 0.1646 | 0.2535 |
| text_top1_top2_margin | 0.2019 | 0.1596 | 0.2442 |
| text_assigned_label_margin | 0.2066 | 0.1612 | 0.2521 |

해석:

- 5-class exact classification 성능은 모두 낮다.
- 이건 margin feature가 confidence 1~5를 정확히 분류하지 못한다는 뜻이다.
- 특히 true confidence 분포가 불균형한데 predicted bin은 강제로 20%씩 나뉘기 때문에 accuracy 자체는 보조 지표로만 봐야 한다.
- 그래도 네 feature 중에서는 `audio_assigned_label_margin`이 가장 높다.

## 6. Margin 구간별 high-confidence enrichment

더 중요한 해석은 Q1과 Q5 비교다.

### audio_assigned_label_margin

```text
Q1 high-confidence(4/5) rate: 52.24%
Q5 high-confidence(4/5) rate: 67.37%
gain: +15.13%p

Q1 confidence 5 rate: 3.28%
Q5 confidence 5 rate: 10.59%
gain: +7.30%p
```

해석:

- assigned label margin이 높은 구간일수록 confidence 4/5가 확실히 많다.
- confidence 5 비율도 3.28%에서 10.59%로 증가한다.
- 이 feature는 confidence ranking/filtering 보조 feature로 의미가 있다.

### text_assigned_label_margin

```text
Q1 high-confidence(4/5) rate: 56.30%
Q5 high-confidence(4/5) rate: 65.54%
gain: +9.25%p

Q1 confidence 5 rate: 5.98%
Q5 confidence 5 rate: 10.22%
gain: +4.25%p
```

해석:

- text assigned margin도 같은 방향성이 있다.
- 하지만 audio assigned margin보다 증가폭은 작다.

### audio_top1_top2_margin

```text
Q1 high-confidence(4/5) rate: 63.64%
Q5 high-confidence(4/5) rate: 60.20%
gain: -3.44%p

Q1 confidence 5 rate: 7.12%
Q5 confidence 5 rate: 6.30%
gain: -0.82%p
```

해석:

- audio top1/top2 margin은 오히려 high-confidence 비율이 감소한다.
- 이 feature는 confidence 설명에 도움이 되지 않는다.

### text_top1_top2_margin

```text
Q1 high-confidence(4/5) rate: 63.87%
Q5 high-confidence(4/5) rate: 63.72%
gain: -0.15%p

Q1 confidence 5 rate: 5.70%
Q5 confidence 5 rate: 11.14%
gain: +5.43%p
```

해석:

- confidence 5 비율은 증가한다.
- 하지만 confidence 4/5 전체 비율은 거의 변하지 않는다.
- 최상위 confidence 5를 보는 보조 신호 정도로만 볼 수 있다.

## 7. 최종 결론

이번 실험에서 가장 의미 있는 feature는 다음이다.

```text
audio_assigned_label_margin
```

이유:

- confidence별 평균이 단조 증가한다.
- confidence 5 - confidence 1 평균 차이가 가장 크다.
- Q5 구간에서 confidence 4/5 비율이 Q1보다 +15.13%p 높다.
- Q5 구간에서 confidence 5 비율이 Q1보다 +7.30%p 높다.
- 5-bin classification metric도 네 feature 중 가장 높다.

반면 다음 feature는 confidence feature로는 약하다.

```text
audio_top1_top2_margin
```

이유:

- confidence가 높아질수록 평균 margin이 증가하지 않는다.
- high-confidence enrichment도 없다.

정리하면:

```text
CLAP margin은 confidence 1~5를 정확히 분류하는 feature는 아니다.
하지만 assigned label margin, 특히 audio_assigned_label_margin은
high-confidence sample을 ranking/filtering하는 보조 feature로 의미가 있다.
```

다음 실험에서 사용할 후보는:

```text
audio_assigned_label_margin
text_assigned_label_margin
```

우선순위는:

```text
1순위: audio_assigned_label_margin
2순위: text_assigned_label_margin
보류: text_top1_top2_margin
제외 후보: audio_top1_top2_margin
```
