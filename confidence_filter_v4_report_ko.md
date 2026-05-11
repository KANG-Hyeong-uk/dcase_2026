# BSD10k Confidence Filtering v4 보고서

## 1. 먼저 threshold란 무엇인가?

모델은 보통 최종 답을 바로 `0/1`로 내지 않고, 먼저 점수를 낸다.

예를 들어 어떤 모델이 sample마다 이런 점수를 냈다고 하자.

```text
sample A: 0.82
sample B: 0.61
sample C: 0.37
```

여기서 어디부터 “high-confidence sample로 가져갈 것인가?”를 정하는 기준선이 **threshold**다.

예를 들어:

```text
threshold = 0.5
```

이면:

```text
0.82 >= 0.5 -> 가져감
0.61 >= 0.5 -> 가져감
0.37 <  0.5 -> 버림
```

threshold를 높이면 더 엄격해진다.

```text
threshold = 0.8
```

이면:

```text
0.82 >= 0.8 -> 가져감
0.61 <  0.8 -> 버림
0.37 <  0.8 -> 버림
```

즉 threshold는 **모델이 낸 score를 최종 filtering 결정으로 바꾸는 선**이다.

중요한 점:

- threshold가 낮으면 더 많이 가져가지만 precision이 낮아질 수 있다.
- threshold가 높으면 더 적게 가져가지만 precision이 높아질 수 있다.
- filtering 목적에서는 모델 자체만큼 threshold 선택이 중요하다.

## 2. v3 실험은 무엇이었나?

v3는 새 binary classifier를 직접 학습한 실험이다.

원래 confidence 라벨은 정수 `1, 2, 3, 4, 5`였다. v3에서는 이것을 다음처럼 binary 라벨로 바꿨다.

```python
target = 1 if confidence >= 4 else 0
```

즉:

| 원래 confidence | v3 target |
|---:|---:|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 1 |
| 5 | 1 |

의미:

- `0`: low-confidence, 버릴 후보
- `1`: high-confidence, 가져갈 후보

## 3. v3 모델 입력값

v3는 원본 sample의 audio/text/metadata 정보를 직접 입력으로 받는다.

입력값은 이전 v2 실험의 `baseline` feature set과 같다.

### 3.1 CLAP audio embedding

오디오 자체를 CLAP 모델로 숫자 벡터로 바꾼 값이다.

의미:

```text
이 소리가 실제로 어떤 음향적 특징을 갖고 있는가?
```

예를 들어 금속음, 사람 말소리, 악기음, 환경음 같은 정보가 embedding 안에 들어간다.

### 3.2 CLAP text embedding

title, tags, description 같은 텍스트 정보를 CLAP 모델로 숫자 벡터로 바꾼 값이다.

의미:

```text
이 sample의 텍스트 설명은 어떤 내용을 말하고 있는가?
```

### 3.3 class one-hot

세부 class 정보를 one-hot vector로 바꾼 것이다.

예:

```text
class = fx-o
class = is-w
class = sp-s
```

이런 class label을 모델이 숫자로 이해할 수 있게 만든다.

### 3.4 class_top one-hot

큰 카테고리 정보다.

예:

```text
fx
is
sp
```

세부 class보다 더 상위의 분류 정보다.

### 3.5 metadata numeric features

텍스트 자체의 길이와 품질에 가까운 간단한 숫자 feature다.

사용한 값:

- title 글자 수
- tag 개수
- description 글자 수
- description이 있는지 여부

예를 들어 description이 너무 짧거나 tag가 거의 없으면 sample이 덜 명확할 가능성이 있을 수 있다.

## 4. v3 모델 구조

v3는 단순 MLP 구조다.

입력 vector:

```text
audio embedding
+ text embedding
+ class one-hot
+ class_top one-hot
+ metadata numeric features
```

모델 구조:

```text
input vector
↓
Linear(input_dim -> 512)
↓
GELU
↓
Dropout(0.3)
↓
Linear(512 -> 256)
↓
GELU
↓
Dropout(0.3)
↓
Linear(256 -> 1)
↓
sigmoid
↓
high-confidence probability
```

PyTorch 코드로는 다음과 같다.

```python
class BinaryConfidenceMLP(nn.Module):
    def __init__(self, input_dim, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
```

출력 예:

```text
0.73
```

의미:

```text
이 sample은 confidence 4~5일 가능성이 높다.
```

## 5. v3 결과

| model | threshold | accuracy | precision | recall | f1 | auc_pr |
|---|---:|---:|---:|---:|---:|---:|
| Binary MLP @ F1-optimal | 0.400 | 0.6872 | 0.6904 | 0.9021 | 0.7822 | 0.8260 |
| Binary MLP @ precision-oriented | 0.585 | 0.6847 | 0.7723 | 0.7000 | 0.7344 | 0.8260 |
| Binary MLP @ default | 0.500 | 0.7004 | 0.7343 | 0.8131 | 0.7717 | 0.8260 |

비교해야 할 baseline:

| baseline | f1 |
|---|---:|
| Majority: all high | 0.7674 |
| Binary MLP v3 | 0.7822 |

해석:

- v3는 majority baseline보다 좋아졌다.
- 하지만 개선폭은 크지 않다.
- `0.7674 -> 0.7822` 정도의 개선이다.
- 따라서 binary task로 바꾼 것은 맞았지만, binary MLP 단독으로는 성능이 크게 뛰지 않았다.

## 6. v4 실험은 무엇인가?

v4는 새 audio model을 다시 학습한 실험이 아니다.

v4는 이미 존재하는 두 모델의 예측값을 조합하는 실험이다.

사용한 기존 모델:

1. v3 Binary MLP
2. 이전 5-class confidence model

v3 모델은 이런 점수를 낸다.

```text
binary_mlp_prob = 0.73
```

이전 5-class 모델은 이런 값들을 낸다.

```text
P(confidence=1)
P(confidence=2)
P(confidence=3)
P(confidence=4)
P(confidence=5)
predicted_confidence_score
```

v4의 질문은 이것이다.

```text
v3 점수만 쓰는 게 좋을까?
이전 5-class 점수만 쓰는 게 좋을까?
둘을 섞으면 더 좋을까?
```

즉 v4는 **원본 오디오 embedding을 직접 입력으로 받는 모델이 아니라, 이미 학습된 모델들이 낸 score를 입력으로 받는 meta-level 실험**이다.

## 7. v4 입력값 전체 설명

v4에서 사용한 입력값은 다음과 같다.

### 7.1 `binary_mlp_prob`

v3 Binary MLP가 낸 high-confidence 확률이다.

예:

```text
binary_mlp_prob = 0.73
```

의미:

```text
v3 모델 기준 이 sample이 confidence 4~5일 가능성
```

### 7.2 `fiveclass_score`

이전 5-class 모델의 expected confidence score다.

5-class 모델은 각 confidence class의 확률을 낸다.

예:

```text
P1 = 0.01
P2 = 0.04
P3 = 0.25
P4 = 0.60
P5 = 0.10
```

이때 expected score는 다음처럼 계산한다.

```text
fiveclass_score
= 1*P1 + 2*P2 + 3*P3 + 4*P4 + 5*P5
= 1*0.01 + 2*0.04 + 3*0.25 + 4*0.60 + 5*0.10
= 3.74
```

의미:

```text
이전 5-class 모델이 보기에는 confidence가 약 3.74 정도다.
```

### 7.3 `fiveclass_score_01`

`fiveclass_score`는 1~5 범위다. 이것을 0~1 범위로 바꾼 값이다.

```text
fiveclass_score_01 = (fiveclass_score - 1) / 4
```

예:

```text
fiveclass_score = 3.74
fiveclass_score_01 = (3.74 - 1) / 4 = 0.685
```

### 7.4 `fiveclass_p45`

5-class 모델이 confidence 4 또는 5라고 본 확률의 합이다.

```text
fiveclass_p45 = P(confidence=4) + P(confidence=5)
```

예:

```text
P4 = 0.60
P5 = 0.10
fiveclass_p45 = 0.70
```

의미:

```text
이전 5-class 모델 기준 high-confidence일 확률
```

### 7.5 `prob_confidence_3`

5-class 모델이 confidence 3이라고 본 확률이다.

confidence 3은 애매한 sample에 해당한다.

```text
prob_confidence_3이 높다
-> 이 sample은 high라기보다 애매할 가능성이 높다
```

### 7.6 `prob_confidence_4`

5-class 모델이 confidence 4라고 본 확률이다.

### 7.7 `prob_confidence_5`

5-class 모델이 confidence 5라고 본 확률이다.

### 7.8 `fiveclass_margin_p45_minus_p3`

high 쪽 확률이 애매한 confidence 3 확률보다 얼마나 큰지를 보는 값이다.

```text
fiveclass_margin_p45_minus_p3 = P4 + P5 - P3
```

예:

```text
P3 = 0.25
P4 = 0.60
P5 = 0.10

margin = 0.60 + 0.10 - 0.25 = 0.45
```

의미:

```text
high-confidence 쪽 확신이 애매한 3보다 얼마나 강한가?
```

값이 클수록 high-confidence sample일 가능성이 높다.

### 7.9 `rank_average_binary_p45`

`binary_mlp_prob`와 `fiveclass_p45`를 각각 순위로 바꾼 뒤 평균낸 값이다.

왜 순위를 쓰는가?

두 score의 scale과 분포가 다를 수 있기 때문이다.

예:

```text
binary_mlp_prob는 0~1 확률
fiveclass_p45도 0~1이지만 분포가 다름
```

그래서 값 자체를 평균내기보다, 각 score에서 sample이 얼마나 위쪽에 있는지를 본다.

예:

```text
sample A
binary_mlp_prob 기준 순위 점수 = 0.90
fiveclass_p45 기준 순위 점수 = 0.80

rank_average_binary_p45 = (0.90 + 0.80) / 2 = 0.85
```

### 7.10 `rank_average_binary_score`

v4에서 가장 좋은 결과를 낸 score다.

계산:

```text
rank_average_binary_score
= rank(binary_mlp_prob)와 rank(fiveclass_score)의 평균
```

의미:

```text
v3 binary model도 좋게 보고,
이전 5-class expected score도 좋게 보는 sample을 위로 올린다.
```

## 8. v4 모델 구조

v4에는 두 종류의 구조가 있다.

## 8.1 Rank Average 방식

이 방식은 neural network가 아니다.

구조:

```text
binary_mlp_prob
fiveclass_score
↓
각각 순위로 변환
↓
두 순위 평균
↓
v4_filter_score
↓
threshold로 최종 filtering
```

예:

```text
sample A
binary_mlp_prob 순위 점수 = 0.90
fiveclass_score 순위 점수 = 0.80

v4_filter_score = (0.90 + 0.80) / 2 = 0.85
```

그 다음 threshold를 적용한다.

```text
v4_filter_score >= threshold -> 가져감
v4_filter_score <  threshold -> 버림
```

v4에서 가장 좋았던 방식:

```text
Rank average: binary + expected score
```

## 8.2 Logistic Stacker 방식

이 방식은 아주 작은 meta-model이다.

원본 audio/text feature를 입력으로 받지 않는다.

입력은 기존 모델들이 낸 score들이다.

입력 feature:

```python
[
    "binary_mlp_prob",
    "fiveclass_score_01",
    "fiveclass_p45",
    "prob_confidence_3",
    "prob_confidence_4",
    "prob_confidence_5",
    "fiveclass_margin_p45_minus_p3",
]
```

모델 구조:

```text
7개 score 입력
↓
StandardScaler
↓
LogisticRegression
↓
high-confidence probability
```

의미:

```text
기존 모델들의 예측값을 보고,
어떤 조합일 때 실제 confidence 4~5인지 다시 학습한다.
```

## 9. v4 결과

| method | threshold | accuracy | precision | recall | f1 | auc_pr |
|---|---:|---:|---:|---:|---:|---:|
| Rank average: binary + expected score | 0.180 | 0.6873 | 0.6859 | 0.9183 | 0.7853 | 0.8421 |
| OOF logistic stacker | 0.365 | 0.6811 | 0.6773 | 0.9317 | 0.7844 | 0.8404 |
| Rank average: binary + P45 | 0.205 | 0.6908 | 0.6932 | 0.9029 | 0.7843 | 0.8390 |
| 5-class expected score | 3.140 | 0.6712 | 0.6655 | 0.9490 | 0.7823 | 0.8374 |
| Binary MLP v3 | 0.400 | 0.6872 | 0.6904 | 0.9021 | 0.7822 | 0.8260 |

핵심 비교:

| model | f1 | auc_pr |
|---|---:|---:|
| v3 Binary MLP | 0.7822 | 0.8260 |
| 이전 5-class expected score | 0.7823 | 0.8374 |
| v4 Rank average binary + expected score | 0.7853 | 0.8421 |

해석:

- v3 binary MLP 단독보다 이전 5-class score의 ranking 성능이 더 좋았다.
- 둘을 rank average로 섞으면 F1과 AUC-PR이 가장 좋았다.
- 다만 F1 개선폭은 작다.

F1 개선폭:

```text
v3: 0.7822
v4 best: 0.7853
차이: +0.0031
```

AUC-PR 개선폭:

```text
v3: 0.8260
v4 best: 0.8421
차이: +0.0161
```

즉 v4는 “정답을 딱 맞히는 F1”에서는 아주 조금만 좋아졌고, “좋은 sample을 위로 정렬하는 능력”에서는 더 의미 있게 좋아졌다.

## 10. v4 threshold 해석

v4 best method의 threshold는 0.18이다.

```text
Rank average: binary + expected score
threshold = 0.18
```

주의:

```text
0.18은 확률 18%라는 뜻이 아니다.
```

왜냐하면 `rank_average_binary_score`는 확률이 아니라 순위 기반 score이기 때문이다.

즉 0.18은:

```text
순위 평균 score에서 어디를 기준으로 자를 것인가?
```

라는 의미다.

## 11. BSD35k-CS filtering scenario

v4 best score를 BSD35k-CS에 적용했을 때:

| threshold | retained_samples | retained_ratio | expected_precision_from_oof | expected_recall_from_oof | expected_f1_from_oof |
|---:|---:|---:|---:|---:|---:|
| 0.180 | 26,150 | 0.8311 | 0.6859 | 0.9183 | 0.7853 |
| 0.500 | 15,566 | 0.4947 | 0.8040 | 0.6435 | 0.7148 |
| 0.600 | 12,357 | 0.3927 | 0.8409 | 0.5285 | 0.6491 |
| 0.700 | 9,197 | 0.2923 | 0.8813 | 0.4092 | 0.5589 |
| 0.800 | 6,069 | 0.1929 | 0.9284 | 0.2834 | 0.4342 |
| 0.900 | 3,008 | 0.0956 | 0.9616 | 0.1284 | 0.2266 |

해석:

- threshold 0.18은 F1은 가장 좋지만 너무 관대하다. BSD35k-CS의 83.11%를 가져간다.
- threshold 0.5는 약 절반만 가져가고 expected precision이 0.8040이다.
- threshold 0.7은 약 29.23%만 가져가고 expected precision이 0.8813이다.
- threshold 0.9는 매우 엄격해서 expected precision은 0.9616이지만 9.56%만 남는다.

## 12. 실제 추천 threshold

목적에 따라 다르게 선택하는 것이 좋다.

### 많이 가져가고 싶을 때

```text
threshold = 0.18
```

- retain ratio: 83.11%
- expected precision: 0.6859
- F1 기준으로는 가장 좋음
- 하지만 filtering 목적에서는 너무 관대할 수 있음

### 균형 있게 가져가고 싶을 때

```text
threshold = 0.5
```

- retain ratio: 49.47%
- expected precision: 0.8040
- BSD35k-CS에서 약 절반을 가져감
- 실무 filtering 기준으로 가장 무난한 선택

### 깨끗한 subset을 만들고 싶을 때

```text
threshold = 0.7
```

- retain ratio: 29.23%
- expected precision: 0.8813
- 더 적게 가져가지만 더 깨끗함

### 아주 엄격하게 가져가고 싶을 때

```text
threshold = 0.9
```

- retain ratio: 9.56%
- expected precision: 0.9616
- 거의 확실한 sample만 가져가는 설정

## 13. 최종 결론

v3와 v4의 차이는 다음과 같다.

| version | 무엇을 학습/사용했나 | 입력값 | 출력 |
|---|---|---|---|
| v3 | binary MLP 직접 학습 | audio/text/class/metadata feature | high-confidence probability |
| v4 | 기존 모델 score 조합 | v3 probability + 5-class model score/probabilities | filtering score |

최종적으로 v4가 가장 좋았다.

```text
Best v4 method = Rank average: binary + expected score
```

하지만 F1 개선폭은 작다.

```text
v3 F1 = 0.7822
v4 F1 = 0.7853
```

따라서 v4의 의미는:

```text
정확도가 크게 뛰었다
```

라기보다는:

```text
sample을 high-confidence 순서로 정렬하는 score가 더 좋아졌다
```

에 가깝다.

실제 BSD35k-CS filtering에는 `threshold=0.5` 또는 `threshold=0.7`을 추천한다.

- `0.5`: 약 절반 retain, expected precision 약 0.80
- `0.7`: 약 29% retain, expected precision 약 0.88

저장 파일:

- OOF score: `outputs/confidence_filter_v4/predictions/BSD10k_oof_v4_scores.csv`
- BSD35k-CS prediction: `outputs/confidence_filter_v4/predictions/BSD35k-CS_filter_predictions_v4.csv`
