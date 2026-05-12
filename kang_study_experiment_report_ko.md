# kang_study 실험 결과 보고서

작성일: 2026-05-10

## 1. 실험 목적

이 실험의 목적은 BSD10k의 `confidence`를 1-5 정수 class로 그대로 두고, 제공된 CLAP embedding만으로 confidence 분류가 어느 정도 가능한지 확인하는 것이다.

이전 실험에서는 metadata 길이 feature(`title_chars`, `tag_count`, `description_chars`, `has_description`)도 같이 사용했지만, 해당 feature는 제목/태그/설명의 의미를 담는 feature가 아니라 metadata의 양만 나타내는 약한 proxy다. 따라서 이번 `kang_study`에서는 의미가 더 직접적인 feature만 남겼다.

사용한 feature는 다음 세 가지뿐이다.

| feature | dim | 설명 |
|---|---:|---|
| CLAP audio embedding | 512 | LAION-CLAP 기반 오디오 임베딩 |
| CLAP text embedding | 512 | LAION-CLAP 기반 텍스트 임베딩 |
| class one-hot | 23 | 2차 class 코드 one-hot |
| total | 1047 | MLP 입력 차원 |

사용하지 않은 feature는 다음과 같다.

```text
title_chars
tag_count
description_chars
has_description
class_top one-hot
cosine / L2 / norm 등 handcrafted similarity feature
ensemble / stacking / two-tower
```

## 2. 데이터와 target

학습 데이터는 BSD10k 중 confidence label과 audio/text embedding이 모두 존재하는 sample만 사용했다.

```text
usable rows: 10,956
target: confidence ∈ {1, 2, 3, 4, 5}
CV: 5-fold StratifiedKFold
```

confidence 분포는 class 4에 강하게 치우쳐 있다.

| confidence | n | rate |
|---:|---:|---:|
| 1 | 106 | 0.97% |
| 2 | 749 | 6.84% |
| 3 | 3,280 | 29.94% |
| 4 | 6,045 | 55.18% |
| 5 | 776 | 7.08% |

이 분포 때문에 단순 accuracy만 보면 모델이 실제로 confidence 구조를 잘 배웠는지 판단하기 어렵다. 그래서 MAE, Spearman, QWK, macro F1, balanced accuracy도 함께 확인했다.

## 3. 모델 구조

모델은 plain MLP 하나만 사용했다.

```text
Input x: 1047 dim
  = audio_emb 512
  + text_emb 512
  + class_onehot 23

Linear(1047 -> 512)
GELU
Dropout(0.3)

Linear(512 -> 256)
GELU
Dropout(0.3)

Linear(256 -> 5)

Output: logits for confidence 1, 2, 3, 4, 5
```

PyTorch 구조는 다음과 같다.

```python
class ConfidenceMLP(nn.Module):
    def __init__(self, input_dim, n_classes=5, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden[1], n_classes)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        return self.classifier(self.forward_features(x))
```

모델 output은 5개 logit이다.

```text
logit_1 -> confidence 1
logit_2 -> confidence 2
logit_3 -> confidence 3
logit_4 -> confidence 4
logit_5 -> confidence 5
```

hard prediction은 `argmax(softmax(logits)) + 1`로 계산했다.

soft confidence score는 다음 expected score로 계산했다.

```python
predicted_confidence_score = sum(P(confidence=k) * k for k in [1, 2, 3, 4, 5])
```

## 4. Loss와 학습 설정

loss는 EMD loss를 사용했다.

```python
probs = softmax(logits)
pred_cdf = cumsum(probs)
true_cdf = cumsum(one_hot_target)
loss = mean((pred_cdf - true_cdf) ** 2)
```

EMD loss를 쓴 이유는 confidence label이 nominal class가 아니라 ordinal class이기 때문이다. 예를 들어 true confidence가 4일 때, 3으로 예측하는 것과 1로 예측하는 것은 같은 오류가 아니다. EMD loss는 이 순서 정보를 반영한다.

학습 설정은 다음과 같다.

| setting | value |
|---|---:|
| optimizer | AdamW |
| learning rate | 1e-3 |
| weight decay | 1e-4 |
| scheduler | CosineAnnealingLR |
| batch size | 256 |
| max epochs | 50 |
| early stopping | val MAE patience 7 |
| CV | 5-fold StratifiedKFold |
| seed | 42 |

각 fold의 best epoch와 MAE는 다음과 같다.

| fold | best epoch | best MAE |
|---:|---:|---:|
| 0 | 10 | 0.5022 |
| 1 | 4 | 0.5124 |
| 2 | 1 | 0.5206 |
| 3 | 1 | 0.5172 |
| 4 | 1 | 0.5166 |

## 5. OOF 성능

5-fold OOF 전체 성능은 다음과 같다.

| metric | value |
|---|---:|
| MAE on expected score | 0.5138 |
| Spearman | 0.4653 |
| Accuracy | 0.5802 |
| Balanced accuracy | 0.2886 |
| Quadratic weighted kappa | 0.3315 |
| Macro precision | 0.4233 |
| Macro recall | 0.2886 |
| Macro F1 | 0.3054 |

겉보기 accuracy는 0.58 정도지만, balanced accuracy와 macro F1이 낮다. 이는 모델이 전체적으로 majority class인 confidence 4에 많이 수렴하고 있으며, minority class인 1, 2, 5를 잘 분리하지 못한다는 뜻이다.

class별 성능은 다음과 같다.

| confidence | precision | recall | F1 | support |
|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | 106 |
| 2 | 0.4514 | 0.1055 | 0.1710 | 749 |
| 3 | 0.4571 | 0.3585 | 0.4018 | 3,280 |
| 4 | 0.6225 | 0.8242 | 0.7093 | 6,045 |
| 5 | 0.5854 | 0.1546 | 0.2446 | 776 |

confidence 4는 recall 0.8242로 가장 잘 맞지만, confidence 1은 한 번도 직접 예측되지 않았다. confidence 2와 5도 recall이 낮다.

## 6. Confusion matrix 해석

OOF confusion matrix는 다음과 같다.

| true \ pred | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| true 1 | 0 | 10 | 51 | 45 | 0 |
| true 2 | 0 | 79 | 369 | 301 | 0 |
| true 3 | 0 | 48 | 1,176 | 2,053 | 3 |
| true 4 | 0 | 36 | 945 | 4,982 | 82 |
| true 5 | 0 | 2 | 32 | 622 | 120 |

예측 class 분포는 다음과 같다.

| predicted confidence | n | rate |
|---:|---:|---:|
| 1 | 0 | 0.00% |
| 2 | 175 | 1.60% |
| 3 | 2,573 | 23.49% |
| 4 | 8,003 | 73.05% |
| 5 | 205 | 1.87% |

true 분포에서 confidence 4는 55.18%였지만, 예측에서는 confidence 4가 73.05%까지 늘었다. 즉 모델은 불확실한 sample을 세밀하게 1/2/3/4/5로 나누기보다 3과 4, 특히 4로 많이 몰아준다.

시각화 파일:

![Confusion matrix count](outputs/kang_study/plots/confusion_matrix_count.png)

![Confusion matrix row normalized](outputs/kang_study/plots/confusion_matrix_row_normalized.png)

row-normalized confusion matrix를 보면 true 4의 대부분은 4로 가지만, true 3도 상당수가 4로 올라가고 true 5도 상당수가 4로 내려온다. 이는 class 3/4/5 사이 경계가 뚜렷하지 않다는 것을 보여준다.

## 7. Classification 시각화 해석

class별 precision/recall/F1 시각화는 다음과 같다.

![Per-class metrics](outputs/kang_study/plots/per_class_metrics.png)

이 그림에서 confidence 4만 상대적으로 높은 성능을 보이고, confidence 1/2/5는 recall이 낮다. 즉 모델은 minority confidence class를 독립적인 class로 학습하지 못하고 있다.

true confidence별 predicted score 분포는 다음과 같다.

![Predicted score by true confidence](outputs/kang_study/plots/predicted_score_by_true_confidence_boxplot.png)

이 boxplot은 expected confidence score가 true confidence에 따라 어느 정도 증가하는 경향은 있지만, 각 class의 분포가 크게 겹친다는 것을 보여준다. 특히 3, 4, 5는 분포가 많이 겹친다.

true class 분포와 predicted class 분포 비교는 다음과 같다.

![True vs predicted class distribution](outputs/kang_study/plots/true_vs_predicted_class_distribution.png)

이 그림은 모델이 실제 분포보다 confidence 4를 더 많이 예측한다는 점을 명확하게 보여준다.

## 8. Hidden feature PCA 시각화

MLP의 마지막 hidden layer는 256차원이다. 각 sample에 대해 OOF 방식으로 hidden feature를 추출한 뒤 PCA로 2차원에 투영했다.

PCA explained variance ratio는 다음과 같았다.

```text
PC1: 0.1960
PC2: 0.1058
sum: 0.3018
```

즉 2D PCA 그림은 원래 256차원 hidden feature 전체 분산의 약 30.18%만 설명한다. 따라서 이 그림이 hidden space 전체를 완벽히 나타내는 것은 아니지만, 가장 큰 두 변화 방향에서 confidence가 어떻게 섞이는지 보는 데에는 유용하다.

PCA 결과는 다음과 같다.

![Hidden PCA true vs predicted](outputs/kang_study/plots/hidden_pca_true_vs_predicted.png)

### TRUE 그림의 의미

왼쪽 그림은 실제 데이터가 hidden feature 공간에서 어떻게 분포하는지 보여준다. 색은 true confidence label이다.

중요한 점은 같은 hidden feature cluster 안에도 여러 confidence가 섞여 있다는 것이다. 즉 이 공간은 `이 좌표면 반드시 confidence=4`처럼 딱 결정되는 공간이 아니다.

따라서 TRUE 그림은 다음과 같이 해석해야 한다.

```text
잘못된 해석: 완벽한 정답 공간
올바른 해석: 실제 데이터 구조
```

만약 confidence가 feature space에서 완벽히 separable했다면, PCA 그림에서 특정 영역은 전부 confidence 1, 다른 영역은 전부 confidence 3, 또 다른 영역은 전부 confidence 5처럼 나뉘었을 것이다.

하지만 실제 그림에서는 한 cluster 안에 1, 2, 3, 4, 5가 섞여 있다. 이는 confidence 자체가 semantic class처럼 명확히 분리되는 label이 아니라는 뜻이다.

confidence는 `dog bark`, `engine sound`, `ambience` 같은 소리의 의미 class가 아니다. confidence는 annotation 과정에서의 명확함 정도, ambiguity, label certainty에 가깝다. 따라서 같은 semantic class 안에서도 어떤 sample은 매우 명확하고, 어떤 sample은 애매할 수 있다.

### PREDICTED 그림의 의미

오른쪽 그림은 모델이 실제로 어떻게 판단했는지를 보여준다. 색은 predicted confidence class이다.

이 그림에서 모델은 대부분 sample을 confidence 4로 예측한다. 즉 모델은 hidden feature 공간 안에서 confidence 1-5의 세밀한 경계를 학습했다기보다 majority class인 4로 수렴하는 경향이 강하다.

이 현상의 핵심 이유는 세 가지다.

1. 데이터 imbalance

confidence 4가 전체의 55.18%다. 따라서 모델 입장에서는 confidence 4를 많이 예측하는 것이 training objective상 유리하다.

2. feature space 자체의 overlap

TRUE 그림에서 이미 confidence들이 많이 섞여 있다. 즉 decision boundary 자체가 깔끔하지 않다.

3. label noise 또는 fuzzy label 성격

annotator도 confidence 3과 4, 4와 5 사이를 헷갈릴 수 있다. 즉 ground truth 자체가 hard categorical label이라기보다 fuzzy ordinal score에 가깝다.

## 9. t-SNE와 군집화 지표

t-SNE 결과는 다음과 같다.

![Hidden t-SNE true vs predicted](outputs/kang_study/plots/hidden_tsne_true_vs_predicted.png)

PCA와 마찬가지로, true confidence 기준으로는 class가 깔끔하게 분리되지 않는다. predicted class 기준으로는 모델이 주로 confidence 3/4 중심으로 나누며, 특히 4로 많이 수렴하는 구조가 보인다.

군집화 정도를 수치로 보기 위해 hidden 256차원에서 silhouette score를 계산했다.

| space | label | silhouette |
|---|---|---:|
| hidden_256 | true_confidence | -0.0764 |
| hidden_256 | predicted_confidence_class | -0.0590 |

silhouette score는 높을수록 cluster separation이 좋다. 0보다 작다는 것은 같은 label끼리 뚜렷하게 뭉쳐 있기보다 다른 label들과 많이 겹친다는 뜻이다.

따라서 hidden feature 기준에서도 confidence 1-5는 clean cluster structure를 갖지 않는다.

## 10. BSD35k-CS 적용 결과

BSD10k 전체로 final model을 다시 학습한 뒤 BSD35k-CS 31,464개에 적용했다.

저장 파일:

```text
outputs/kang_study/predictions/BSD35k-CS_predicted_kang_study.csv
```

BSD35k-CS 예측 요약은 다음과 같다.

```text
rows: 31,464
mean predicted confidence score: 3.5201
```

예측 class 분포:

| predicted confidence | n | rate |
|---:|---:|---:|
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 9,771 | 31.05% |
| 4 | 21,669 | 68.87% |
| 5 | 24 | 0.08% |

score 분포:

| statistic | value |
|---|---:|
| mean | 3.5201 |
| std | 0.2179 |
| min | 2.7015 |
| 10% | 3.2618 |
| 25% | 3.3715 |
| 50% | 3.4985 |
| 75% | 3.6594 |
| 90% | 3.8102 |
| max | 4.5751 |

BSD35k-CS에서도 모델은 대부분 confidence 3 또는 4로 예측한다. confidence 1/2는 전혀 나오지 않고, confidence 5도 거의 나오지 않는다. 이는 BSD10k OOF에서 관찰된 majority-class 수렴과 같은 현상이다.

## 11. 핵심 결론

이 실험의 가장 중요한 결론은 다음과 같다.

```text
confidence는 clean categorical structure가 아니다.
```

즉 confidence 1, 2, 3, 4, 5가 feature space에서 완전히 분리된 class처럼 존재하지 않는다. 오히려 confidence는 annotation ambiguity 또는 label certainty를 나타내는 연속적 성격의 score에 가깝다.

TRUE PCA/t-SNE 그림은 실제 hidden feature 공간에서 confidence label들이 많이 overlap한다는 것을 보여준다. PREDICTED PCA/t-SNE 그림은 모델이 그 overlap 구조를 세밀하게 학습하지 못하고 majority class인 confidence 4로 많이 수렴한다는 것을 보여준다.

따라서 이 결과는 다음을 시사한다.

1. 5-class hard classification은 feature space 구조상 본질적으로 어렵다.
2. confidence 3/4/5 경계는 특히 fuzzy하다.
3. confidence 1과 2는 sample 수가 너무 적고 hidden space에서 독립 cluster로 분리되지 않는다.
4. 모델 성능 병목은 MLP 구조보다 label ambiguity, class imbalance, feature-label separability에 가깝다.
5. `predicted_confidence_score`처럼 연속 expected score로 사용하는 것이 hard class보다 더 자연스럽다.

추가적으로, 낮은 confidence 영역과 높은 confidence 영역을 큰 범주로 나누는 방식은 5-class hard classification보다 안정적일 가능성이 있다. 다만 이 경우에도 target을 단순히 `confidence >= 4` 같은 binary로 바꾸면 ordinal 정보가 사라지므로, 목적에 따라 expected score 기반 ranking/filtering 또는 ordinal-aware grouping을 조심스럽게 설계하는 것이 좋다.

## 12. 산출물

주요 산출물은 다음 위치에 저장되어 있다.

```text
kang_study.ipynb
outputs/kang_study/reports/kang_study_cv_summary.csv
outputs/kang_study/reports/kang_study_fold_metrics.csv
outputs/kang_study/reports/kang_study_per_class_metrics.csv
outputs/kang_study/reports/kang_study_confusion_matrix.csv
outputs/kang_study/reports/kang_study_clustering_metrics.csv
outputs/kang_study/predictions/BSD10k_oof_kang_study.csv
outputs/kang_study/predictions/BSD35k-CS_predicted_kang_study.csv
outputs/kang_study/plots/
```

