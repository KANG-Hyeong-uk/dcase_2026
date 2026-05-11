# kang_study t-SNE cluster 진단 보고서

작성일: 2026-05-11

## 질문

t-SNE 그림에서 5개 그룹처럼 보이는데, 이것이 실제 confidence 1-5 class가 잘못 라벨링되어 있어서 생긴 현상인지 확인했다.

검증 방법은 다음과 같다.

1. `kang_study_hidden_tsne_sample.csv`에 저장된 t-SNE 좌표를 사용했다.
2. t-SNE 2D 좌표에 대해 KMeans `k=5`를 적용해 5개 cluster를 강제로 할당했다.
3. 각 cluster가 실제 `true_confidence`, 모델의 `predicted_confidence_class`, `class_top`, `class` 중 무엇과 대응되는지 확인했다.
4. 각 cluster의 대표 sample을 추출했다.

## 산출 파일

```text
outputs/kang_study/reports/kang_study_tsne_cluster_assignments.csv
outputs/kang_study/reports/kang_study_tsne_cluster_summary.csv
outputs/kang_study/reports/kang_study_tsne_cluster_examples.csv
outputs/kang_study/reports/kang_study_tsne_cluster_agreement.csv
outputs/kang_study/reports/kang_study_tsne_cluster_true_confidence_crosstab.csv
outputs/kang_study/reports/kang_study_tsne_cluster_predicted_class_crosstab.csv
outputs/kang_study/reports/kang_study_tsne_cluster_class_top_crosstab.csv
outputs/kang_study/reports/kang_study_tsne_cluster_class_crosstab.csv
```

## 핵심 결과

t-SNE 좌표상에서 5개 덩어리처럼 보이는 구조는 confidence 1-5와 거의 대응하지 않았다.

cluster와 label 간 agreement는 다음과 같다.

| comparison | ARI | NMI |
|---|---:|---:|
| t-SNE cluster vs true confidence | -0.00005 | 0.00129 |
| t-SNE cluster vs predicted confidence class | 0.00030 | 0.00609 |
| t-SNE cluster vs class_top | -0.00006 | 0.00128 |
| t-SNE cluster vs class | 0.00008 | 0.00657 |

ARI와 NMI가 거의 0이다. 즉 t-SNE에서 보이는 5개 그룹은 true confidence, predicted confidence, top-level class, second-level class 중 어느 것과도 의미 있게 정렬되지 않는다.

반면 t-SNE 좌표 자체에서 KMeans cluster compactness는 높게 나온다.

| metric | value |
|---|---:|
| silhouette on t-SNE coordinates | 0.6371 |

이 말은 다음과 같다.

```text
t-SNE 2D 그림 안에서는 5개 덩어리처럼 보인다.
하지만 그 5개 덩어리가 confidence class나 sound class를 의미하지는 않는다.
```

즉 이것은 label 오류의 증거가 아니라, t-SNE projection 자체가 만든 2D 시각적 구조일 가능성이 크다.

## Cluster별 요약

5개 cluster 모두 dominant true confidence는 4였고, dominant predicted class도 4였다.

| t-SNE cluster | n | mean true confidence | mean predicted score | dominant true confidence | dominant true rate | dominant predicted class | dominant predicted rate | dominant top class |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 576 | 3.6250 | 3.6212 | 4 | 0.5729 | 4 | 0.7795 | fx |
| 1 | 597 | 3.5997 | 3.6088 | 4 | 0.5394 | 4 | 0.7002 | fx |
| 2 | 610 | 3.6377 | 3.6184 | 4 | 0.5787 | 4 | 0.7508 | fx |
| 3 | 578 | 3.5727 | 3.6383 | 4 | 0.5242 | 4 | 0.7163 | fx |
| 4 | 639 | 3.6228 | 3.6798 | 4 | 0.5603 | 4 | 0.7246 | fx |

이 표에서 중요한 점은 cluster별 평균 true confidence와 평균 predicted score가 거의 비슷하다는 것이다.

```text
mean true confidence: 약 3.57 - 3.64
mean predicted score: 약 3.61 - 3.68
```

만약 t-SNE의 5개 cluster가 confidence 1-5를 의미한다면, 어떤 cluster는 평균 confidence가 1 또는 2에 가깝고, 다른 cluster는 5에 가까워야 한다. 하지만 실제로는 모든 cluster가 confidence 4 중심이다.

## Cluster별 true confidence 비율

각 cluster 내부의 true confidence 분포는 전체 데이터 분포와 거의 비슷하다.

| cluster | conf 1 rate | conf 2 rate | conf 3 rate | conf 4 rate | conf 5 rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0087 | 0.0556 | 0.3003 | 0.5729 | 0.0625 |
| 1 | 0.0134 | 0.0637 | 0.3082 | 0.5394 | 0.0754 |
| 2 | 0.0082 | 0.0557 | 0.2918 | 0.5787 | 0.0656 |
| 3 | 0.0069 | 0.0727 | 0.3287 | 0.5242 | 0.0675 |
| 4 | 0.0156 | 0.0563 | 0.2926 | 0.5603 | 0.0751 |

즉 cluster 0은 confidence 1, cluster 1은 confidence 2, 이런 식의 대응이 전혀 없다. 모든 cluster에 1-5가 섞여 있고, 모든 cluster가 confidence 4를 가장 많이 포함한다.

## 대표 sample 관찰

각 cluster의 중심 근처 sample을 보면 여러 class와 confidence가 섞여 있다.

예를 들어 cluster 0 중심 근처 sample:

| sound_id | class_top | class | true confidence | predicted class | predicted score |
|---:|---|---|---:|---:|---:|
| 59053 | fx | fx-n | 3 | 4 | 3.6565 |
| 169282 | ss | ss-n | 1 | 4 | 3.6123 |
| 83986 | fx | fx-n | 3 | 4 | 3.6379 |
| 92846 | fx | fx-n | 4 | 4 | 3.6760 |
| 111403 | fx | fx-a | 3 | 4 | 3.5425 |
| 63618 | fx | fx-a | 4 | 4 | 3.6359 |

같은 cluster 안에 true confidence 1, 3, 4가 같이 있다. 따라서 이 cluster는 특정 confidence class가 아니다.

또한 cluster 0에서 predicted score가 높은 sample들은 `is-k`, `is-p` 같은 class가 포함되어 있었고, 낮은 score sample은 `sp-s`, `ss-s`, `sp-p` 등이 포함되어 있었다. 하지만 이것도 cluster 전체를 설명하는 규칙은 아니었다. 각 cluster의 top class 분포가 서로 거의 비슷했기 때문이다.

## 왜 t-SNE에서 5개로 보이나?

t-SNE는 원래 고차원 공간의 local neighborhood를 2D에 강제로 펼치는 알고리즘이다. 따라서 실제 고차원에서 명확한 5개 class가 없어도, 2D 상에서는 덩어리처럼 보이는 구조가 생길 수 있다.

이번 경우에는 다음 이유가 가능성이 높다.

1. t-SNE projection artifact

t-SNE는 local density와 neighborhood를 강조한다. 고차원에서는 연속적인 구조여도 2D에서는 섬처럼 분리되어 보일 수 있다.

2. KMeans `k=5`는 강제 분할이다

이번 분석은 사용자가 본 5개 덩어리를 검증하기 위해 k=5로 나누었다. 하지만 k=5로 나누었다고 해서 실제 데이터에 5개 의미 class가 있다는 뜻은 아니다.

3. confidence label과 무관한 hidden feature 구조

MLP hidden representation은 confidence만 담는 공간이 아니다. 입력에는 audio embedding, text embedding, class one-hot이 들어간다. 따라서 hidden space에는 sound semantic structure, CLAP embedding geometry, class 정보, 모델의 confidence decision 정보가 섞여 있다.

4. confidence 자체가 clean cluster label이 아니다

confidence는 semantic class가 아니라 annotation ambiguity에 가까운 값이다. 같은 sound class 안에서도 confidence 3, 4, 5가 섞일 수 있다.

## 정답 레이블이 잘못 들어갔을 가능성은?

현재 분석 결과만 보면, 정답 레이블이 잘못 들어갔다고 보기는 어렵다.

그 이유는 다음과 같다.

1. t-SNE cluster와 true confidence의 ARI/NMI가 거의 0이다.
2. 모든 t-SNE cluster 안에 confidence 1-5가 섞여 있다.
3. 모든 cluster의 dominant true confidence가 4이다.
4. cluster별 평균 true confidence가 거의 같다.
5. cluster별 top class 분포도 서로 크게 다르지 않다.

즉 t-SNE의 5개 시각적 덩어리는 `confidence 1-5가 잘못 들어간 것`이라기보다, t-SNE가 hidden representation을 2D로 펼치면서 만든 geometry로 보는 것이 더 타당하다.

## 결론

이번 추가 분석의 결론은 다음과 같다.

```text
t-SNE에서 5개 그룹처럼 보이는 것은 confidence 1-5 class가 잘 분리된 증거가 아니다.
```

오히려 강제로 5개 cluster를 할당해보면, 각 cluster는 모두 confidence 4 중심이고 1-5가 섞여 있다. true confidence, predicted confidence, class_top, class 중 어느 것과도 cluster가 의미 있게 일치하지 않는다.

따라서 현재 그림은 다음을 의미한다.

```text
hidden representation은 2D t-SNE에서 시각적 덩어리를 만들지만,
그 덩어리는 confidence class structure가 아니다.
```

confidence label이 잘못 주어졌다기보다는, confidence라는 label 자체가 audio/text/class feature 공간에서 cleanly separable하지 않다고 해석하는 것이 더 맞다.

