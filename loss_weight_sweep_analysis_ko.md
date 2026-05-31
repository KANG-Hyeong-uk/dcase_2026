# Confidence 1/2 Loss Weight Sweep 분석

## 실험 목적

이 실험은 confidence 1과 2 샘플에 loss weight를 여러 강도로 부여했을 때, 5-class confidence classification과 binary confidence filter 성능이 어떻게 변하는지 확인하기 위한 실험이다.

기준 모델은 `04_similarity_feature_mlp_confidence.ipynb`와 같은 입력/모델 구조를 사용했다.

- 입력: audio embedding + text embedding + class one-hot + similarity/margin feature 4개
- 모델: `Linear(input_dim, 32) -> ReLU -> Dropout(0.2) -> Linear(32, output_dim)`
- 비교한 loss weight:

```python
LOSS_WEIGHT_SWEEP = [
    {'tag': 'baseline_w1_1_w2_1', 'w1': 1.0, 'w2': 1.0},
    {'tag': 'mild_w1_5_w2_1p5', 'w1': 5.0, 'w2': 1.5},
    {'tag': 'medium_w1_10_w2_2', 'w1': 10.0, 'w2': 2.0},
    {'tag': 'current_w1_20p54_w2_2p93', 'w1': 20.540625, 'w2': 2.927839643652561},
    {'tag': 'strong_w1_30_w2_4', 'w1': 30.0, 'w2': 4.0},
    {'tag': 'very_strong_w1_40_w2_6', 'w1': 40.0, 'w2': 6.0},
]
```

## 주요 결과

| setting | 5-class acc | 5-class macro F1 | 5-class MAE | recall 1 | recall 2 | binary acc | binary macro F1 | binary recall 123 | binary recall 45 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline 1,1 | 0.5937 | 0.2969 | 0.4629 | 0.0000 | 0.0982 | 0.7086 | 0.6780 | 0.5306 | 0.8164 |
| mild 5,1.5 | 0.5791 | 0.3196 | 0.4866 | 0.0625 | 0.1786 | 0.7147 | 0.6917 | 0.5855 | 0.7930 |
| medium 10,2 | 0.5760 | 0.3247 | 0.5024 | 0.1250 | 0.2232 | 0.6983 | 0.6783 | 0.5952 | 0.7607 |
| current 20.54,2.93 | 0.5675 | 0.3211 | 0.5182 | 0.1250 | 0.3036 | 0.6989 | 0.6844 | 0.6419 | 0.7334 |
| strong 30,4 | 0.5499 | 0.3034 | 0.5572 | 0.1250 | 0.3304 | 0.6940 | 0.6850 | 0.6952 | 0.6934 |
| very strong 40,6 | 0.5414 | 0.3002 | 0.5742 | 0.1250 | 0.4554 | 0.6940 | 0.6806 | 0.6484 | 0.7217 |

## 해석

loss weight를 주지 않은 baseline은 confidence 1을 전혀 맞추지 못했고, confidence 2 recall도 0.0982로 낮았다. 이 상태의 모델은 낮은 confidence를 독립적인 영역으로 학습하기보다 class 3 또는 4 쪽으로 밀어내는 경향이 강하다.

loss weight를 주기 시작하면 confidence 1과 2 recall이 즉시 상승한다. confidence 1 recall은 `w1=10` 부근에서 0.125에 도달한 뒤 더 커지지 않는다. 즉 confidence 1은 weight를 더 강하게 주어도 추가 개선이 제한적이며, 샘플 수 부족이나 feature 분포 중첩의 영향을 강하게 받는 것으로 보인다.

confidence 2는 weight가 커질수록 지속적으로 개선된다. baseline 0.0982에서 very strong 0.4554까지 상승했다. 그러나 이 개선은 confidence 3 recall 하락과 함께 발생한다. 즉 모델의 decision boundary가 confidence 3 일부를 confidence 2 쪽으로 끌어내리는 방식으로 이동한다.

confidence 4 recall은 강한 weight에서도 비교적 안정적으로 유지된다. baseline 0.8490에서 strong 0.8181 정도로만 감소했다. 따라서 loss weighting의 주요 영향은 confidence 1/2/3 사이의 경계에서 발생하고, confidence 4 영역은 비교적 안정적으로 유지된다.

confidence 5는 거의 개선되지 않았다. 모든 setting에서 recall이 약 0.10 전후이며, 대부분 confidence 4로 예측된다. 이는 기존 실험들과 같은 패턴으로, 현재 feature와 classification 구조에서는 confidence 4와 5를 분리하기 어렵다는 신호다.

## Binary 결과의 의미

binary 123 vs 45에서는 loss weight의 효과가 더 명확하다.

- baseline은 recall 45가 0.8164로 높지만 recall 123은 0.5306에 머문다.
- mild는 binary macro F1이 0.6917로 가장 높고, recall 123을 0.5855까지 올리면서 recall 45도 0.7930으로 잘 유지한다.
- strong은 recall 123이 0.6952로 가장 높고 recall 45도 0.6934로 균형을 이룬다.

따라서 binary confidence filter의 목적에 따라 선택이 달라진다.

## 최종 선택

운영형 균형 모델로는 `mild_w1_5_w2_1p5`가 가장 적합하다. binary accuracy와 binary macro F1이 가장 높고, high confidence recall 손실이 작다.

낮은 confidence를 적극적으로 탐지하는 목적이라면 `strong_w1_30_w2_4`가 가장 적합하다. recall 123이 가장 높고, recall 123과 recall 45가 거의 균형을 이룬다.

5-class macro F1만 기준으로 보면 `medium_w1_10_w2_2`가 가장 좋다. confidence 1/2를 baseline보다 잘 잡으면서 전체 5-class 구조가 과하게 무너지지 않는다.

## 결론

이 실험은 confidence 1/2가 feature상 완전히 구분 불가능한 클래스가 아니라는 점을 보여준다. 기존 모델은 class imbalance 때문에 confidence 1/2를 학습 objective에서 충분히 반영하지 못했고, loss weighting을 적용하자 낮은 confidence recall이 의미 있게 상승했다.

다만 weight를 너무 강하게 주면 5-class accuracy와 MAE가 악화되고 confidence 3이 confidence 2 쪽으로 과도하게 이동한다. 따라서 최종 후보는 목적에 따라 `mild_w1_5_w2_1p5`, `medium_w1_10_w2_2`, `strong_w1_30_w2_4` 세 가지로 보는 것이 타당하다.
