# BSD10k Confidence Filtering 분류기 v3 보고서

## 1. Task 재정의

이번 실험은 confidence를 `1~5`로 정확히 맞히는 ordinal classification이 아니라, BSD35k-CS에서 가져갈 만한 sample을 고르는 filtering 문제로 재정의했다.

```python
target_binary = 1 if confidence >= 4 else 0
```

- `confidence 1, 2, 3`: low confidence, 버릴 후보
- `confidence 4, 5`: high confidence, 가져갈 후보
- BSD10k rows with embeddings: 10,956
- high-confidence 비율: 0.6226
- feature set: 이전 v2의 `baseline`
- 모델: plain `BinaryConfidenceMLP`

## 2. 모델 구조

이번 모델은 이전 실험에서 복잡한 구조가 큰 이득을 주지 않았다는 결론을 반영해, audio/text/meta/class feature를 concat한 뒤 단순 MLP에 넣는 구조로 고정했다.

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

학습 설정:

- loss: `BCEWithLogitsLoss`
- optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
- batch size: 256
- epochs: 최대 50
- early stopping: validation F1 기준, patience 7
- CV: 5-fold stratified OOF
- threshold: 학습 후 OOF probability에서 tuning

입력 feature:

- CLAP audio embedding
- CLAP text embedding
- class one-hot
- class_top one-hot
- metadata numeric features: title length, tag count, description length, has_description

## 3. Stage 0 Baselines

| baseline                       | accuracy | precision | recall | f1     | auc_pr | auc_roc |
| ------------------------------ | -------- | --------- | ------ | ------ | ------ | ------- |
| Majority: all high             | 0.6226   | 0.6226    | 1.0000 | 0.7674 | 0.6226 | 0.5000  |
| Random: class prior            | 0.5258   | 0.6210    | 0.6118 | 0.6164 | 0.6200 | 0.4943  |
| Previous 5-class: score >= 4.0 | 0.5139   | 0.9318    | 0.2365 | 0.3772 | 0.8374 | 0.7523  |
| Previous 5-class: P4+P5 >= 0.5 | 0.6943   | 0.7230    | 0.8251 | 0.7707 | 0.8305 | 0.7489  |

해석:

- `Majority: all high`의 F1이 0.7674로 높다. 이는 모델이 좋은 것이 아니라 positive class가 62.26%라서 생기는 착시다.
- 따라서 이번 문제에서는 F1만 보면 안 되고, AUC-PR, precision/recall trade-off, retained sample 수를 함께 봐야 한다.
- 이전 5-class 모델의 `score >= 4.0`은 precision은 0.9318로 높지만 recall이 0.2365로 너무 낮다. 매우 엄격한 filter로는 쓸 수 있지만 범용 threshold로는 부적절하다.
- 이전 5-class의 `P4+P5`는 AUC-PR 0.8305로 ranking signal이 꽤 강하다. binary MLP가 반드시 압도해야 하는 baseline이다.

## 4. Stage 1 Binary MLP 성능

| model                                      | accuracy | precision | recall | f1     | auc_pr | auc_roc | threshold |
| ------------------------------------------ | -------- | --------- | ------ | ------ | ------ | ------- | --------- |
| Binary MLP @ F1-optimal                    | 0.6872   | 0.6904    | 0.9021 | 0.7822 | 0.8260 | 0.7461  | 0.4000    |
| Binary MLP @ precision-optimal recall>=0.7 | 0.6847   | 0.7723    | 0.7000 | 0.7344 | 0.8260 | 0.7461  | 0.5850    |
| Binary MLP @ threshold 0.5 reference       | 0.7004   | 0.7343    | 0.8131 | 0.7717 | 0.8260 | 0.7461  | 0.5000    |

해석:

- F1-optimal 기준으로 majority 대비 F1은 0.7674에서 0.7822로 약 +0.0148 개선이다.
- 하지만 이전 5-class `P4+P5 >= 0.5` 대비 F1 개선은 0.7707에서 0.7822로 약 +0.0115에 불과하다.
- AUC-PR은 binary MLP 0.8260, 이전 5-class score 0.8374, 이전 5-class P4+P5 0.8305로, ranking 성능은 이전 5-class 쪽이 오히려 약간 더 좋다.
- 따라서 “처음부터 binary로 학습하면 명확히 좋아질 것”이라는 가설은 약하게만 지지된다. F1은 조금 좋아졌지만, ranking/filtering score로는 이전 5-class 출력이 아직 경쟁력 있다.

## 5. Stage 2 Threshold Sweep

| threshold | accuracy | precision | recall | f1     |
| --------- | -------- | --------- | ------ | ------ |
| 0.3000    | 0.6631   | 0.6599    | 0.9468 | 0.7777 |
| 0.3500    | 0.6770   | 0.6748    | 0.9286 | 0.7816 |
| 0.4000    | 0.6872   | 0.6904    | 0.9021 | 0.7822 |
| 0.4500    | 0.6948   | 0.7104    | 0.8604 | 0.7783 |
| 0.5000    | 0.7004   | 0.7343    | 0.8131 | 0.7717 |
| 0.5500    | 0.6963   | 0.7585    | 0.7515 | 0.7550 |
| 0.6000    | 0.6810   | 0.7787    | 0.6811 | 0.7267 |
| 0.6500    | 0.6647   | 0.8043    | 0.6099 | 0.6937 |
| 0.7000    | 0.6426   | 0.8237    | 0.5419 | 0.6537 |

- F1-optimal threshold: 0.4000
- Precision-optimal threshold @ Recall>=0.7: 0.5850
- PR curve: `outputs/confidence_filter_v3/plots/pr_curve.png`
- ROC curve: `outputs/confidence_filter_v3/plots/roc_curve.png`
- Threshold sweep plot: `outputs/confidence_filter_v3/plots/threshold_sweep.png`

## 6. Stage 3 Calibration

| model                      | ece_10bins | note                                          |
| -------------------------- | ---------- | --------------------------------------------- |
| Binary MLP raw probability | 0.0304     | export probability uses this raw score        |
| Platt scaling on OOF       | 0.0234     | diagnostic calibration candidate              |
| Isotonic regression on OOF | 0.0000     | diagnostic calibration candidate; can overfit |

해석:

- raw probability의 ECE 0.0304는 나쁘지 않다.
- Platt scaling은 ECE를 0.0234로 조금 개선한다.
- Isotonic regression의 ECE 0.0000은 같은 OOF prediction에 직접 fit한 진단값이라 과적합 가능성이 크다. 운영용으로 쓰려면 별도 calibration split 또는 nested calibration이 필요하다.

## 7. Stage 4 BSD35k-CS Filtering Scenarios

| scenario                   | threshold | retained_samples | retained_ratio | expected_precision_from_oof | expected_recall_from_oof | expected_f1_from_oof |
| -------------------------- | --------- | ---------------- | -------------- | --------------------------- | ------------------------ | -------------------- |
| default_0.5                | 0.5000    | 19811            | 0.6296         | 0.7343                      | 0.8131                   | 0.7717               |
| f1_optimal                 | 0.4000    | 25945            | 0.8246         | 0.6904                      | 0.9021                   | 0.7822               |
| precision_optimal_recall70 | 0.5850    | 13715            | 0.4359         | 0.7723                      | 0.7000                   | 0.7344               |
| strict_0.7                 | 0.7000    | 7033             | 0.2235         | 0.8237                      | 0.5419                   | 0.6537               |
| very_strict_0.8            | 0.8000    | 2997             | 0.0953         | 0.8668                      | 0.3939                   | 0.5417               |
| ultra_strict_0.9           | 0.9000    | 670              | 0.0213         | 0.9084                      | 0.2224                   | 0.3573               |

해석:

- F1-optimal threshold 0.4는 BSD35k-CS의 82.46%를 retain한다. filtering 관점에서는 너무 관대할 수 있다.
- threshold 0.7은 22.35%만 retain하고 expected precision은 0.8237이다. 깨끗한 subset을 만들려면 이쪽이 더 실용적이다.
- threshold 0.9는 precision은 0.9084까지 올라가지만 retain 비율이 2.13%로 너무 작다.

## 8. Stage 5 5-class vs Binary 직접 비교

| method                | binary_conversion          | accuracy | precision | recall | f1     | auc_pr | auc_roc | threshold |
| --------------------- | -------------------------- | -------- | --------- | ------ | ------ | ------ | ------- | --------- |
| 5-class previous best | predicted_score >= 4.0     | 0.5139   | 0.9318    | 0.2365 | 0.3772 | 0.8374 | 0.7523  |           |
| 5-class previous best | P(c=4)+P(c=5) >= 0.5       | 0.6943   | 0.7230    | 0.8251 | 0.7707 | 0.8305 | 0.7489  |           |
| Binary MLP v3         | prob >= F1-optimal (0.400) | 0.6872   | 0.6904    | 0.9021 | 0.7822 | 0.8260 | 0.7461  | 0.4000    |

핵심 결론:

- binary MLP는 F1만 보면 가장 높다.
- 하지만 AUC-PR은 이전 5-class score가 가장 높다.
- 즉, 현재 결과는 “binary model이 압도적으로 더 낫다”가 아니라 “binary target으로 바꾸면 threshold-tuned F1은 조금 좋아지지만, 이전 5-class의 score/ranking 정보를 버리면 손해일 수 있다”에 가깝다.

## 9. Stage 6 BSD35k-CS 적용 결과

- prediction file: `outputs/confidence_filter_v3/predictions/BSD35k-CS_filter_predictions.csv`
- BSD35k-CS rows predicted: 31,464
- mean predicted high-confidence probability: 0.5636

## 10. 다음 실험 방향

이번 결과를 보면 다음 실험은 deep model 구조를 더 키우는 쪽보다, 이미 강한 ranking signal을 가진 score들을 제대로 조합하는 쪽이 우선이다.

다음 실험 v4의 목표:

1. 이전 5-class score와 `P4+P5`도 threshold tuning한다.
2. binary MLP probability와 5-class probability를 score-level ensemble한다.
3. OOF 기반 logistic stacker로 두 모델의 정보를 결합한다.
4. 같은 BSD10k OOF에서 F1, AUC-PR, precision@recall 조건, retained ratio를 비교한다.
5. 가장 좋은 filtering score를 BSD35k-CS에 적용한다.

이 방향이 맞는 이유:

- binary MLP는 F1만 약간 높고 AUC-PR은 이전 5-class보다 낮다.
- 이전 5-class의 `score >= 4.0` threshold는 너무 엄격했을 뿐, score 자체의 ranking 성능은 좋다.
- 따라서 “새 모델 하나”보다 “score calibration + threshold tuning + stacker”가 더 가능성이 높다.
