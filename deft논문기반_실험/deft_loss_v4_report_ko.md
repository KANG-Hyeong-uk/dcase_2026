# DEFT Loss v4 OOF Score Stacking 분석 보고서

## 1. 실험 목적

이 실험의 목적은 `04_similarity_feature_mlp_confidence.ipynb` 계열의 DEFT/CLAP similarity feature와 04 MLP 구조를 유지하면서, confidence 1과 2에 더 큰 loss를 주는 mild weighted loss를 적용한 뒤, 원래 `confidence_filter_v4_score_stacking.ipynb`처럼 OOF score stacking으로 BSD35k-CS filtering score를 만드는 것이다.

실험 흐름은 다음과 같다.

- 입력 feature: audio embedding + text embedding + class one-hot + similarity/margin scalar feature
- 모델 구조: `Linear(input, 32) -> ReLU -> Dropout(0.2) -> Linear(output)`
- loss weight: confidence 1 = 5.0, confidence 2 = 1.5, confidence 3/4/5 = 1.0
- base model:
  - 5-class confidence classifier
  - binary classifier: confidence 123 vs 45
- stacking:
  - OOF logistic stacker
  - OOF ridge stacker
  - rank average score
- 평가 방식: BSD10k OOF
- 최종 목적: BSD35k-CS에서 high-confidence sample을 고르는 filtering score 생성

## 2. 핵심 결과 요약

`deft_loss_v4`에서 가장 좋은 F1은 OOF ridge stacker가 냈다.

| 목적 | 선택된 방법 | threshold | precision | recall_45 | F1 | macro-F1 | recall_123 | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F1 최적 | OOF ridge stacker | 0.386 | 0.6753 | 0.9320 | 0.7832 | 0.5816 | 0.2609 | 0.8375 |
| macro-F1 최적 | OOF ridge stacker | 0.569 | 0.7686 | 0.7216 | 0.7443 | 0.6776 | 0.6416 | 0.8375 |
| 123/45 recall 균형 | rank average: binary + expected + P45 | 0.454 | 0.7815 | 0.6847 | 0.7299 | 0.6753 | 0.6842 | 0.8366 |
| precision-oriented | rank average: binary + expected + P45 | 0.436 | 0.7745 | 0.7000 | 0.7354 | 0.6752 | 0.6638 | 0.8366 |
| recall-oriented | rank average: binary + expected + P45 | 0.225 | 0.7002 | 0.8812 | 0.7803 | 0.6301 | 0.3775 | 0.8366 |

가장 중요한 관찰은 다음과 같다.

- F1만 보면 `OOF ridge stacker`가 가장 좋다.
- 하지만 F1 최적 threshold는 recall_45가 매우 높고 recall_123이 낮아서, BSD35k를 많이 가져오는 대신 low-confidence sample을 충분히 거르지 못한다.
- 실제 noisy filtering 관점에서는 macro-F1 최적 또는 balanced recall threshold가 더 중요하다.
- `deft_loss_v4`는 confidence 1/2에 loss를 더 준 효과 때문에 123 쪽 recall을 명시적으로 끌어올릴 수 있다.

## 3. 원래 v4와의 정량 비교

비교 대상은 `notebooks/confidence_filter_v4_score_stacking.ipynb`의 기존 v4 결과다.

### 3.1 최고 F1 비교

| 실험 | 최고 방법 | F1 | precision | recall_45 | AUC-PR |
|---|---|---:|---:|---:|---:|
| 원래 v4 | Rank average: binary + expected score | 0.7853 | 0.6859 | 0.9183 | 0.8421 |
| deft_loss_v4 | OOF ridge stacker | 0.7832 | 0.6753 | 0.9320 | 0.8375 |

최고 F1 기준으로는 원래 v4가 약간 더 좋다.

- F1 차이: `0.7853 - 0.7832 = 0.0021`
- AUC-PR 차이: `0.8421 - 0.8375 = 0.0047`

차이는 크지 않지만, 전체 ranking 성능은 원래 v4가 아직 우세하다. BSD35k를 score 순서대로 정렬해서 상위 sample을 뽑는 목적에서는 AUC-PR이 중요하므로, 이 지점은 의미가 있다.

### 3.2 precision-oriented 비교

| 실험 | 방법 | threshold | precision | recall_45 | F1 | AUC-PR |
|---|---|---:|---:|---:|---:|---:|
| 원래 v4 | Rank average: binary + P45 | 0.440 | 0.7808 | 0.7036 | 0.7402 | 0.8390 |
| 원래 v4 | Rank average: binary + expected score | 0.440 | 0.7801 | 0.7025 | 0.7393 | 0.8421 |
| deft_loss_v4 | rank average: binary + expected + P45 | 0.436 | 0.7745 | 0.7000 | 0.7354 | 0.8366 |

precision-oriented 영역에서도 원래 v4가 약간 낫다.

원래 v4는 비슷한 recall 조건에서 precision과 F1이 모두 조금 높다. 따라서 “BSD35k를 최대한 깨끗하게 가져오는” 목적만 보면 원래 v4를 바로 대체하기에는 `deft_loss_v4`가 아직 부족하다.

## 4. deft_loss_v4가 개선한 부분

`deft_loss_v4`의 장점은 최고 F1이 아니라, confidence 1/2/3을 더 잘 의식하는 균형성이다.

macro-F1 최적 threshold의 confusion matrix는 다음과 같다.

```text
true_123 -> pred_123: 0.6416
true_123 -> pred_45 : 0.3584

true_45  -> pred_123: 0.2784
true_45  -> pred_45 : 0.7216
```

balanced recall threshold의 confusion matrix는 다음과 같다.

```text
true_123 -> pred_123: 0.6842
true_123 -> pred_45 : 0.3158

true_45  -> pred_123: 0.3153
true_45  -> pred_45 : 0.6847
```

이 결과는 중요하다. 기존 단순 binary 또는 5-class score는 high-confidence 쪽 recall을 높이는 데 강했지만, low-confidence인 123을 충분히 잡는 데 약했다. 반면 `deft_loss_v4`는 confidence 1과 2에 loss를 더 준 덕분에 threshold를 조정했을 때 123 recall을 0.64~0.68 수준까지 끌어올릴 수 있다.

즉, `deft_loss_v4`는 “많이 가져오기”보다 “나쁜 sample을 더 조심스럽게 걸러내기”에 더 가까운 모델이다.

## 5. 왜 원래 v4가 아직 더 좋은가

원래 v4가 전체 F1과 AUC-PR에서 더 좋은 이유는 크게 두 가지로 보인다.

첫째, 원래 v4의 binary MLP는 모델 capacity가 더 크다.

```text
원래 v4/v3 binary MLP:
Linear(input -> 512)
GELU
Dropout(0.3)
Linear(512 -> 256)
GELU
Dropout(0.3)
Linear(256 -> 1)
```

반면 `deft_loss_v4`는 04 실험 구조를 맞추기 위해 작은 MLP를 사용한다.

```text
deft_loss_v4 MLP:
Linear(input -> 32)
ReLU
Dropout(0.2)
Linear(32 -> output)
```

따라서 `deft_loss_v4`는 구조적으로 더 보수적이고 작다. confidence 1/2에 대한 loss weighting 효과는 잘 반영되지만, 복잡한 ranking boundary를 학습하는 힘은 원래 v4의 binary MLP보다 약할 수 있다.

둘째, 원래 v4는 metadata와 기존 confidence model score의 조합이 강하다. 원래 v4는 단일 모델을 새로 크게 만드는 실험이라기보다, 이미 성능이 있던 v3 binary score와 5-class score를 OOF 기반으로 조합하는 실전형 stacking이다. 반면 `deft_loss_v4`는 04 similarity feature 중심의 더 좁은 실험이다.

정리하면 다음과 같다.

| 항목 | 원래 v4 | deft_loss_v4 |
|---|---|---|
| 전체 ranking 성능 | 더 좋음 | 약간 낮음 |
| 최고 F1 | 더 좋음 | 약간 낮음 |
| AUC-PR | 더 좋음 | 약간 낮음 |
| confidence 1/2/3 감지 | 상대적으로 약함 | 더 명시적으로 개선 |
| 해석 가능성 | score stacking 중심 | loss weighting 효과가 명확함 |
| BSD35k 대량 추가 | 유리 | 가능하지만 보수적 선택 필요 |
| noisy sample 제거 | threshold 선택에 따라 강점 | 강점 |

## 6. BSD35k-CS 적용 관점의 추천

`deft_loss_v4`를 단독으로 원래 v4의 완전한 대체재로 쓰는 것은 아직 이르다. 최고 F1과 AUC-PR이 원래 v4보다 약간 낮기 때문이다.

하지만 `deft_loss_v4`는 보조 필터로는 가치가 크다. 특히 confidence 1/2를 더 잘 잡으려는 목적이 있다면 원래 v4 score와 함께 사용하는 것이 좋다.

추천 전략은 다음과 같다.

| 목적 | 추천 score/threshold |
|---|---|
| BSD35k를 많이 가져오기 | 원래 v4 best F1 score 유지 |
| 성능과 noise 제거 균형 | `deft_loss_v4` OOF ridge stacker threshold 0.569 |
| low-confidence 제거를 강하게 보기 | `deft_loss_v4` rank average threshold 0.454 |
| 가장 안전한 clean subset 만들기 | 원래 v4 high score와 `deft_loss_v4` high score의 교집합 |
| ablation 실험용 | 원래 v4 단독, deft_loss_v4 단독, 두 score 교집합을 모두 비교 |

실전적으로는 다음 세 가지 BSD35k subset을 만들면 좋다.

1. **v4_only**
   - 원래 `confidence_filter_v4` score 기준
   - 가장 강한 baseline

2. **deft_loss_v4_balanced**
   - `rank_avg_binary_expected_p45 >= 0.454`
   - 123/45 recall 균형이 가장 좋음

3. **v4_and_deft_intersection**
   - 원래 v4 score도 높고, `deft_loss_v4` score도 높은 sample만 선택
   - sample 수는 줄지만 noise가 가장 적을 가능성이 높음

## 7. 최종 결론

`deft_loss_v4`는 원래 v4를 성능 수치로 완전히 이기지는 못했다. 최고 F1은 원래 v4가 0.7853, `deft_loss_v4`가 0.7832이며, AUC-PR도 원래 v4가 0.8421로 더 높다.

그러나 이 실험은 실패가 아니다. `deft_loss_v4`는 confidence 1과 2에 loss를 더 준 효과가 실제로 low-confidence 감지 성향으로 이어진다는 것을 보여준다. 특히 macro-F1 최적과 balanced recall 조건에서 123 recall을 0.64~0.68까지 올릴 수 있었고, 이는 BSD35k noisy label filtering에서 중요한 성질이다.

따라서 최종 판단은 다음과 같다.

- 원래 v4는 여전히 가장 강한 단독 filtering baseline이다.
- `deft_loss_v4`는 원래 v4를 대체하기보다는 low-confidence noise를 더 보수적으로 제거하는 보조 필터로 쓰는 것이 좋다.
- BSD35k 최종 selection은 원래 v4 단독보다, 원래 v4와 `deft_loss_v4`를 함께 사용한 교집합 또는 multi-score threshold 전략이 더 유망하다.

## 8. 저장 파일

- OOF prediction: `deft논문기반_실험/deft_loss_v4_outputs/predictions/BSD10k_oof_deft_loss_v4_scores_with_stackers.csv`
- threshold summary: `deft논문기반_실험/deft_loss_v4_outputs/reports/deft_loss_v4_threshold_summary.csv`
- best thresholds: `deft논문기반_실험/deft_loss_v4_outputs/reports/deft_loss_v4_best_by_objective.csv`
- PR curves: `deft논문기반_실험/deft_loss_v4_outputs/plots/deft_loss_v4_pr_curves.png`
- threshold F1 plot: `deft논문기반_실험/deft_loss_v4_outputs/plots/deft_loss_v4_threshold_f1.png`
