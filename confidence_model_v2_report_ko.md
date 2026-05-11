# BSD10k Confidence 분류기 v2 재실험 보고서

작성일: 2026-05-10
실행 장치: cpu
총 실행 시간: 7.2분

## 1. Stage 0 ceiling estimate 결과

| baseline                                        | mae    | accuracy | quadratic_weighted_kappa | note                                             |
| ----------------------------------------------- | ------ | -------- | ------------------------ | ------------------------------------------------ |
| Random ±1 perturbation                          | 0.5001 |          |                          | 50% of labels moved to a valid adjacent class    |
| Majority class = 4                              | 0.5360 | 0.5518   | 0.0000                   | All samples predicted as confidence 4            |
| Class prior random sampling                     | 0.7646 |          |                          | Mean over 500 seeded draws; std=0.0064           |
| Previous best: ensemble_avg_best_ordinal_towers | 0.5088 | 0.5888   | 0.3768                   | From confidence_model_true_5class_experiments.py |

## 2. Stage 1 feature ablation

| experiment      | feature_set | loss | feature_dim | mae    | spearman | accuracy | quadratic_weighted_kappa | macro_f1 |
| --------------- | ----------- | ---- | ----------- | ------ | -------- | -------- | ------------------------ | -------- |
| A_baseline_ce   | baseline    | ce   | 1056        | 0.5145 | 0.4792   | 0.5768   | 0.3299                   | 0.3075   |
| B_similarity_ce | similarity  | ce   | 1060        | 0.5151 | 0.4772   | 0.5784   | 0.3376                   | 0.3129   |
| C_clap_stats_ce | clap_stats  | ce   | 1064        | 0.5162 | 0.4758   | 0.5801   | 0.3260                   | 0.3116   |
| D_everything_ce | everything  | ce   | 2096        | 0.5181 | 0.4698   | 0.5729   | 0.3294                   | 0.3158   |

## 3. Stage 2 loss 비교

| experiment                        | feature_set | loss              | feature_dim | mae    | spearman | accuracy | quadratic_weighted_kappa | macro_f1 |
| --------------------------------- | ----------- | ----------------- | ----------- | ------ | -------- | -------- | ------------------------ | -------- |
| stage2_baseline_emd               | baseline    | emd               | 1056        | 0.5087 | 0.4781   | 0.5809   | 0.3375                   | 0.3062   |
| stage2_baseline_ordinal_smoothing | baseline    | ordinal_smoothing | 1056        | 0.5142 | 0.4798   | 0.5800   | 0.3433                   | 0.3139   |
| stage2_baseline_ce                | baseline    | ce                | 1056        | 0.5145 | 0.4792   | 0.5768   | 0.3299                   | 0.3075   |
| stage2_baseline_expected_mse_aux  | baseline    | expected_mse_aux  | 1056        | 0.5146 | 0.4754   | 0.5779   | 0.3258                   | 0.3009   |

## 4. 이전 best vs 새 best

| model                                          | mae    | spearman | accuracy | quadratic_weighted_kappa | macro_f1 |
| ---------------------------------------------- | ------ | -------- | -------- | ------------------------ | -------- |
| previous best ensemble_avg_best_ordinal_towers | 0.5088 |          | 0.5888   | 0.3768                   | 0.3339   |
| new best stage2_baseline_emd                   | 0.5087 | 0.4781   | 0.5809   | 0.3375                   | 0.3062   |

## 5. Confusion matrix

|        | pred 1 | pred 2 | pred 3 | pred 4 | pred 5 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| true 1 | 0      | 13     | 54     | 39     | 0      |
| true 2 | 0      | 91     | 356    | 302    | 0      |
| true 3 | 0      | 64     | 1203   | 2011   | 2      |
| true 4 | 0      | 44     | 960    | 4962   | 79     |
| true 5 | 0      | 2      | 37     | 629    | 108    |

## 6. BSD35k-CS 적용 결과 요약

- 저장 파일: `outputs/confidence_model_v2/predictions/BSD35k-CS_predicted_v2.csv`
- 적용 row 수: 31,464
- 평균 predicted confidence score: 3.4787
- 표준편차: 0.2783

예측 class 분포:

| predicted_confidence_class | n     | rate   |
| -------------------------- | ----- | ------ |
| 2                          | 122   | 0.0039 |
| 3                          | 10337 | 0.3285 |
| 4                          | 20985 | 0.6670 |
| 5                          | 20    | 0.0006 |

예측 score 요약:

|       | predicted_confidence_score |
| ----- | -------------------------- |
| count | 31464.0000                 |
| mean  | 3.4787                     |
| std   | 0.2783                     |
| min   | 2.3532                     |
| 10%   | 3.1241                     |
| 25%   | 3.2991                     |
| 50%   | 3.4766                     |
| 75%   | 3.6707                     |
| 90%   | 3.8318                     |
| max   | 4.5999                     |

## 7. 결론

1. 새 모델의 MAE 0.5087는 Random ±1 perturbation ceiling proxy 0.5001와 +0.0087 차이다. 이 proxy가 human-level adjacent-label noise를 뜻한다면, 모델은 그 근처까지 왔는지 여부를 이 차이로 판단할 수 있다.

2. feature engineering 개선폭은 baseline 대비 +0.0000 MAE 감소이고, loss 변경 개선폭은 CE 대비 +0.0057 MAE 감소다. 이번 실행에서는 ordinal-aware loss 쪽 기여가 더 컸다.

3. best Spearman은 0.4781이다. MAE가 ceiling proxy의 ±0.02 안이면 추가 모델 복잡도보다 라벨 품질/추가 annotation 점검이 우선이고, 그보다 멀면 confidence-relevant feature 추가가 다음 단계다.
