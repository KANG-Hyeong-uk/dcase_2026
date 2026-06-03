# DCASE 2026 Task 1 실험 정리 큰 틀

Notion에 옮길 최종 실험 정리의 목차 초안이다.

`DCASE2026_Task1_Colab.ipynb`는 현재 정리 대상에서 제외한다.

---

## 0. 정리 원칙

이번 정리는 단순히 성능표를 나열하는 것이 아니라, 실험이 발전한 순서를 따라간다.

핵심 흐름:

1. BSD10k-only baseline으로 기준 성능 확인
2. BSD10k-only에서 loss만 바꿔 계층 loss 효과 확인
3. BSD35k를 직접 투입했을 때 성능 하락 확인
4. BSD35k label noise 때문에 confidence 분류기가 필요해짐
5. 일반 MLP, 1-layer classification/regression, v2/v3/v4, clean label, XGBoost, sampling, weighted loss, DEFT 기반 feature 실험을 통해 confidence score를 개선
6. confidence score로 선별한 데이터가 실제 DCASE baseline 학습에 도움이 되는지 검증
7. 최종적으로 confidence score 자체 성능과 downstream DCASE 성능은 다른 문제라는 결론 도출

---

## 1. 전체 실험 파일 인벤토리

### 1.1 제외 파일

| 파일 | 처리 |
|---|---|
| `DCASE2026_Task1_Colab.ipynb` | 이번 Notion 정리에서는 제외 |

### 1.2 주요 실험 파일 묶음

| 실험 묶음 | 주요 파일 | 역할 |
|---|---|---|
| BSD10k baseline | `DCASE2026_Task1_VSCode.ipynb` | BSD10k-only 공식 baseline 재현 |
| BSD10k loss 수정 | 사용자 제공 결과, 관련 loss 분석 | `CE + HATR TopClassLoss + HATR ContrastiveLoss` 효과 확인 |
| HATR confidence1 filtering | 사용자 제공 HATR confidence1 filtering 결과 | confidence1 전체 제거와 class-selective 유지 전략 비교 |
| BSD35k 직접 투입 | `DCASE2026_Task1_VSCode_BSD35kTrain_BSD10kTest.ipynb` | BSD35k train/val, BSD10k test에서 성능 하락 확인 |
| 일반 confidence MLP | `kang_study.ipynb`, `confidence_mlp_tsne_mapping.ipynb`, `confidence_regression_embeddings_class.ipynb` | 초기 MLP confidence 예측과 representation 분석 |
| 1-layer classification/regression | `up_sampling_confi4/1layer_mlp_*` | 단순 모델 baseline, collapse 확인 |
| v2 3-layer MLP | `confidence_model_v2_experiments.py`, `confidence_model_v2_report_ko.md` | 5-class confidence MLP |
| v3 3-layer MLP | `notebooks/confidence_filter_v3_binary.ipynb`, `confidence_filter_v3_report_ko.md` | binary high-confidence filter |
| v4 ensemble | `notebooks/confidence_filter_v4_score_stacking.ipynb`, `confidence_filter_v4_report_ko.md` | v2 expected score + v3 binary score ensemble |
| v4 BSD35k baseline 영향 | `notebooks/v4_35k_baseline모델.ipynb` | v4 score로 BSD35k를 선별해 downstream baseline 영향 확인 |
| clean label v2/v3/v4 | `notebooks/clean_label,v2,v3,v4.ipynb`, `clean_v2_v3_v4_report_ko.md` | clean feature가 confidence 성능을 개선하는지 검증 |
| confidence 모델별 baseline 재학습 | `baseline_confidnce_train/*.ipynb` | confidence score로 train data를 자른 뒤 DCASE baseline 재학습 |
| 5-17 confidence 실험 | `confidence 분류기 5-17실험/*.ipynb` | MLP classification/regression/v5 pipeline |
| XGBoost/RF/sampling | `notebooks/xg_*`, `notebooks/rf_*`, `notebooks/over_sample_*`, `notebooks/down_sample_*`, `up_sampling_confi4/*` | tree model, sampling, upsampling 비교 |
| DEFT similarity feature | `deft논문기반_실험/01_*`, `02_*`, `03_*`, `04_similarity_feature_mlp_confidence.ipynb` | audio-text/class similarity, margin feature 분석 |
| weighted loss | `deft논문기반_실험/04_weighted_loss_confidence12*.ipynb` | confidence 1/2 minority 보정 |
| DEFT loss v4 / OOF stacking | `deft논문기반_실험/deft_loss_v4.ipynb` | OOF score stacking, threshold objective 비교 |
| BSD35k label quality | `notebooks/01_bsd35k_classifier_scoring_analysis.ipynb`, `experiments/bsd35k_*` | BSD35k label reliability 분석 |

---

## 2. BSD10k-only baseline 실험

### 2.1 공식 baseline 재현

| 항목 | 내용 |
|---|---|
| Notebook | `DCASE2026_Task1_VSCode.ipynb` |
| 목적 | DCASE 2026 Task 1 공식 baseline 재현 |
| 학습 데이터 | BSD10k-v1.2 |
| 평가 데이터 | BSD10k-v1.2 fold test |
| split | Stratified K-Fold 5-fold |
| mode | `both`, `audio` |
| 모델 | `BaseClassifier(hidden_size=128)` |
| 입력 | CLAP audio embedding + CLAP text embedding |
| loss | 기본 Cross Entropy |

### 2.2 결과 요약

| mode | accuracy | top accuracy | macro accuracy | hierarchical accuracy | hierarchical F1 | 코멘트 |
|---|---:|---:|---:|---:|---:|---|
| audio | 77.31 ± 0.39 | 87.92 ± 0.66 | 70.67 ± 0.96 | 77.20 ± 0.91 | 75.82 ± 0.60 | audio-only 기준선 |
| both | 79.63 ± 0.39 | 88.88 ± 0.15 | 74.02 ± 1.05 | 79.45 ± 0.67 | 78.33 ± 0.41 | 최종 비교의 기본 기준선 |

### 2.3 코멘트

- BSD10k 내부 평가에서는 baseline이 정상적으로 재현된다.
- `both` 입력이 `audio` 단독보다 전반적으로 더 좋다.
- 이후 BSD35k 추가, confidence filtering, loss 변경 실험은 이 성능을 기준으로 비교한다.

---

## 3. BSD10k-only loss 수정 실험

데이터는 BSD10k-only로 유지하고 loss만 수정한 실험이다.  
BSD35k noise와 무관하게 HATR 계층 loss가 metric에 어떤 영향을 주는지 확인한다.

### 3.1 실험 조건

| 항목 | 내용 |
|---|---|
| 데이터 | BSD10k-v1.2 only |
| 모델 | baseline `BaseClassifier` |
| 입력 | CLAP audio + text |
| 기존 loss | Cross Entropy |
| 수정 loss | `CE + HATR TopClassLoss + HATR ContrastiveLoss` |
| 평가 | 5-fold 결과 평균 |

### 3.2 fold별 결과

| fold | accuracy | top accuracy | macro accuracy | macro top accuracy | hierarchical accuracy | hierarchical precision | hierarchical recall | hierarchical F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 79.105839 | 88.640511 | 73.283408 | 84.278713 | 78.781061 | 76.569661 | 76.135745 | 75.855663 |
| 2 | 79.872204 | 89.502510 | 74.889105 | 85.830701 | 80.359903 | 78.565896 | 77.587521 | 77.954004 |
| 3 | 79.096303 | 88.863533 | 74.110313 | 85.128963 | 79.619638 | 77.736250 | 76.971403 | 77.022929 |
| 4 | 80.419900 | 88.954815 | 75.517113 | 85.296264 | 80.406689 | 78.195648 | 78.029404 | 77.865695 |
| 5 | 80.602465 | 89.091739 | 74.512775 | 85.066166 | 79.789470 | 79.331116 | 77.247470 | 77.697765 |

### 3.3 평균 결과

| metric | mean | std | mean ± std |
|---|---:|---:|---|
| accuracy | 79.819342 | 0.633816 | 79.82 ± 0.63 |
| top accuracy | 89.010622 | 0.286449 | 89.01 ± 0.29 |
| macro accuracy | 74.462543 | 0.749803 | 74.46 ± 0.75 |
| macro top accuracy | 85.120161 | 0.499381 | 85.12 ± 0.50 |
| hierarchical accuracy | 79.791352 | 0.591957 | 79.79 ± 0.59 |
| hierarchical precision | 78.079714 | 0.918047 | 78.08 ± 0.92 |
| hierarchical recall | 77.194309 | 0.636407 | 77.19 ± 0.64 |
| hierarchical F1 | 77.279211 | 0.783156 | 77.28 ± 0.78 |

### 3.4 코멘트

- CE baseline보다 accuracy와 hierarchical accuracy는 소폭 상승했다.
- hierarchical F1은 CE baseline보다 낮아질 수 있어, 모든 계층 metric이 일관되게 좋아진 것은 아니다.
- 계층 loss는 유효하지만, metric별 trade-off를 보고 선택해야 한다.

---

## 4. BSD35k 직접 투입 실험

`DCASE2026_Task1_VSCode.ipynb`는 BSD10k-only baseline이므로 이 카테고리에 넣지 않는다.  
이 섹션은 BSD35k를 학습에 직접 사용했을 때의 문제를 다룬다.

### 4.1 BSD35k train/validation + BSD10k test

| 항목 | 내용 |
|---|---|
| Notebook | `DCASE2026_Task1_VSCode_BSD35kTrain_BSD10kTest.ipynb` |
| 목적 | BSD35k-CS를 그대로 train/validation에 사용했을 때 BSD10k test 성능 확인 |
| train/validation | BSD35k-CS |
| test | BSD10k-v1.2 original fold test split |
| 데이터 규모 | BSD35k-CS usable 31,464 / BSD10k test source 10,956 |
| 유지한 조건 | 원래 baseline의 모델 구조, 학습 설정, 평가 방식 유지 |

### 4.2 결과 요약

출처: `dcase2026_task1_baseline/bsd35k_model_ouput/summary_metrics.txt`

| mode | accuracy | top accuracy | macro accuracy | hierarchical accuracy | hierarchical F1 | 코멘트 |
|---|---:|---:|---:|---:|---:|---|
| audio | 50.56 ± 1.35 | 68.21 ± 1.18 | 41.50 ± 1.09 | 52.20 ± 0.95 | 49.15 ± 1.29 | BSD10k-only 대비 크게 하락 |
| both | 56.20 ± 1.47 | 72.26 ± 1.53 | 48.30 ± 0.87 | 57.87 ± 1.10 | 55.63 ± 0.53 | audio보다 낫지만 여전히 낮음 |

### 4.3 코멘트

- BSD35k를 그대로 학습에 사용하면 BSD10k test에서 성능이 크게 떨어진다.
- BSD35k의 label distribution, pseudo-label 품질, class boundary noise가 downstream 성능을 강하게 훼손한 것으로 볼 수 있다.
- 이 결과가 confidence filtering 실험으로 넘어가는 직접적인 이유다.

---

## 5. confidence 모델 실험 계보

이 섹션은 confidence 모델이 어떻게 발전했는지 정리한다.

중요한 구분:

- 일반 MLP 실험: confidence label을 예측할 수 있는지 초기 확인
- 1-layer classification/regression: 단순 모델의 한계 확인
- v2/v3: 3-layer MLP 기반 핵심 confidence 모델
- v4: v2와 v3를 앙상블한 score
- clean label v2/v3/v4: feature를 더 깨끗하게 만들면 좋아지는지 검증
- v4_35k_baseline: v4 score가 실제 downstream DCASE baseline에 주는 영향 확인

---

### 5.1 일반 MLP confidence 실험

| 파일 | 역할 |
|---|---|
| `kang_study.ipynb` | confidence MLP 초기 실험 |
| `kang_study_experiment_report_ko.md` | kang_study 결과 보고서 |
| `kang_study_tsne_cluster_analysis_ko.md` | hidden feature / t-SNE cluster 분석 |
| `notebooks/confidence_mlp_tsne_mapping.ipynb` | confidence MLP representation 시각화 |
| `notebooks/confidence_regression_embeddings_class.ipynb` | embedding + class 기반 confidence regression |

정리할 내용:

| 항목 | 작성할 내용 |
|---|---|
| 목적 | confidence label 1-5가 audio/text/class embedding으로 예측 가능한지 확인 |
| 주요 결과 | accuracy, MAE, macro F1, t-SNE cluster |
| 통찰 | confidence 4 majority, minority class 분리 어려움, representation이 label을 완전히 분리하지 못함 |

---

### 5.2 1-layer classification / regression

| 모델 | 대표 결과 | 통찰 |
|---|---:|---|
| Simple Classification, 1-layer | acc 0.5757, macro F1 0.2834 | confidence 4 collapse, class 1/5 분리 약함 |
| Simple Regression, 1-layer | MAE 0.5769, rounded acc 0.5123 | 평균 근처 collapse, majority-class proxy보다도 불리 |

코멘트:

- 1-layer 모델은 가장 단순한 sanity check다.
- classification은 confidence 4로 강하게 몰린다.
- regression은 ordinal 구조를 기대했지만 실제로는 score tail이 안정적이지 않았다.
- 따라서 BSD35k filtering 주 모델로 쓰기 어렵다.

---

### 5.3 v2 3-layer MLP: 5-class confidence model

| 항목 | 내용 |
|---|---|
| 파일 | `confidence_model_v2_experiments.py`, `confidence_model_v2_report_ko.md` |
| 구조 | 3-layer MLP 계열 |
| target | confidence 1-5 |
| 핵심 출력 | predicted class, class probability, expected confidence score, P4+P5 |

| 지표 | 값 |
|---|---:|
| MAE | 0.5087 |
| accuracy | 0.5809 |
| macro F1 | 0.3062 |

통찰:

- hard 5-class prediction은 class 4 쏠림 때문에 불안정하다.
- 하지만 expected confidence score는 high-confidence ranking signal로 의미가 있다.
- v2는 단독 hard predictor보다 v4 ensemble의 구성 요소로 더 가치가 있다.

---

### 5.4 v3 3-layer MLP: binary high-confidence filter

| 항목 | 내용 |
|---|---|
| 파일 | `notebooks/confidence_filter_v3_binary.ipynb`, `confidence_filter_v3_report_ko.md` |
| 구조 | 3-layer MLP 계열 |
| target | confidence >= 4 vs confidence <= 3 |
| 핵심 출력 | high-confidence probability |

| threshold | precision | recall | F1 | AUC-PR | 비고 |
|---:|---:|---:|---:|---:|---|
| 0.4 | 0.6904 | 0.9021 | 0.7822 | 0.8260 | F1-optimal |
| 0.5 | 0.7343 | 0.8131 | 0.7717 | 0.8260 | 균형 기준 |
| 0.7 | 0.8237 | 0.5419 | 0.6537 | 0.8260 | strict filtering |
| 0.8 | 0.8668 | 0.3939 | 0.5417 | 0.8260 | high precision |

통찰:

- binary target이 실제 filtering 목적과 잘 맞는다.
- 단독 ranking AUC-PR은 v2 expected score보다 낮다.
- threshold를 올리면 precision은 올라가지만 retained ratio가 빠르게 줄어든다.

---

### 5.5 v4 ensemble: v2 + v3 rank average

| 항목 | 내용 |
|---|---|
| 파일 | `notebooks/confidence_filter_v4_score_stacking.ipynb`, `confidence_filter_v4_report_ko.md` |
| 구조 | v2 expected confidence score + v3 binary probability의 rank-average ensemble |
| 목적 | BSD35k high-confidence sample ranking |

| threshold | retained_samples | retained_ratio | expected_precision_from_oof | expected_recall_from_oof | expected_f1_from_oof | 기준 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.180 | 26,150 | 0.8311 | 0.6859 | 0.9183 | 0.7853 | 널널 | 
| 0.500 | 15,566 | 0.4947 | 0.8040 | 0.6435 | 0.7148 |
| 0.600 | 12,357 | 0.3927 | 0.8409 | 0.5285 | 0.6491 |
| 0.700 | 9,197 | 0.2923 | 0.8813 | 0.4092 | 0.5589 |
| 0.800 | 6,069 | 0.1929 | 0.9284 | 0.2834 | 0.4342 |
| 0.900 | 3,008 | 0.0956 | 0.9616 | 0.1284 | 0.2266 | 매우 엄격 |

통찰:

- v4의 핵심은 5-class accuracy를 올린 것이 아니라 high-confidence sample ranking을 안정화한 것이다.
- AUC-PR 0.8421로 v3 binary보다 좋다.
- strict threshold에서 false high-confidence가 더 빠르게 줄어든다.

---

### 5.6 clean label v2/v3/v4 실험

| 항목 | 내용 |
|---|---|
| Notebook | `notebooks/clean_label,v2,v3,v4.ipynb` |
| Report | `clean_v2_v3_v4_report_ko.md` |
| 목적 | top_class one-hot, metadata length feature를 제거한 clean input이 confidence pipeline을 개선하는지 확인 |
| clean input | audio embedding 512 + text embedding 512 + class one-hot 23 = 1047 dims |
| original input | 기존 confidence pipeline input |

#### v2 clean vs original

| Model | Feature dim | MAE | Accuracy | QWK | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Original v2 EMD | 1056 | 0.5087 | 0.5809 | 0.3375 | 0.3062 |
| Clean v2 EMD | 1047 | 0.5118 | 0.5771 | 0.3243 | 0.3036 |
| Original v2 Ordinal Smoothing | 1056 | 0.5142 | 0.5800 | 0.3433 | 0.3139 |
| Clean v2 Ordinal Smoothing | 1047 | 0.5181 | 0.5740 | 0.3079 | 0.3062 |
| Original v2 CE | 1056 | 0.5145 | 0.5768 | 0.3299 | 0.3075 |
| Clean v2 CE | 1047 | 0.5169 | 0.5750 | 0.3048 | 0.3108 |

#### v3 clean vs original

| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC-PR | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original v3 F1-optimal | 0.400 | 0.6872 | 0.6904 | 0.9021 | 0.7822 | 0.8260 | 0.7461 |
| Clean v3 F1-optimal | 0.395 | 0.6899 | 0.6946 | 0.8959 | 0.7825 | 0.8190 | 0.7387 |
| Original v3 @ 0.5 | 0.500 | 0.7004 | 0.7343 | 0.8131 | 0.7717 | 0.8260 | 0.7461 |
| Clean v3 @ 0.5 | 0.500 | 0.6933 | 0.7270 | 0.8126 | 0.7674 | 0.8190 | 0.7387 |

#### v4 clean vs original

| Model | Best Method | Threshold | Precision | Recall | F1 | AUC-PR | AUC-ROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Original v4 | Rank average: binary + expected score | 0.180 | 0.6859 | 0.9183 | 0.7853 | 0.8421 | 0.7593 |
| Clean v4 | OOF logistic stacker | 0.360 | 0.6729 | 0.9355 | 0.7828 | 0.8354 | 0.7535 |

#### clean label 실험 통찰

- clean input이 항상 성능을 개선하지는 않았다.
- v2는 대부분 original이 더 좋았다.
- v3는 F1-optimal에서는 clean이 아주 근소하게 좋아 보이지만, ranking metric인 AUC-PR/AUC-ROC는 original이 더 좋다.
- v4도 original이 AUC-PR과 F1 기준으로 더 안정적이다.
- 중요한 통찰: top_class/meta feature는 단순 noise가 아니라 confidence prediction에서 calibration signal로 작동했을 수 있다.
- 결론: confidence filtering 목적에서는 clean input보다 original v4를 유지하는 것이 더 안정적이다.

---

## 6. v4 score로 BSD35k baseline 영향 확인

대상 파일:

- `notebooks/v4_35k_baseline모델.ipynb`
- 결과 저장 위치: `baseline_confidnce_train/outputs/v4_35k_baseline_model/`

### 6.1 실험 목적

v4 confidence score가 confidence 분류기 내부 지표에서만 좋은지, 아니면 실제 DCASE Task 1 baseline 학습 데이터 선별에도 도움이 되는지 확인한다. 즉, BSD35k-CS를 무조건 추가하는 방식과 v4 score로 선별해서 추가하는 방식을 비교해 `confidence filtering -> downstream ASC 성능`의 연결을 검증한다.

### 6.2 실험 세팅

| 항목 | 내용 |
|---|---|
| 기준 데이터 | BSD10k |
| 분할 | BSD10k full 10,956개를 train_pool 80% 8,764개 / final_test 20% 2,192개로 고정 |
| 추가 데이터 | BSD35k-CS |
| BSD35k 사용 가능 row | 31,464개 |
| embedding 누락 | audio 0개, text 0개 |
| class 수 | 23개 모두 유지 |
| baseline 모델 | `dcase2026_task1_baseline`의 `BaseClassifier` |
| 입력 mode | both, audio + text |
| 평가 방식 | train_pool + BSD35k subset을 5-fold로 학습하고 동일한 BSD10k final_test 2,192개에서 평가 |
| v4 score 변환 | `predicted_confidence_score = 1 + 4 * v4_filter_score` |
| subset 기준 | ge2: score >= 2, ge3: score >= 3, ge4: score >= 4 |

BSD35k subset 규모:

| subset | threshold | BSD35k 추가 수 | BSD35k 유지 비율 | class 수 |
|---|---:|---:|---:|---:|
| v4_ge2 | 2.0 | 23,851 | 75.80% | 23 |
| v4_ge3 | 3.0 | 15,566 | 49.47% | 23 |
| v4_ge4 | 4.0 | 7,633 | 24.26% | 23 |
| v4_all | 없음 | 31,464 | 100.00% | 23 |

### 6.3 결과

| 조건 | 추가 BSD35k 수 | accuracy | top accuracy | macro accuracy | macro top accuracy | H-Acc | H-F1 | 해석 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v4_all | 31,464 | 77.35 ± 0.45 | 87.02 ± 0.62 | 70.37 ± 0.66 | 82.11 ± 0.58 | 76.24 ± 0.44 | 75.29 ± 0.61 | 전체 BSD35k를 넣으면 noise가 그대로 들어와 성능이 가장 낮음 |
| v4_ge2 | 23,851 | 77.38 ± 0.58 | 87.00 ± 0.49 | 71.13 ± 0.78 | 82.71 ± 0.65 | 76.92 ± 0.69 | 75.77 ± 0.59 | ge2는 양은 많지만 여전히 noisy sample이 많음 |
| v4_ge3 | 15,566 | 78.74 ± 0.47 | 87.73 ± 0.52 | 72.62 ± 0.63 | 83.36 ± 0.87 | 77.99 ± 0.73 | 76.89 ± 0.72 | noise 감소 효과가 뚜렷해짐 |
| v4_ge4 | 7,633 | 79.28 ± 0.83 | 88.34 ± 0.69 | 73.45 ± 0.65 | 84.16 ± 0.64 | 78.80 ± 0.63 | 77.72 ± 0.72 | 가장 좋은 downstream 결과 |

best fold 기준:

| 조건 | best fold | accuracy | top accuracy | macro accuracy | H-Acc | H-F1 |
|---|---|---:|---:|---:|---:|---:|
| v4_all | fold 2 | 77.97 | 88.05 | 70.72 | 76.88 | 76.16 |
| v4_ge2 | fold 4 | 77.92 | 87.00 | 71.91 | 77.62 | 76.39 |
| v4_ge3 | fold 1 | 79.11 | 88.00 | 73.38 | 78.67 | 77.48 |
| v4_ge4 | fold 4 | 80.02 | 89.05 | 73.92 | 79.37 | 78.28 |

### 6.4 confusion matrix 기반 분석

best fold confusion matrix 기준으로 v4_ge4에서 강한 class:

| class | row-normalized recall | 해석 |
|---|---:|---|
| is-w | 97.35% | 매우 안정적으로 구분됨 |
| is-k | 93.06% | 같은 `is-*` top class 내부에서도 비교적 잘 분리됨 |
| is-s | 92.45% | 높은 recall 유지 |
| sp-s | 91.30% | sports 계열 중 가장 안정적 |
| is-p | 89.34% | 주요 오분류는 `fx-o`, `fx-h` 방향 일부 |
| fx-h | 87.69% | effect 계열 중 안정적 |
| fx-n | 87.12% | `ss-n`으로 일부 혼동 |
| m-sp | 84.67% | `m-si`, `m-m`으로 일부 혼동 |
| fx-o | 83.82% | `fx-m`으로 7.05% 혼동 |
| m-si | 82.39% | `m-m`, `is-s`, `is-w`, `ss-s`로 분산 혼동 |

취약 class와 주요 오분류:

| class | recall | 주요 오분류 | 해석 |
|---|---:|---|---|
| sp-c | 42.86% | `ss-u` 22.86%, `fx-h` 20.00%, `sp-s` 8.57% | sample 수/특징 경계가 불안정한 class |
| fx-a | 47.06% | `ss-n` 41.18%, `fx-h` 5.88%, `fx-n` 5.88% | effect 계열과 sound-source 계열 사이에서 혼동 |
| fx-ex | 41.67% | `fx-h` 18.33%, `m-si` 8.33%, `sp-p` 8.33% | rare/ambiguous class 성격이 강함 |
| ss-i | 21.95% | `ss-u` 39.02%, `sp-p` 14.63%, `sp-c` 12.20% | `ss-i`와 `ss-u`의 경계가 가장 취약 |
| ss-u | 71.33% | `fx-v` 4.90%, `ss-n` 4.20%, `sp-c` 3.50% | recall은 중간 이상이나 주변 class로 분산 오분류 |

v4_ge3 best fold와 비교하면 v4_ge4는 전체적으로 더 보수적인 subset이라 `is-e`, `sp-s`, `fx-h`, `fx-ex`, `fx-el` 등 일부 class의 recall이 좋아졌다. 반면 `ss-i`, `ss-u`, `sp-c`, `fx-a`처럼 class 자체가 모호하거나 sample 수가 적은 경우에는 v4 score로 filtering해도 구조적 혼동이 남아 있다.

### 6.5 해석

- v4 score threshold가 높아질수록 downstream 성능이 좋아지는 단조적 경향이 보인다. `v4_all -> ge2 -> ge3 -> ge4` 순서로 accuracy, macro accuracy, H-Acc, H-F1이 모두 개선된다.
- 이 결과는 v4 score가 단순 confidence classifier 점수가 아니라 BSD35k label quality ranking으로도 작동한다는 근거다.
- 하지만 BSD10k-only baseline과 비교하면 v4_ge4도 압도적으로 이기는 구조는 아니다. BSD35k를 추가하면 데이터 양은 늘지만 domain gap, pseudo-label noise, class imbalance가 동시에 들어오기 때문이다.
- 따라서 v4 score의 역할은 “BSD35k를 그대로 학습에 넣어도 된다”가 아니라, “BSD35k를 넣어야 한다면 최소한 v4_ge3/ge4처럼 강하게 filtering해야 한다”에 가깝다.
- confusion matrix상 취약 class는 confidence score만으로 해결되지 않는다. `sp-c`, `fx-a`, `fx-ex`, `ss-i`는 class-balanced sampling, class별 threshold, top-class-aware loss가 추가로 필요하다.

---

## 7. hier loss + v4 BSD35k baseline 실험

대상 파일:

- `notebooks/hier_loss_v4_35k_baseline모델.ipynb`
- 결과 저장 위치: `baseline_confidnce_train/outputs/v4_35k_hier_loss/`

### 7.1 실험 목적

6번 실험은 BSD35k 선별 기준만 v4 score로 바꾸고 baseline loss는 CE 그대로 둔 실험이다. 7번 실험은 데이터 조건은 6번과 완전히 동일하게 유지하고, 손실함수만 `CE + L_Top + L_Contr` 형태의 계층 손실로 바꿔서 v4-selected BSD35k를 더 잘 활용할 수 있는지 확인한다.

### 7.2 실험 세팅

| 항목 | 내용 |
|---|---|
| 기준 notebook | `v4_35k_baseline모델.ipynb`와 동일 |
| 고정 split | BSD10k train_pool 80% 8,764개 / final_test 20% 2,192개 |
| 추가 데이터 | v4_all, v4_ge2, v4_ge3, v4_ge4 |
| baseline model | 동일한 `BaseClassifier` |
| 변경점 | loss만 변경 |
| classification loss | CE |
| top-class loss | `L_Top` |
| contrastive loss | `L_Contr` |
| sweep | `lam_top` sweep, `lam_contr=0.1`, `tau=0.07` |
| output | fold별 metric, training history, confusion matrix |

### 7.3 현재 확인된 결과

현재 저장된 결과는 `v4_all`, `v4_ge2`, `v4_ge3`는 여러 `lam_top` 조합이 들어 있고, `v4_ge4`는 `lam_top=0.1` 5-fold와 `lam_top=0.3` 일부 fold가 partial로 남아 있다. 따라서 문서화에서는 먼저 완료된 조합을 기준으로 쓰고, 나머지는 재실행 후 평균표를 채우는 구조가 좋다.

완료된 v4_ge4, `lam_top=0.1`:

| fold | accuracy | top accuracy | macro accuracy | macro top accuracy | H-Acc | H-Precision | H-Recall | H-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fold 0 | 79.01 | 88.55 | 73.17 | 84.93 | 79.05 | 78.62 | 77.58 | 77.73 |
| fold 1 | 79.97 | 88.46 | 72.92 | 84.44 | 78.68 | 80.34 | 77.24 | 78.18 |
| fold 2 | 79.97 | 88.87 | 74.51 | 84.58 | 79.55 | 79.08 | 78.29 | 78.54 |
| fold 3 | 79.38 | 87.96 | 73.95 | 83.98 | 78.97 | 78.54 | 77.71 | 77.64 |
| fold 4 | 78.83 | 88.09 | 73.37 | 84.39 | 78.88 | 77.78 | 77.50 | 77.29 |

v4_ge4, `lam_top=0.1` 평균:

| 조건 | accuracy | top accuracy | macro accuracy | macro top accuracy | H-Acc | H-F1 |
|---|---:|---:|---:|---:|---:|---:|
| CE baseline, v4_ge4 | 79.28 ± 0.83 | 88.34 ± 0.69 | 73.45 ± 0.65 | 84.16 ± 0.64 | 78.80 ± 0.63 | 77.72 ± 0.72 |
| hier loss, v4_ge4, lam_top=0.1 | 약 79.63 | 약 88.38 | 약 73.59 | 약 84.46 | 약 79.03 | 약 77.87 |

해석:

- 동일한 v4_ge4 데이터에서 계층 손실을 추가하면 CE baseline 대비 H-Acc와 H-F1이 소폭 개선되는 방향이다.
- 개선 폭은 크지 않지만, top-class 구조를 직접 loss에 넣었을 때 hierarchical metric이 약간 좋아지는 경향은 있다.
- accuracy/top accuracy도 크게 무너지지 않으므로 loss가 불안정하게 작동한 것은 아니다.
- 다만 현재 `v4_ge4, lam_top=0.3` 이후가 partial이므로 최종 결론은 `lam_top=0.1/0.3/0.5/1.0` 전체 5-fold 평균을 만든 뒤 확정하는 것이 맞다.

### 7.4 Notion에 채울 최종 비교 틀

| dataset | lam_top | added BSD35k | accuracy | top accuracy | macro accuracy | H-Acc | H-F1 | CE 대비 변화 | 해석 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| v4_all | 0.1 | 31,464 |  |  |  |  |  |  | noise가 큰 subset에서 계층 손실이 도움이 되는지 |
| v4_all | 0.3 | 31,464 |  |  |  |  |  |  |  |
| v4_all | 0.5 | 31,464 |  |  |  |  |  |  |  |
| v4_all | 1.0 | 31,464 |  |  |  |  |  |  |  |
| v4_ge2 | 0.1 | 23,851 |  |  |  |  |  |  |  |
| v4_ge2 | 0.3 | 23,851 |  |  |  |  |  |  |  |
| v4_ge2 | 0.5 | 23,851 |  |  |  |  |  |  |  |
| v4_ge2 | 1.0 | 23,851 |  |  |  |  |  |  |  |
| v4_ge3 | 0.1 | 15,566 |  |  |  |  |  |  |  |
| v4_ge3 | 0.3 | 15,566 |  |  |  |  |  |  |  |
| v4_ge3 | 0.5 | 15,566 |  |  |  |  |  |  |  |
| v4_ge3 | 1.0 | 15,566 |  |  |  |  |  |  |  |
| v4_ge4 | 0.1 | 7,633 | 79.63 | 88.38 | 73.59 | 79.03 | 77.87 | H 계열 소폭 개선 | 현재 가장 먼저 문서화 가능한 완료 조합 |
| v4_ge4 | 0.3 | 7,633 | partial | partial | partial | partial | partial | 확인 필요 | 일부 fold만 저장됨 |

### 7.5 해석 방향

- 6번의 결론이 “v4_ge4처럼 강하게 filtering해야 한다”라면, 7번은 “그렇게 선별한 데이터에 계층 구조 loss를 얹으면 hierarchical metric을 더 밀어올릴 수 있는가”를 보는 실험이다.
- 현재 v4_ge4 lam_top=0.1만 보면 CE 대비 H-Acc/H-F1이 아주 약하게 좋아진다.
- 추가 확인 포인트는 `lam_top`을 높였을 때 H-F1이 더 오르는지, 아니면 class-level accuracy가 희생되는지다.
- confusion matrix 분석은 6번과 같은 방식으로 `sp-c`, `fx-a`, `fx-ex`, `ss-i`가 개선되는지를 중심으로 보면 된다.

---

## 8. XGBoost / RF / sampling / upsampling 실험

### 8.1 tree model 실험

| 실험 | 파일 | 목적 |
|---|---|---|
| XGBoost classification | `notebooks/xg_classfication.ipynb` | confidence 1-5 분류 |
| XGBoost regression | `notebooks/xg_regression.ipynb` | confidence score regression |
| RandomForest classification | `notebooks/rf_classfication.ipynb` | tree baseline |
| RandomForest regression | `notebooks/rf_regression.ipynb` | tree regression baseline |

대표 XGBoost 결과:

| 모델 | feature set | accuracy | macro F1 | MAE | 비고 |
|---|---|---:|---:|---:|---|
| XGBClassifier | method2: text+audio+label+desc_len | 0.5912 | 0.3242 | 0.4699 | XGB 중 최고 |
| XGBClassifier | method1: text+audio+label | 0.5899 | 0.3238 | 0.4745 | method2와 차이 작음 |
| XGBRegressor | method1 | 0.5661 | 0.2786 | 0.5308 | classification보다 낮음 |
| XGBRegressor | method2 | 0.5625 | 0.2769 | 0.5325 | regression 불리 |

### 8.2 sampling 실험

관련 파일:

| 실험 | 파일 |
|---|---|
| over-sampling XGBoost | `notebooks/over_sample_xg_classfication.ipynb`, `notebooks/over_sample_xg_regression.ipynb` |
| down-sampling XGBoost | `notebooks/down_sample_xg_classfication.ipynb`, `notebooks/down_sample_xg_regression.ipynb` |
| upsampling confidence4 | `up_sampling_confi4/*` |
| binary 1 vs 2345 | `up_sampling_confi4/w_loss_1_2345.ipynb` |
| binary 12 vs 345 | `up_sampling_confi4/w_loss_12_345.ipynb` |
| binary 123 vs 45 | `up_sampling_confi4/w_loss_123_45.ipynb` |

실험 축별 정리:

| 실험 축 | 목적 | 대표 세팅 | 관찰한 결과 | 해석 |
|---|---|---|---|---|
| confidence 4 upsampling | 부족한 confidence 4 class 보강 | confidence 4 sample을 반복/증폭 | minority recall을 올리려 했으나 전체 성능 안정성은 제한적 | 단순 upsampling은 decision boundary를 근본적으로 개선하지 못함 |
| confidence 1 기준 binary | confidence 1만 제거 가능한지 확인 | `1 vs 2345` | test accuracy 0.9699, macro F1 0.5340 | accuracy는 매우 높지만 class imbalance 효과가 큼 |
| confidence 12 기준 binary | confidence 1/2 제거 가능한지 확인 | `12 vs 345` | test accuracy 0.8636, macro F1 0.5763 | 더 어려운 문제지만 low-confidence 제거 목적에는 더 의미 있음 |
| confidence 123 기준 binary | high-confidence 4/5만 선별 | `123 vs 45` | baseline macro F1 0.6256, weighted sweep best macro F1 0.6576 | high-confidence filtering에는 가장 직접적인 split |
| downsampling | majority class 영향 완화 | confidence 4/5 또는 majority class 축소 | notebook output 추가 확인 필요 | 전체 sample을 줄이므로 variance 증가 가능 |
| class-balanced / weighted | minority confidence recall 개선 | confidence 1/2에 loss weight 부여 | recall_negative는 상승하지만 positive recall/accuracy 하락 | filtering 목적에 따라 threshold/weight trade-off 필요 |

binary split별 결과:

| binary 기준 | split | accuracy | macro precision | macro recall | macro F1 | MAE | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| 1 vs 2345 | validation | 0.9671 | 0.5299 | 0.5826 | 0.5416 | 0.0329 | confidence 1이 적어 accuracy가 매우 높음 |
| 1 vs 2345 | test | 0.9699 | 0.5252 | 0.5604 | 0.5340 | 0.0301 | confidence 1 탐지 모델이라기보다 majority 예측 성격이 큼 |
| 12 vs 345 | validation | 0.8722 | 0.5988 | 0.6256 | 0.6096 | 0.1278 | low-confidence 범위를 넓혀 난도 상승 |
| 12 vs 345 | test | 0.8636 | 0.5697 | 0.5861 | 0.5763 | 0.1364 | filtering 기준으로는 1 vs 2345보다 현실적 |

`123 vs 45` loss weight sweep:

| weight 조건 | accuracy | macro recall | macro F1 | recall_negative(123) | recall_positive(45) | 해석 |
|---|---:|---:|---:|---:|---:|---|
| baseline 1/1 | 0.6729 | 0.6232 | 0.6256 | 0.4208 | 0.8256 | high-confidence recall은 높지만 low-confidence를 많이 놓침 |
| mild 5/1.5 | 0.6729 | 0.6370 | 0.6400 | 0.4909 | 0.7832 | low-confidence recall 개선 |
| medium 10/2 | 0.6715 | 0.6488 | 0.6493 | 0.5562 | 0.7414 | 균형이 좋아짐 |
| current 20.54/2.93 | 0.6697 | 0.6635 | 0.6576 | 0.6385 | 0.6886 | macro F1 최고, 양쪽 recall 균형 우수 |
| strong 30/4 | 0.6569 | 0.6647 | 0.6509 | 0.6965 | 0.6330 | low-confidence recall은 높지만 high-confidence recall 하락 |
| very strong 40/6 | 0.6451 | 0.6669 | 0.6431 | 0.7557 | 0.5780 | low-confidence에 과도하게 치우침 |

`12 vs 345` loss weight sweep:

| weight 조건 | accuracy | macro recall | macro F1 | recall_negative(12) | recall_positive(345) | 해석 |
|---|---:|---:|---:|---:|---:|---|
| baseline 1/1 | 0.9229 | 0.5139 | 0.5078 | 0.0292 | 0.9985 | confidence 12를 거의 잡지 못함 |
| mild 5/1.5 | 0.9202 | 0.5472 | 0.5643 | 0.1053 | 0.9891 | low-confidence recall 개선 시작 |
| medium 10/2 | 0.9120 | 0.5936 | 0.6151 | 0.2164 | 0.9708 | 균형 개선 |
| current 20.54/2.93 | 0.8818 | 0.6388 | 0.6260 | 0.3509 | 0.9268 | macro F1 최고 |
| strong 30/4 | 0.8422 | 0.6360 | 0.5953 | 0.3918 | 0.8803 | positive 성능 하락 |
| very strong 40/6 | 0.7851 | 0.6720 | 0.5773 | 0.5380 | 0.8060 | low-confidence recall은 오르지만 전체 품질 하락 |

통찰:

- sampling/weighting은 minority confidence recall을 올리는 데는 효과가 있다.
- 그러나 weight를 강하게 주면 high-confidence recall과 전체 accuracy가 떨어진다.
- BSD35k filtering 목적에서는 단순 accuracy가 아니라 `low-confidence 제거 recall`, `high-confidence precision`, `retained ratio`를 함께 봐야 한다.
- 이 결과가 v3/v4처럼 threshold와 score ranking을 함께 보는 실험으로 이어졌다.

---

## 9. dropout / same-structure MLP 실험

| dropout | accuracy | macro F1 | MAE | 비고 |
|---:|---:|---:|---:|---|
| 0.2 | 0.5680 | 0.3640 | 0.4872 | best experiment |
| 0.3 | 0.5643 | 0.3539 | 0.4881 | 작성 예정 |
| 0.4 | 0.5630 | 0.3518 | 0.4868 | 작성 예정 |
| 0.5 | 0.5589 | 0.3483 | 0.4891 | 작성 예정 |
| 0.6 | 0.5064 | 0.3413 | 0.5725 | 과한 dropout |
| 0.7 | 0.5123 | 0.3487 | 0.5652 | 과한 dropout |
| 0.8 | 0.5333 | 0.3397 | 0.5233 | 과한 dropout |

통찰:

- dropout 0.2가 가장 좋았다.
- dropout을 강하게 주면 accuracy/MAE가 나빠졌다.
- similarity feature를 포함한 단순 MLP도 macro F1 관점에서는 일정한 개선을 보였다.

---

## 10. weighted loss / DEFT weighted loss 실험

관련 파일:

- `deft논문기반_실험/04_weighted_loss_confidence12_sweep_classification_binary.ipynb`
- `up_sampling_confi4/w_loss_123_45.ipynb`
- `up_sampling_confi4/w_loss_12_345.ipynb`
- `up_sampling_confi4/w_loss_1_2345.ipynb`
- `deft논문기반_실험/deft_loss_v4.ipynb`

### 10.1 실험 목적

confidence label은 class imbalance가 심하다. 특히 confidence 1/2는 적고, confidence 4 쪽으로 예측이 쏠리기 쉽다. weighted loss 실험의 목적은 낮은 confidence sample에 더 큰 loss weight를 줘서 low-confidence recall을 올리고, BSD35k filtering에서 noisy sample을 더 잘 제거할 수 있는지 확인하는 것이다.

### 10.2 일반 weighted loss 실험 요약

| 실험 | binary 기준 | 핵심 목적 | 가장 중요한 관찰 |
|---|---|---|---|
| `w_loss_1_2345` | 1 vs 2345 | confidence 1만 탐지 | accuracy는 높지만 macro F1이 낮아 imbalance 영향 큼 |
| `w_loss_12_345` | 12 vs 345 | confidence 1/2 제거 | weight를 키울수록 confidence 12 recall이 오르지만 전체 accuracy 하락 |
| `w_loss_123_45` | 123 vs 45 | high-confidence 4/5 선별 | current weight에서 macro F1과 recall balance가 가장 좋음 |

핵심 해석:

- weighted loss는 minority recall을 올리는 데 효과가 있다.
- 그러나 weight를 크게 줄수록 high-confidence class recall이 떨어져 BSD35k를 많이 가져오는 목적과 충돌한다.
- 따라서 weighted loss는 “최고 F1 모델”이라기보다 “low-confidence 제거 성향을 조절하는 실험”으로 보는 것이 맞다.

### 10.3 DEFT 기반 confidence12 weighted sweep

대상 파일:

- `deft논문기반_실험/04_weighted_loss_confidence12_sweep_classification_binary.ipynb`
- 결과 위치: `deft논문기반_실험/weighted_loss_confidence12_sweep_outputs/`

실험 세팅:

| 항목 | 내용 |
|---|---|
| 입력 feature | embedding + DEFT/CLAP similarity feature |
| task 1 | 5-class confidence classification |
| task 2 | binary confidence 123 vs 45 |
| weight 대상 | confidence 1, confidence 2 |
| sweep | baseline 1/1, mild 5/1.5, medium 10/2, current 20.54/2.93, strong 30/4, very strong 40/6 |

5-class 결과:

| weight 조건 | accuracy | macro F1 | macro recall | MAE | recall_1 | recall_2 | recall_4 | 해석 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline 1/1 | 0.5937 | 0.2969 | 0.2843 | 0.4629 | 0.0000 | 0.0982 | 0.8490 | confidence 1/2를 거의 못 잡음 |
| mild 5/1.5 | 0.5791 | 0.3196 | 0.3026 | 0.4866 | 0.0625 | 0.1786 | 0.8434 | low confidence recall 개선 |
| medium 10/2 | 0.5760 | 0.3247 | 0.3201 | 0.5024 | 0.1250 | 0.2232 | 0.8324 | macro F1 최고 |
| current 20.54/2.93 | 0.5675 | 0.3211 | 0.3272 | 0.5182 | 0.1250 | 0.3036 | 0.8302 | confidence 2 recall 증가 |
| strong 30/4 | 0.5499 | 0.3034 | 0.3203 | 0.5572 | 0.1250 | 0.3304 | 0.8181 | accuracy 하락 |
| very strong 40/6 | 0.5414 | 0.3002 | 0.3336 | 0.5742 | 0.1250 | 0.4554 | 0.8203 | confidence 2 recall은 크지만 전체 품질 낮음 |

binary 123 vs 45 결과:

| weight 조건 | accuracy | macro F1 | macro recall | recall_123 | recall_45 | 해석 |
|---|---:|---:|---:|---:|---:|---|
| baseline 1/1 | 0.7086 | 0.6780 | 0.6735 | 0.5306 | 0.8164 | 45 쪽 recall이 높고 123 탐지가 약함 |
| mild 5/1.5 | 0.7147 | 0.6917 | 0.6892 | 0.5855 | 0.7930 | 가장 안정적인 개선 |
| medium 10/2 | 0.6983 | 0.6783 | 0.6780 | 0.5952 | 0.7607 | 균형은 좋아지지만 accuracy 하락 |
| current 20.54/2.93 | 0.6989 | 0.6844 | 0.6877 | 0.6419 | 0.7334 | 123 recall 개선 |
| strong 30/4 | 0.6940 | 0.6850 | 0.6943 | 0.6952 | 0.6934 | 양쪽 recall 균형 |
| very strong 40/6 | 0.6940 | 0.6806 | 0.6850 | 0.6484 | 0.7217 | 과한 weight의 이득 제한 |

해석:

- DEFT similarity feature를 넣고 confidence 1/2에 weight를 주면 confidence 123 recall이 실제로 개선된다.
- 5-class에서는 medium weight가 macro F1 기준 가장 좋고, binary에서는 mild/current/strong이 목적에 따라 선택 가능하다.
- 하지만 confidence 45 recall과 전체 accuracy를 동시에 유지하는 것은 어렵다. 이 실험은 v4를 완전히 대체하기보다 low-confidence 제거 성향을 보완하는 모델로 해석하는 것이 자연스럽다.

### 10.4 deft_loss_v4 실험

대상 파일:

- `deft논문기반_실험/deft_loss_v4.ipynb`
- 결과 위치: `deft논문기반_실험/deft_loss_v4_outputs/`

목적:

- DEFT/CLAP similarity feature와 mild weighted loss를 사용한 5-class 모델, binary 모델을 만들고, 기존 v4처럼 OOF score stacking으로 BSD35k filtering score를 생성한다.
- 원래 v4가 binary MLP + 5-class expected score의 ensemble이었다면, deft_loss_v4는 similarity feature와 low-confidence weighted loss를 결합한 v4 변형이다.

주요 결과:

| 목적 | 선택 방법 | threshold | precision | recall_45 | F1 | macro F1 | recall_123 | AUC-PR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F1 최적 | OOF ridge stacker | 0.386 | 0.6753 | 0.9320 | 0.7832 | 0.5816 | 0.2609 | 0.8375 |
| macro-F1 최적 | OOF ridge stacker | 0.569 | 0.7686 | 0.7216 | 0.7443 | 0.6776 | 0.6416 | 0.8375 |
| 123/45 recall 균형 | rank average: binary + expected + P45 | 0.454 | 0.7815 | 0.6847 | 0.7299 | 0.6753 | 0.6842 | 0.8366 |
| precision-oriented | rank average: binary + expected + P45 | 0.436 | 0.7745 | 0.7000 | 0.7354 | 0.6752 | 0.6638 | 0.8366 |
| recall-oriented | rank average: binary + expected + P45 | 0.225 | 0.7002 | 0.8812 | 0.7803 | 0.6301 | 0.3775 | 0.8366 |

원래 v4와 비교:

| 실험 | best F1 | precision | recall_45 | AUC-PR | 해석 |
|---|---:|---:|---:|---:|---|
| 원래 v4 | 0.7853 | 0.6859 | 0.9183 | 0.8421 | 전체 ranking과 F1이 약간 더 좋음 |
| deft_loss_v4 | 0.7832 | 0.6753 | 0.9320 | 0.8375 | F1은 거의 비슷하지만 AUC-PR은 약간 낮음 |

해석:

- 최고 F1과 AUC-PR 기준으로는 원래 v4가 아직 더 강하다.
- deft_loss_v4의 장점은 low-confidence 123 recall을 더 명시적으로 조절할 수 있다는 점이다.
- BSD35k를 “많이 가져오기”가 목적이면 원래 v4가 유리하고, “noisy sample을 더 보수적으로 제거하기”가 목적이면 deft_loss_v4 score를 보조 필터로 쓰는 전략이 유리하다.
- 최종적으로는 `v4_only`, `deft_loss_v4_only`, `v4 ∩ deft_loss_v4` 세 subset을 만들어 DCASE baseline downstream 성능으로 비교하는 것이 필요하다.

---

## 11. DEFT / similarity feature 기반 실험

### 11.1 실험 목적
DEFT 논문의 관점에서 audio embedding과 text/class embedding 사이의 similarity가 annotation confidence를 설명할 수 있는지 확인했다. 핵심 질문은 다음과 같다.

- CLAP audio-text similarity가 높을수록 confidence가 높은가?
- sample의 assigned label과 audio embedding이 잘 맞는 정도가 confidence를 설명하는가?
- 단일 similarity feature만으로는 부족하더라도, embedding 기반 confidence model의 보조 feature로는 의미가 있는가?
- DEFT-style score를 BSD35k filtering에 사용할 수 있는가?

### 11.2 사용 파일

| 파일 | 역할 |
|---|---|
| `deft논문기반_실험/01_audio_text_sim_experiment.ipynb` | audio embedding과 text embedding 간 similarity 분석 |
| `deft논문기반_실험/02_audio_class_sim_experiment.ipynb` | audio embedding과 assigned class prompt embedding 간 similarity 분석 |
| `deft논문기반_실험/03_clap_similarity_margin_experiment.ipynb` | assigned label margin, top1-top2 margin 분석 |
| `deft논문기반_실험/04_similarity_feature_mlp_confidence.ipynb` | embedding + similarity feature 기반 MLP confidence 예측 |
| `deft논문기반_실험/deft_loss_v4.ipynb` | v4 score와 DEFT-style objective 비교 |
| `deft논문기반_실험/bsd35k_deft_loss_v4_score_generation.ipynb` | BSD35k에 DEFT-style score 적용 |

### 11.3 Audio-text similarity 단일 feature 분석
분석 대상은 BSD10k metadata 기준 10,956개 sample이다. confidence 분포는 confidence 1이 106개, 2가 749개, 3이 3,280개, 4가 6,045개, 5가 776개였다.

| confidence | n | audio_text_sim mean | std | median |
|---:|---:|---:|---:|---:|
| 1 | 106 | 0.4602 | 0.1525 | 0.4949 |
| 2 | 749 | 0.4694 | 0.1294 | 0.4912 |
| 3 | 3,280 | 0.4707 | 0.1323 | 0.4931 |
| 4 | 6,045 | 0.4912 | 0.1211 | 0.5124 |
| 5 | 776 | 0.5385 | 0.1018 | 0.5599 |

similarity를 5개 quantile bin으로 나누어 confidence 1-5를 직접 예측했을 때의 결과는 낮았다.

| feature | accuracy | macro precision | macro recall | macro F1 | weighted F1 |
|---|---:|---:|---:|---:|---:|
| audio_text_sim quantile 5-bin | 0.2110 | 0.2110 | 0.2398 | 0.1678 | 0.2542 |

하지만 ranking 관점에서는 의미가 있었다.

| 비교 | 낮은 similarity 구간 Q1 | 높은 similarity 구간 Q5 | 변화 |
|---|---:|---:|---:|
| confidence 4/5 비율 | 54.20% | 67.73% | +13.53%p |
| confidence 5 비율 | 2.97% | 11.55% | +8.58%p |
| 평균 confidence | 3.469 | 3.719 | +0.250 |

해석:
- audio-text similarity 하나만으로 confidence 1-5를 분류하기에는 정보량이 부족하다.
- 다만 similarity가 높아질수록 confidence 4/5, 특히 confidence 5의 비율이 증가한다.
- 따라서 similarity는 hard classifier라기보다 high-confidence 후보를 정렬하거나 보조 feature로 사용하는 것이 적절하다.

### 11.4 Audio-assigned class similarity 분석
assigned class prompt embedding과 audio embedding의 similarity도 confidence가 높을수록 평균이 증가했다.

| confidence | n | audio_class_sim mean | std | median |
|---:|---:|---:|---:|---:|
| 1 | 106 | 0.0377 | 0.1317 | 0.0374 |
| 2 | 749 | 0.0661 | 0.1331 | 0.0668 |
| 3 | 3,280 | 0.0744 | 0.1320 | 0.0741 |
| 4 | 6,045 | 0.0780 | 0.1306 | 0.0750 |
| 5 | 776 | 0.1331 | 0.1353 | 0.1142 |

| feature | monotonic | conf5 - conf1 mean | conf5 - conf4 mean | 5-bin accuracy | macro F1 | weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| audio_class_sim | True | 0.0953 | 0.0550 | 0.2107 | 0.1677 | 0.2536 |

해석:
- assigned class와 audio가 잘 맞는 정도는 confidence와 단조 증가 관계를 보인다.
- 하지만 audio-text similarity와 마찬가지로 단독 feature로 confidence를 정확히 분류하기는 어렵다.
- confidence 5와 confidence 1의 평균 차이는 존재하지만, class별 overlap이 커서 threshold 하나로 분리하기 어렵다.

### 11.5 CLAP similarity margin 실험
`03_clap_similarity_margin_experiment.ipynb`에서는 단순 similarity 값이 아니라 margin feature를 확인했다.

| feature | mean monotonic | conf5 - conf1 | conf5 - conf4 | 5-bin accuracy | macro F1 | weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| audio_top1_top2_margin | False | -0.0056 | -0.0010 | 0.1907 | 0.1434 | 0.2380 |
| audio_assigned_label_margin | True | 0.1017 | 0.0387 | 0.2090 | 0.1646 | 0.2535 |
| text_top1_top2_margin | False | 0.0216 | 0.0155 | 0.2019 | 0.1596 | 0.2442 |
| text_assigned_label_margin | True | 0.0616 | 0.0278 | 0.2066 | 0.1612 | 0.2521 |

해석:
- top1-top2 margin은 confidence와 일관된 관계를 보이지 않았다.
- 반대로 assigned label margin은 audio/text 모두에서 monotonic했다.
- 즉, "모델이 가장 헷갈리지 않는가"보다 "주어진 label이 embedding 공간에서 얼마나 지지되는가"가 confidence와 더 관련이 있었다.

### 11.6 Similarity feature + MLP confidence model
`04_similarity_feature_mlp_confidence.ipynb`에서는 embedding과 similarity feature를 함께 사용해 confidence를 예측했다.

| 모델 | task | accuracy | macro F1 | weighted F1 | MAE | best epoch |
|---|---|---:|---:|---:|---:|---:|
| confidence_5class_mlp | 1-5 multi-class | 0.5937 | 0.2969 | 0.5476 | 0.4629 | 4 |
| confidence_binary_123_vs_45_mlp | 1/2/3 vs 4/5 binary | 0.7032 | 0.6702 | 0.6958 | 0.2968 | 3 |

동일 구조 dropout sweep에서는 dropout 0.2가 가장 좋았다.

| dropout | accuracy | macro F1 | MAE | best epoch |
|---:|---:|---:|---:|---:|
| 0.2 | 0.5680 | 0.3640 | 0.4872 | 13 |
| 0.3 | 0.5643 | 0.3539 | 0.4881 | 10 |
| 0.4 | 0.5630 | 0.3518 | 0.4868 | 10 |
| 0.5 | 0.5589 | 0.3483 | 0.4891 | 8 |
| 0.6 | 0.5064 | 0.3413 | 0.5725 | 12 |
| 0.7 | 0.5123 | 0.3487 | 0.5652 | 9 |
| 0.8 | 0.5333 | 0.3397 | 0.5233 | 13 |

해석:
- 5-class confidence를 정확히 맞추는 것은 여전히 어렵다.
- binary confidence, 즉 low/mid confidence와 high confidence를 나누는 문제에서는 similarity feature가 더 안정적으로 작동한다.
- dropout을 강하게 주면 representation이 약해져 성능이 떨어졌고, 0.2 수준의 regularization이 가장 적절했다.

### 11.7 BSD35k DEFT-style score 적용
`bsd35k_deft_loss_v4_score_generation.ipynb`에서는 DEFT-style score를 BSD35k에 적용해 threshold별로 선별되는 sample 수를 확인했다.

| score 방식 | threshold 목적 | threshold | selected rows | selected ratio |
|---|---|---:|---:|---:|
| OOF ridge stacker | F1-optimal | 0.386 | 25,541 | 75.50% |
| OOF ridge stacker | macro-F1-optimal | 0.569 | 10,612 | 31.37% |
| rank average | balanced recall | 0.454 | 18,470 | 54.60% |
| rank average | precision-oriented | 0.436 | 19,147 | 56.60% |
| rank average | recall-oriented | 0.225 | 27,162 | 80.29% |

해석:
- recall-oriented threshold는 너무 많은 BSD35k sample을 통과시켜 augmentation noise를 줄이기 어렵다.
- macro-F1 threshold는 더 보수적이며, BSD35k 추가량을 10k 수준으로 제한한다.
- 다만 DEFT-style score가 downstream baseline 성능 개선으로 바로 이어진다는 증거는 아직 부족하다.
- 현재까지의 결론은 DEFT/similarity feature를 단독 기준으로 쓰기보다 v4 score, classifier agreement, class balance와 함께 사용하는 것이 적절하다는 것이다.

### 11.8 이전 초안 메모

| 파일 | 역할 |
|---|---|
| `deft논문기반_실험/01_audio_text_sim_experiment.ipynb` | audio-text similarity 분석 |
| `deft논문기반_실험/02_audio_class_sim_experiment.ipynb` | audio-class prompt similarity 분석 |
| `deft논문기반_실험/03_clap_similarity_margin_experiment.ipynb` | assigned label margin 분석 |
| `deft논문기반_실험/04_similarity_feature_mlp_confidence.ipynb` | similarity feature + MLP confidence model |
| `deft논문기반_실험/deft_loss_v4.ipynb` | OOF score stacking / threshold objective |
| `deft논문기반_실험/bsd35k_deft_loss_v4_score_generation.ipynb` | BSD35k score generation |

통찰:

- similarity feature 단독으로 confidence를 정확히 예측하기는 어렵다.
- 하지만 embedding + class + similarity feature를 함께 쓰면 confidence model 입력으로 의미가 있다.
- label quality 또는 confidence score의 보조 feature로 활용 가치가 있다.

---

## 12. BSD35k confidence filtering / augmentation 실험

### 12.1 실험 목적
v4 confidence score로 BSD35k를 선별해 DCASE baseline 학습 데이터에 추가했을 때, 실제 downstream 성능이 향상되는지 확인했다. 단순히 confidence classifier 자체의 성능이 좋은지를 보는 것이 아니라, 선별된 외부 데이터가 최종 DCASE 분류 성능에 도움이 되는지를 검증한 실험이다.

### 12.2 Global v4 threshold augmentation

기본 세팅:
- BSD10k train pool: 8,764개
- BSD10k final test: 2,192개
- BSD35k 후보: 31,464개
- v4 score 기준 threshold: `ge4`, `ge3`, `ge2`, `all`
- backbone/loss는 DCASE baseline과 동일하게 유지하고, 추가 데이터만 변경
- 5-fold 반복으로 평균과 표준편차 기록

출처: `baseline_confidnce_train/outputs/v4_35k_baseline_model/fold_metric_summary_mean_std.csv`

| 전략 | BSD35k 추가 수 | accuracy | top accuracy | macro accuracy | hierarchical accuracy | hierarchical F1 |
|---|---:|---:|---:|---:|---:|---:|
| v4_ge4 | 7,633 | 79.279 ± 0.827 | 88.339 ± 0.693 | 73.451 ± 0.651 | 78.803 ± 0.635 | 77.718 ± 0.718 |
| v4_ge3 | 15,566 | 78.741 ± 0.469 | 87.728 ± 0.515 | 72.622 ± 0.629 | 77.992 ± 0.730 | 76.893 ± 0.721 |
| v4_ge2 | 23,851 | 77.381 ± 0.579 | 86.999 ± 0.492 | 71.130 ± 0.781 | 76.922 ± 0.694 | 75.768 ± 0.588 |
| v4_all | 31,464 | 77.354 ± 0.455 | 87.016 ± 0.619 | 70.371 ± 0.658 | 76.239 ± 0.436 | 75.291 ± 0.611 |

해석:
- threshold를 엄격하게 할수록 성능 하락이 줄어든다.
- `v4_ge4`가 BSD35k 추가 전략 중 가장 안정적이다.
- 그러나 BSD10k-only baseline을 넘지는 못했다.
- 따라서 v4 confidence score는 "나쁜 sample을 줄이는 데"는 효과가 있지만, "추가 데이터로 성능을 높이는 데"는 충분하지 않았다.

### 12.3 Pure BSD10k validation 기준 비교
60/20/20 split으로 pure BSD10k validation을 유지한 상태에서 augmentation 전략을 비교했다. 이는 BSD35k가 validation에 섞이지 않도록 하여, 실제 BSD10k 분포에서의 성능을 더 엄격하게 보기 위한 세팅이다.

출처: `baseline_confidnce_train/outputs/v4_35k_602020_pureval/aug_strategy_comparison.csv`

| strategy | added BSD35k | accuracy | top accuracy | macro accuracy | hierarchical accuracy | hierarchical F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_b0 | 0 | 80.326 ± 0.928 | 89.051 ± 0.416 | 74.006 ± 1.728 | 79.318 ± 1.287 | 78.402 ± 1.383 |
| v4_ge4 | 7,633 | 79.279 ± 0.827 | 88.339 ± 0.693 | 73.451 ± 0.651 | 78.803 ± 0.635 | 77.718 ± 0.718 |
| v4_ge3 | 15,566 | 78.741 ± 0.469 | 87.728 ± 0.515 | 72.622 ± 0.629 | 77.992 ± 0.730 | 76.893 ± 0.721 |
| CP1 | 9,335 | 78.148 ± 0.651 | 87.146 ± 0.358 | 71.673 ± 1.184 | 76.956 ± 0.972 | 76.012 ± 1.082 |
| CP | 10,121 | 77.486 ± 0.784 | 86.918 ± 0.767 | 71.518 ± 1.310 | 77.066 ± 1.242 | 75.904 ± 0.999 |
| v4_ge2 | 23,851 | 77.381 ± 0.579 | 86.999 ± 0.492 | 71.130 ± 0.781 | 76.922 ± 0.694 | 75.768 ± 0.588 |
| v4_all | 31,464 | 77.354 ± 0.455 | 87.016 ± 0.619 | 70.371 ± 0.658 | 76.239 ± 0.436 | 75.291 ± 0.611 |

해석:
- pure BSD10k validation에서도 baseline이 가장 높다.
- `v4_ge4`는 가장 보수적인 global threshold로서 성능 하락 폭이 가장 작다.
- BSD35k를 많이 넣을수록 성능이 대체로 하락한다.
- CP/CP1처럼 class-wise 선택을 적용해도 baseline을 넘지 못했다.

### 12.4 Class-selective CP / CP1 실험
CP와 CP1은 class별로 BSD35k 추가가 도움이 되는지 확인한 뒤, 도움이 되는 class만 추가하려는 전략이다.

- CP: class별 best threshold의 validation delta가 0보다 크면 추가, 아니면 drop
- CP1: class별 best threshold의 validation delta가 +1.0 이상일 때만 추가하는 더 보수적인 전략
- CP 추가 수: 10,121개
- CP1 추가 수: 9,335개

출처: `baseline_confidnce_train/outputs/v4_35k_class_selective/aug_strategy_comparison.csv`

| strategy | added BSD35k | accuracy | top accuracy | macro accuracy | hierarchical accuracy | hierarchical F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_b0 | 0 | 80.383 ± 0.340 | 89.051 ± 0.276 | 73.944 ± 0.488 | 79.231 ± 0.412 | 78.466 ± 0.436 |
| v4_ge4 | 7,633 | 79.279 ± 0.827 | 88.339 ± 0.693 | 73.451 ± 0.651 | 78.803 ± 0.635 | 77.718 ± 0.718 |
| v4_ge3 | 15,566 | 78.741 ± 0.469 | 87.728 ± 0.515 | 72.622 ± 0.629 | 77.992 ± 0.730 | 76.893 ± 0.721 |
| CP1 | 9,335 | 78.403 ± 0.695 | 87.427 ± 0.602 | 72.280 ± 1.116 | 77.531 ± 0.870 | 76.457 ± 0.782 |
| CP | 10,121 | 78.139 ± 0.615 | 87.190 ± 0.727 | 71.909 ± 1.116 | 77.278 ± 0.992 | 76.149 ± 1.202 |

해석:
- class-selective로 sample 수를 줄여도 baseline에는 도달하지 못했다.
- CP1은 CP보다 보수적이라 약간 더 안정적이지만, 여전히 `v4_ge4`보다 낮다.
- class별 validation delta만 보고 추가 여부를 결정하는 방식은 전체 decision boundary 변화를 충분히 설명하지 못했다.

### 12.5 Class별 threshold 분석
출처: `baseline_confidnce_train/outputs/v4_35k_class_selective/per_class_threshold_analysis.csv`

BSD35k 추가로 크게 나빠진 class:

| class | baseline recall | best threshold | best delta | CP/CP1 판단 | BSD35k 후보 수 |
|---|---:|---|---:|---|---:|
| fx-n | 92.27 | ge4 | -10.91 | DROP | 659 |
| fx-m | 77.86 | ge4 | -5.00 | DROP | 1,542 |
| m-sp | 88.32 | ge4 | -2.92 | DROP | 1,645 |
| sp-p | 81.11 | ge4 | -2.50 | DROP | 1,697 |
| m-si | 81.69 | ge4 | -1.27 | DROP | 1,344 |
| fx-a | 43.53 | ge4 | -1.18 | DROP | 499 |

BSD35k 추가로 개선 가능성이 있었던 class:

| class | baseline recall | best threshold | best delta | CP 판단 | BSD35k 후보 수 |
|---|---:|---|---:|---|---:|
| ss-n | 70.51 | ge2 | +7.44 | ADD | 4,831 |
| fx-v | 60.00 | ge3 | +7.32 | ADD | 1,852 |
| m-m | 57.19 | ge4 | +6.25 | ADD | 2,044 |
| fx-el | 70.48 | ge4 | +5.71 | ADD | 2,000 |
| ss-i | 36.59 | ge2 | +3.90 | ADD | 383 |
| is-e | 68.39 | ge3 | +3.55 | ADD | 757 |
| ss-u | 67.13 | ge2 | +2.66 | ADD | 2,063 |

해석:
- class에 따라 BSD35k augmentation의 효과가 완전히 다르다.
- `ss-n`, `fx-v`, `m-m`, `fx-el`처럼 추가 데이터가 recall을 올릴 가능성이 있는 class가 있었다.
- 반대로 `fx-n`, `fx-m`, `m-sp`, `sp-p`, `m-si`는 confidence threshold를 높여도 개선되지 않았다.
- 즉, BSD35k는 전체적으로 나쁘다기보다 class별로 도움이 되는 영역과 해로운 영역이 섞여 있다.

### 12.6 CP/CP1의 실제 per-class recall 변화
출처: `baseline_confidnce_train/outputs/v4_35k_class_selective/per_class_recall_change.csv`

CP/CP1에서 좋아진 class 예시:

| class | baseline recall | CP recall | CP1 recall | CP - baseline | CP1 - baseline |
|---|---:|---:|---:|---:|---:|
| ss-n | 69.49 | 82.31 | 81.28 | +12.82 | +11.79 |
| ss-s | 82.79 | 87.44 | 86.51 | +4.65 | +3.72 |
| ss-i | 31.71 | 35.61 | 40.49 | +3.90 | +8.78 |
| sp-s | 90.93 | 93.79 | 91.68 | +2.86 | +0.75 |

CP/CP1에서도 나빠진 class 예시:

| class | baseline recall | CP recall | CP1 recall | CP - baseline | CP1 - baseline |
|---|---:|---:|---:|---:|---:|
| fx-n | 90.91 | 63.18 | 66.21 | -27.73 | -24.70 |
| fx-a | 39.41 | 22.35 | 22.35 | -17.06 | -17.06 |
| m-si | 81.69 | 71.55 | 76.34 | -10.14 | -5.35 |
| fx-m | 79.64 | 70.36 | 68.93 | -9.28 | -10.71 |
| sp-c | 46.29 | 41.71 | 39.43 | -4.58 | -6.86 |

최종 해석:
- CP/CP1은 일부 class recall을 실제로 개선했다.
- 그러나 다른 class의 recall 하락이 더 컸고, 전체 평균 성능은 baseline보다 낮아졌다.
- 특히 DROP으로 판단한 class도 간접적으로 영향을 받았다. 이는 한 class에 데이터를 추가해도 모델의 shared representation과 decision boundary가 전체 class에 영향을 주기 때문이다.
- 다음 단계는 hard filtering보다 sample weight, class-specific augmentation ratio, top-class balance, label-quality intersection을 함께 고려해야 한다.

### 12.7 이전 초안 메모: v4 threshold로 BSD35k 추가

출처: `baseline_confidnce_train/outputs/v4_35k_baseline_model/ranked_compact_summary.csv`

| dataset | added BSD35k | accuracy | H-Acc | H-F1 | macro acc | 코멘트 |
|---|---:|---:|---:|---:|---:|---|
| v4_ge4 | 7,633 | 79.279 ± 0.827 | 78.803 ± 0.635 | 77.718 ± 0.718 | 73.451 ± 0.651 | v4 중 가장 좋음 |
| v4_ge3 | 15,566 | 78.741 ± 0.469 | 77.992 ± 0.730 | 76.893 ± 0.721 | 72.622 ± 0.629 | 더 많이 넣으면 하락 |
| v4_ge2 | 23,851 | 77.381 ± 0.579 | 76.922 ± 0.694 | 75.768 ± 0.588 | 71.130 ± 0.781 | noise 증가 |
| v4_all | 31,464 | 77.354 ± 0.455 | 76.239 ± 0.436 | 75.291 ± 0.611 | 70.371 ± 0.658 | 전체 투입은 가장 낮음 |

### 12.8 이전 초안 메모: 60/20/20 pure BSD10k validation 실험

출처: `baseline_confidnce_train/outputs/v4_35k_602020_pureval/aug_strategy_comparison.csv`

| strategy | added BSD35k | accuracy | H-Acc | H-F1 | macro acc | top acc | 결과 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_b0 | 0 | 80.326 ± 0.928 | 79.318 ± 1.287 | 78.402 ± 1.383 | 74.006 ± 1.728 | 89.051 ± 0.416 | 기준선 |
| v4_ge4 | - | 79.279 ± 0.827 | 78.803 ± 0.635 | 77.718 ± 0.718 | 73.451 ± 0.651 | 88.339 ± 0.693 | baseline 미달 |
| v4_ge3 | - | 78.741 ± 0.469 | 77.992 ± 0.730 | 76.893 ± 0.721 | 72.622 ± 0.629 | 87.728 ± 0.515 | baseline 미달 |
| CP | 10,121 | 77.486 ± 0.784 | 77.066 ± 1.242 | 75.904 ± 0.999 | 71.518 ± 1.310 | 86.918 ± 0.767 | 실패 |
| CP1 | 9,335 | 78.148 ± 0.651 | 76.956 ± 0.972 | 76.012 ± 1.082 | 71.673 ± 1.184 | 87.146 ± 0.358 | 실패 |
| v4_ge2 | - | 77.381 ± 0.579 | 76.922 ± 0.694 | 75.768 ± 0.588 | 71.130 ± 0.781 | 86.999 ± 0.492 | baseline 미달 |
| v4_all | - | 77.354 ± 0.455 | 76.239 ± 0.436 | 75.291 ± 0.611 | 70.371 ± 0.658 | 87.016 ± 0.618 | baseline 미달 |

통찰:

- pure BSD10k val/test로 공정하게 비교해도 BSD35k augmentation은 baseline을 넘지 못했다.
- CP/CP1처럼 class-selective augmentation을 해도 boundary stealing 문제가 발생했다.
- 특히 ss-n을 늘리면 인접한 fx-n이 ss-n으로 흡수되는 현상이 관찰되었다.
- 결론: 단순 hard filtering + equal weight row addition은 충분하지 않다.

---

## 13. BSD35k label quality / classifier scoring

### 13.1 실험 목적
confidence model은 BSD10k의 human confidence를 예측하는 모델이다. 하지만 BSD35k를 augmentation에 사용하려면 또 다른 질문이 필요하다.

- BSD35k의 provided label이 BSD10k 기준 classifier와 얼마나 일치하는가?
- BSD35k label 자체가 깨끗한가, 아니면 pseudo-label 또는 noisy label로 봐야 하는가?
- confidence score와 label quality score는 같은 의미인가, 다른 의미인가?

이 실험에서는 새 confidence classifier가 아니라, 이미 학습된 DCASE baseline classifier를 사용해 BSD35k label agreement를 측정했다.

### 13.2 실험 세팅
출처: `docs/bsd35k_classifier_scoring_experiment.md`, `notebooks/01_bsd35k_classifier_scoring_analysis.ipynb`

| 항목 | 내용 |
|---|---|
| scoring model | BSD10k로 학습된 DCASE baseline classifier |
| checkpoint | `dcase2026_task1_baseline/model_output/both/fold_0~4/best_model.pth` |
| input | BSD35k audio/text CLAP embedding |
| metadata | `data/metadata/BSD35k-CS_metadata.csv` |
| scored rows | 31,464 / 31,464 |
| 주요 출력 | predicted class, predicted top class, provided class probability, classifier confidence, margin, fold disagreement |

scoring에 사용한 baseline classifier 자체의 BSD10k 성능은 다음과 같았다.

| metric | score |
|---|---:|
| accuracy | 79.63 ± 0.39 |
| top accuracy | 88.88 ± 0.15 |
| hierarchical F1 | 78.33 ± 0.41 |

### 13.3 BSD35k 전체 label agreement 결과
출처: `experiments/bsd35k_scoring/classifier_scoring_summary.json`

| metric | value |
|---|---:|
| rows scored | 31,464 |
| same class rate | 57.02% |
| same top class rate | 75.14% |
| mean classifier confidence | 0.764 |
| mean provided class probability | 0.506 |

해석:
- BSD35k provided class와 BSD10k classifier의 fine class prediction이 정확히 일치한 비율은 57.02%다.
- top class 수준에서는 75.14%가 일치했다.
- 즉, 많은 sample이 fine class 수준에서는 흔들리지만, 상위 계층에서는 어느 정도 맞는 경우가 존재한다.
- BSD35k를 그대로 추가하면 fine class boundary를 흔들 가능성이 크다.

### 13.4 Class별 label agreement
낮은 agreement class:

| class | n | same class | same top | provided class prob |
|---|---:|---:|---:|---:|
| fx-m | 1,542 | 24.25% | 93.32% | 0.218 |
| is-p | 2,321 | 24.47% | 31.28% | 0.224 |
| ss-i | 383 | 24.80% | 44.65% | 0.211 |
| is-k | 43 | 27.91% | 53.49% | 0.271 |
| sp-c | 50 | 32.00% | 54.00% | 0.272 |
| fx-ex | 839 | 33.61% | 60.19% | 0.280 |
| fx-el | 2,000 | 34.85% | 77.40% | 0.319 |

높은 agreement class:

| class | n | same class | same top | provided class prob |
|---|---:|---:|---:|---:|
| fx-v | 1,852 | 83.26% | 94.87% | 0.677 |
| m-m | 2,044 | 78.33% | 91.88% | 0.656 |
| fx-o | 4,534 | 77.68% | 95.37% | 0.722 |
| sp-s | 980 | 69.90% | 77.24% | 0.649 |
| fx-n | 659 | 68.44% | 93.02% | 0.604 |
| ss-n | 4,831 | 68.04% | 73.13% | 0.601 |

해석:
- `fx-m`은 same top class가 93.32%로 높지만 fine class agreement는 24.25%로 낮다. 이는 top-level은 맞지만 fine label이 흔들리는 전형적인 hierarchical ambiguity다.
- `is-p`, `ss-i`는 same top class도 낮아 더 위험한 label mismatch 가능성이 있다.
- `fx-v`, `m-m`, `fx-o`는 classifier 관점에서는 비교적 깨끗한 BSD35k class다.
- 다만 `fx-n`처럼 classifier agreement가 높아도 downstream augmentation에서는 recall이 크게 하락한 class가 있었다. 따라서 label agreement만으로 downstream 유용성을 보장할 수는 없다.

### 13.5 Label quality score 설계
출처: `experiments/bsd35k_label_quality/label_quality_summary.json`

label quality score는 BSD35k provided label이 BSD10k classifier 기준으로 얼마나 신뢰 가능한지를 나타내는 보조 score다.

| 구성 요소 | weight |
|---|---:|
| provided class probability | 0.45 |
| same class | 0.20 |
| same top class | 0.12 |
| classifier margin | 0.12 |
| low ensemble disagreement | 0.08 |
| annotation score auxiliary | 0.03 |

전체 결과:

| 항목 | 값 |
|---|---:|
| rows | 31,464 |
| mean label quality score | 0.578 |
| mean label noise score | 0.422 |

quality group 분포:

| group | count |
|---|---:|
| clean_high_quality | 13,080 |
| middle_uncertain | 10,871 |
| pseudo_label_candidate | 4,496 |
| ambiguous_same_top | 1,754 |
| noisy_top_mismatch | 1,263 |

해석:
- BSD35k 전체 중 clean_high_quality로 볼 수 있는 sample은 약 13k개다.
- middle_uncertain과 pseudo_label_candidate도 상당히 많다.
- noisy_top_mismatch는 1,263개로 수는 작지만, top-level label 자체가 달라질 수 있어 downstream에 치명적일 수 있다.

### 13.6 Label quality threshold sweep
출처: `experiments/bsd35k_label_quality/BSD35k_label_quality_threshold_sweep.csv`

| threshold | kept rows | keep rate | classes | min class count | same class | same top | mean provided prob |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 30,107 | 95.69% | 23 | 27 | 59.59% | 78.53% | 0.528 |
| 0.20 | 26,854 | 85.35% | 23 | 21 | 66.81% | 86.50% | 0.583 |
| 0.25 | 22,835 | 72.58% | 23 | 14 | 78.56% | 98.34% | 0.669 |
| 0.30 | 21,394 | 68.00% | 23 | 14 | 83.86% | 100.00% | 0.701 |
| 0.35 | 18,519 | 58.86% | 23 | 14 | 96.87% | 100.00% | 0.795 |
| 0.40 | 17,955 | 57.07% | 23 | 12 | 99.92% | 100.00% | 0.809 |
| 0.50 | 17,799 | 56.57% | 23 | 12 | 100.00% | 100.00% | 0.814 |
| 0.65 | 15,647 | 49.73% | 23 | 8 | 100.00% | 100.00% | 0.862 |
| 0.85 | 10,627 | 33.78% | 23 | 1 | 100.00% | 100.00% | 0.937 |
| 0.90 | 8,472 | 26.93% | 21 | 2 | 100.00% | 100.00% | 0.957 |
| 0.95 | 5,474 | 17.40% | 18 | - | 100.00% | 100.00% | - |

해석:
- threshold 0.30 이상부터 same top class가 100%가 된다.
- threshold 0.40 이상에서는 same class도 거의 100%가 된다.
- 그러나 threshold를 너무 높이면 class coverage와 class balance가 무너진다.
- label quality score는 clean sample 선별에는 강하지만, minority class 보존과 downstream 성능 개선까지 자동으로 보장하지 않는다.

### 13.7 Confidence score와 label quality score의 차이

| 구분 | confidence model score | label quality score |
|---|---|---|
| 학습 기준 | BSD10k human confidence | BSD10k classifier와 BSD35k label agreement |
| 의미 | 사람이 label을 얼마나 확신했는가 | provided label이 classifier 기준으로 얼마나 타당한가 |
| 강점 | annotation uncertainty 반영 | BSD35k label noise 탐지에 직접적 |
| 약점 | downstream usefulness와 직접 일치하지 않음 | classifier bias를 그대로 따를 수 있음 |

최종 해석:
- BSD35k filtering은 confidence score 하나만으로는 부족하다.
- label quality score는 provided label의 신뢰도를 보는 별도 축으로 유용하다.
- 하지만 classifier agreement가 높은 class도 downstream에서 해로울 수 있었기 때문에, 최종 augmentation에는 confidence, label quality, class balance, class별 recall delta를 함께 봐야 한다.

### 13.8 이전 초안 메모

| 파일 | 역할 |
|---|---|
| `notebooks/01_bsd35k_classifier_scoring_analysis.ipynb` | BSD10k classifier로 BSD35k label agreement 분석 |
| `docs/bsd35k_classifier_scoring_experiment.md` | scoring 실험 보고서 |
| `experiments/bsd35k_scoring/*` | classifier score 산출물 |
| `experiments/bsd35k_label_quality/*` | label quality score 산출물 |

정리할 내용:

- BSD35k provided label과 BSD10k classifier prediction 일치율
- same class / same top class rate
- class별/uploader별 label quality
- confidence filtering과 label quality score의 차이

---

## 14. HATR confidence1 filtering 실험

### 14.1 실험 목적
이 실험은 BSD10k metadata에서 `confidence == 1`인 sample이 학습에 도움이 되는지, 아니면 noisy label로 작용해 성능을 떨어뜨리는지 확인하기 위한 실험이다.

비교한 전략은 총 3가지다.

| 전략 | 설명 |
|---|---|
| all confidence | confidence1을 포함한 전체 BSD10k 데이터 사용 |
| no confidence1 | confidence1 데이터를 전부 제거 |
| select confidence1 classes | confidence1 중 특정 class만 선택적으로 유지 |

핵심 질문은 단순히 "confidence1은 나쁜가?"가 아니라, "confidence1의 유용성이 class마다 다른가?"이다.

### 14.2 전체 성능 비교

| 전략 | Accuracy | Top Acc | Macro Acc | Hier Acc | Hier F1 |
|---|---:|---:|---:|---:|---:|
| all confidence | 79.82 ± 0.63 | - | - | - | - |
| no confidence1 | 80.49 ± 0.75 | 89.19 ± 0.54 | 74.64 ± 1.40 | 79.82 ± 0.99 | 77.50 ± 0.82 |
| select confidence1 classes | 80.39 ± 1.41 | 89.61 ± 0.83 | 74.82 ± 1.03 | 80.18 ± 0.56 | 77.63 ± 1.06 |

해석:
- 단순 accuracy 기준으로는 `no confidence1`이 가장 높다.
- 따라서 confidence1 전체에는 noisy sample이 포함되어 있고, 전체를 그대로 사용하는 것은 최선이 아니다.
- 하지만 top accuracy, hierarchical accuracy, hierarchical F1은 `select confidence1 classes`가 더 좋다.
- 즉, confidence1은 전부 버릴 데이터가 아니라 class별로 선별해야 하는 데이터다.

### 14.3 핵심 결과 요약

단순 accuracy 기준:

| 전략 | Accuracy |
|---|---:|
| no confidence1 | 80.49 ± 0.75 |
| select confidence1 classes | 80.39 ± 1.41 |

hierarchical 계열 지표 기준:

| 지표 | no confidence1 | select confidence1 classes |
|---|---:|---:|
| Top Acc | 89.19 ± 0.54 | 89.61 ± 0.83 |
| Hier Acc | 79.82 ± 0.99 | 80.18 ± 0.56 |
| Hier F1 | 77.50 ± 0.82 | 77.63 ± 1.06 |

정리하면 다음과 같다.

```text
confidence1은 전체적으로 noisy한 경향이 있지만,
모든 confidence1 sample이 해로운 것은 아니다.
일부 class에서는 confidence1 sample이 계층적 분류 성능 유지에 도움을 준다.
```

### 14.4 Class별 성능 변화
분석한 confidence1 후보 class는 다음 5개다.

```text
sp-c
sp-p
fx-a
fx-v
fx-m
```

| Class | all Recall | all F1 | no_conf1 Recall | no_conf1 F1 | select Recall | select F1 | select F1 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sp-c | 51.38 | 49.10 | 47.46 | 51.38 | 47.54 | 51.03 | +1.93 |
| sp-p | 79.65 | 75.64 | 77.20 | 76.36 | 77.16 | 76.16 | +0.53 |
| fx-a | 57.29 | 60.18 | 50.60 | 56.76 | 50.61 | 55.28 | -4.91 |
| fx-v | 64.16 | 62.26 | 62.19 | 61.73 | 62.23 | 61.91 | -0.35 |
| fx-m | 79.93 | 74.64 | 78.49 | 72.64 | 78.53 | 72.61 | -2.03 |

F1 기준으로 개선된 class는 `sp-c`, `sp-p` 두 개다.

다만 `sp-c`는 F1은 개선되었지만 confusion이 넓게 퍼져 있어 안정적인 개선이라고 보기 어렵다. 반대로 `sp-p`는 F1 개선 폭은 작지만 confusion 구조가 비교적 안정적이라 confidence1 유지 가치가 가장 명확한 class로 볼 수 있다.

### 14.5 Confusion matrix 기반 분석
`no confidence1` 전략의 confusion matrix를 보면 일부 class에서 강한 오분류가 발생했다.

| True Class | Predicted Class | 오분류 비율 |
|---|---|---:|
| fx-a | ss-n | 36.2% |
| fx-v | ss-u | 21.3% |
| sp-c | ss-u | 19.2% |

`sp-c`의 경우 자기 class recall이 낮고 여러 주변 class로 분산된다.

| True Class | Predicted Class | 비율 |
|---|---|---:|
| sp-c | sp-c | 47.46% |
| sp-c | ss-u | 19.24% |
| sp-c | fx-h | 12.44% |
| sp-c | ss-i | 12.38% |

따라서 `sp-c`는 F1이 개선되었더라도, confusion 구조상 추가 검증이 필요하다.

반면 `sp-p`는 비교적 안정적이다.

| True Class | Predicted Class | 비율 |
|---|---|---:|
| sp-p | sp-p | 77.20% |
| sp-p | sp-s | 9.92% |

`sp-p`의 주요 오분류는 같은 `sp-*` 계열인 `sp-s`로 향한다. 따라서 fine class에서는 틀리더라도 hierarchical 관점에서는 상대적으로 덜 치명적이다.

### 14.6 Class별 해석

#### sp-p
`sp-p`는 confidence1을 선택적으로 유지했을 때 F1이 개선되었고, confusion도 같은 top-class 계열 안에서 비교적 안정적으로 발생했다.

```text
sp-p: confidence1 유지 가치가 가장 명확한 class
```

#### sp-c
`sp-c`는 F1이 개선되었지만 recall이 낮고 `ss-u`, `fx-h`, `ss-i`로 많이 흩어진다.

```text
sp-c: 유지 후보지만 추가 검증 필요
```

#### fx-a
`fx-a`는 confidence1 제거와 선택 조건 모두에서 F1이 하락했다. 특히 `fx-a -> ss-n` 오분류가 36.2%로 매우 크다.

```text
fx-a: 현재 결과만으로는 confidence1 유지 효과가 불명확하며 추가 ablation 필요
```

#### fx-v
`fx-v`도 F1이 소폭 하락했고, `fx-v -> ss-u` 오분류가 21.3%로 나타났다.

```text
fx-v: confidence1 제거로 인한 degradation 가능성이 있어 추가 ablation 필요
```

#### fx-m
`fx-m`은 F1이 하락했다. 주요 이동은 `fx-o`, `fx-v` 방향으로 발생했다.

```text
fx-m: 현재 결과에서는 confidence1 유지 효과가 약함
```

### 14.7 최종 결론
이 실험의 결론은 다음과 같다.

1. confidence1 데이터를 모두 사용하는 것보다 confidence1을 제거하는 것이 전체 accuracy를 개선했다.
2. 따라서 confidence1 전체에는 noisy sample이 포함되어 있다.
3. 하지만 confidence1을 전부 제거하는 것보다 일부 class의 confidence1을 선택적으로 유지했을 때 top accuracy, hierarchical accuracy, hierarchical F1은 더 좋아졌다.
4. confidence1의 유용성은 class-dependent하다.
5. 단순 accuracy를 최우선하면 `no confidence1`이 적합하다.
6. DCASE Task 1에서 계층적 성능을 중요하게 보면 `select confidence1 classes`가 더 적합하다.

### 14.8 최종 추천

| 목적 | 추천 전략 |
|---|---|
| 단순 Accuracy 최우선 | no confidence1 |
| Top-class / Hierarchical 성능 최우선 | select confidence1 classes |
| confidence1 유지 가치가 가장 명확한 class | sp-p |
| F1 개선 기준 추가 후보 | sp-c |
| confusion degradation 때문에 추가 검증할 class | fx-a, fx-v |
| 현재 유지 효과가 약한 class | fx-m |

보고서용 요약:

```text
Removing all confidence-1 samples improved overall accuracy compared with using all samples.
However, selectively retaining confidence-1 samples for specific classes improved top-class
accuracy, hierarchical accuracy, and hierarchical F1. This suggests that confidence-1 samples
are not uniformly harmful; their usefulness is class-dependent. In particular, sp-p showed
the clearest benefit from retaining confidence-1 samples, while sp-c also improved in F1
but still showed substantial confusion with neighboring classes.
```

---

## 15. 최종 결론 초안

### 15.1 Baseline에 대한 결론
BSD10k-only DCASE baseline은 현재 가장 안정적인 기준점이다. BSD10k만 사용했을 때 약 79-80% accuracy, 약 78-79% hierarchical accuracy/F1 수준을 보였고, 60/20/20 pure validation 세팅에서도 baseline이 가장 높은 성능을 유지했다.

HATR TopClassLoss와 HATR ContrastiveLoss를 추가한 BSD10k-only 실험에서는 accuracy와 hierarchical accuracy가 소폭 상승했지만, hierarchical F1은 기본 CE baseline보다 낮아질 수 있었다. 즉, hierarchical loss는 일부 metric을 개선하지만 전체적으로 일관된 우위는 아니었다.

### 15.2 Confidence model에 대한 결론
초기 1-layer classification/regression, 일반 MLP, XGBoost/RF 실험은 confidence 예측의 어려움을 보여줬다. 특히 5-class confidence prediction은 confidence 4 collapse 또는 평균 회귀 문제가 반복적으로 나타났다.

가장 중요한 모델 흐름은 다음과 같다.

- v2: 3-layer MLP 기반 expected confidence score
- v3: 3-layer MLP 기반 binary confidence score
- v4: v2와 v3를 결합한 ensemble score

v4는 confidence filtering score 자체로는 가장 안정적인 후보였다. clean label v2/v3/v4 실험에서는 clean label만 사용하는 것이 항상 더 좋은 결과를 만들지는 않는다는 점도 확인했다. 이는 confidence model이 단순히 label noise만 보는 것이 아니라, class distribution과 representation difficulty까지 함께 반영하고 있음을 시사한다.

### 15.3 Similarity / DEFT 실험에 대한 결론
CLAP audio-text similarity와 audio-class similarity는 confidence와 약한 양의 관계를 보였다. similarity가 높은 구간에서 confidence 4/5와 confidence 5 비율이 증가했지만, 단일 similarity feature만으로 confidence 1-5를 정확히 분류하기에는 부족했다.

특히 assigned label margin은 top1-top2 margin보다 confidence와 더 일관된 관계를 보였다. 이는 confidence를 설명할 때 "가장 가까운 두 class의 차이"보다 "주어진 label이 embedding 공간에서 얼마나 지지되는가"가 더 중요하다는 것을 보여준다.

DEFT-style score는 BSD35k sample을 다양한 threshold로 선별하는 보조 기준이 될 수 있다. 그러나 현재 결과만으로는 DEFT/similarity score가 downstream DCASE baseline을 직접 개선한다고 결론내리기 어렵다.

### 15.4 BSD35k augmentation에 대한 결론
BSD35k를 confidence score로 선별해 추가해도 BSD10k-only baseline을 넘지 못했다.

핵심 결과는 다음과 같다.

- `v4_ge4`: 7,633개 추가, accuracy 79.279, hierarchical F1 77.718
- `v4_ge3`: 15,566개 추가, accuracy 78.741, hierarchical F1 76.893
- `v4_ge2`: 23,851개 추가, accuracy 77.381, hierarchical F1 75.768
- `v4_all`: 31,464개 추가, accuracy 77.354, hierarchical F1 75.291

threshold를 높일수록 성능 하락은 줄어들었지만, baseline보다 좋아지지는 않았다. class-selective CP/CP1도 일부 class recall은 개선했지만, 전체 성능은 baseline을 넘지 못했다.

이 결과는 BSD35k가 전부 나쁘다는 뜻은 아니다. `ss-n`, `fx-v`, `m-m`, `fx-el`처럼 도움이 되는 class가 있었고, `fx-n`, `fx-m`, `m-sp`, `sp-p`, `m-si`처럼 해로운 class도 있었다. 문제는 sample을 추가하는 순간 shared representation과 decision boundary가 전체 class에 영향을 준다는 점이다.

### 15.5 Label quality 분석에 대한 결론
BSD35k classifier scoring 결과, provided fine class와 BSD10k classifier prediction이 일치한 비율은 57.02%, top class 일치율은 75.14%였다. 이는 BSD35k가 fine class 기준으로 상당한 label ambiguity를 포함한다는 뜻이다.

label quality score를 만들면 clean_high_quality sample 약 13,080개를 분리할 수 있었고, threshold 0.40 이상에서는 same class/top class가 거의 완전히 일치하는 sample만 남길 수 있었다. 그러나 threshold를 높이면 class coverage와 minority class 수가 빠르게 줄어든다.

따라서 label quality score는 confidence score와 다른 축으로 사용해야 한다.

- confidence score: human annotation confidence를 반영
- label quality score: BSD35k provided label과 BSD10k classifier agreement를 반영
- downstream usefulness: class balance, boundary shift, sample weight까지 포함한 별도 문제

### 15.6 다음 실험 방향
현재 결과를 기준으로 다음 단계는 hard filtering만 반복하는 것이 아니라, filtering 이후 학습 방식 자체를 바꾸는 방향이 더 타당하다.

우선순위:

1. BSD35k sample을 CE와 동일 weight로 넣지 않고 confidence/label-quality 기반 sample weight 적용
2. class별 augmentation ratio 제한
3. v4 score와 label quality score의 intersection 사용
4. top-class balance를 유지하는 sampling
5. 도움이 되는 class와 해로운 class를 분리한 class-specific weight 또는 curriculum 적용
6. BSD35k를 정답 label로만 쓰지 않고 pseudo-label, same-top auxiliary target, contrastive regularization으로 활용

최종 메시지:
BSD35k 추가 실험의 핵심 발견은 "좋은 confidence classifier를 만들면 augmentation이 자동으로 좋아진다"가 아니라는 점이다. confidence filtering은 필요한 조건이지만 충분조건은 아니었다. 실제 성능을 높이려면 sample quality, class balance, decision boundary, loss weighting을 함께 설계해야 한다.

### 15.7 이전 초안 메모

- BSD10k-only baseline은 안정적이다.
- BSD10k-only에서 HATR TopClassLoss/ContrastiveLoss를 추가하면 accuracy와 hierarchical accuracy는 소폭 상승하지만, hierarchical F1은 기본 CE baseline보다 낮아질 수 있다.
- BSD35k를 그대로 train/validation에 넣으면 BSD10k test 성능이 크게 하락한다.
- 일반 MLP와 1-layer classification/regression은 confidence 4 collapse 또는 평균 회귀 문제를 보였다.
- v2/v3는 3-layer MLP 기반의 핵심 confidence 모델이며, v4는 v2 expected score와 v3 binary score를 결합한 ensemble이다.
- v4는 confidence filtering score 자체로는 가장 안정적이다.
- clean label v2/v3/v4 실험은 clean input이 항상 더 좋은 것이 아니라는 중요한 통찰을 줬다.
- v4 score로 BSD35k를 선별해도 downstream DCASE baseline은 자동으로 좋아지지 않았다.
- 실패 원인은 단순 noise 제거 부족뿐 아니라 class boundary stealing, class prior 변화, equal-weight row addition의 한계다.
- 다음 단계는 hard filtering이 아니라 confidence 기반 sample weighting, class-balanced augmentation, boundary-aware filtering으로 가야 한다.

---

## 16. 다음에 채울 표 목록

| 표 | 출처 | 상태 |
|---|---|---|
| BSD10k CE baseline fold별 결과 | `DCASE2026_Task1_VSCode.ipynb` | 작성 필요 |
| BSD10k HATR loss 수정 fold별 결과 | 사용자 제공 결과 | 작성 완료 초안 |
| BSD35kTrain/BSD10kTest fold별 결과 | `DCASE2026_Task1_VSCode_BSD35kTrain_BSD10kTest.ipynb` | 일부 작성 |
| 일반 MLP confidence 실험 결과 | `kang_study*`, `confidence_mlp_tsne_mapping.ipynb` | 작성 필요 |
| 1-layer classification/regression 결과 | `35k confidence 모델 선택.md`, `up_sampling_confi4/*` | 일부 작성 |
| v2/v3/v4 confidence model 결과 | 각 report md/json/csv | 일부 작성 |
| clean label v2/v3/v4 결과 | `clean_v2_v3_v4_report_ko.md` | 작성 완료 초안 |
| v4_35k_baseline 결과 | `notebooks/v4_35k_baseline모델.ipynb`, 관련 outputs | 작성 필요 |
| confidence model별 downstream 결과 | `baseline_confidnce_train/outputs/*/summary_results.csv` | 작성 필요 |
| v4 BSD35k augmentation 결과 | `v4_35k_baseline_model/ranked_compact_summary.csv` | 작성 완료 초안 |
| pure-val class-selective augmentation 결과 | `v4_35k_602020_pureval/aug_strategy_comparison.csv` | 작성 완료 초안 |
| 실패 원인 confusion matrix 분석 | `per_class_recall_change.csv`, `combined_cm_*.csv` | 작성 필요 |
