# XGBoost confidence 모델 분석

분석 대상:

- `notebooks/xg_classfication.ipynb`
- `notebooks/xg_regression.ipynb`
- 결과 산출물: `outputs/xgboost_confidence_classification`, `outputs/xgboost_confidence_regression`

## 1. 실험 구조

두 노트북은 BSD10k의 `confidence` 라벨 1~5를 예측한다. 공통 전처리는 다음과 같다.

- `confidence`가 1~5인 샘플만 사용
- `class_idx`가 `*99`, `*00` 형태인 항목 제거
- audio CLAP embedding + text CLAP embedding + class one-hot을 결합
- `method2`는 여기에 `description` 문자열 길이 1차원을 추가
- stratified split: train 60%, validation 20%, test 20%

classification 노트북 기준 실행된 데이터 규모:

- 전체 샘플: 10,956
- split: train 6,573 / validation 2,191 / test 2,192
- class category 수: 23
- confidence 분포: 1=106, 2=749, 3=3,280, 4=6,045, 5=776

라벨 분포가 confidence 4에 강하게 몰려 있다. 전체 데이터 중 confidence 4가 약 55.2%이고, confidence 1은 약 1.0%뿐이라 macro 지표와 confusion matrix에서 불균형 문제가 크게 드러난다.

## 2. 모델 구조

### XGBClassifier

목적: confidence 1~5를 직접 5-class classification으로 예측한다.

주요 설정:

- `XGBClassifier`
- `n_estimators=1200`
- `max_depth=3`
- `learning_rate=0.03`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `objective='multi:softprob'`
- `num_class=5`
- `eval_metric='mlogloss'`
- `tree_method='hist'`
- `early_stopping_rounds=60`

출력은 내부 class 0~4로 학습한 뒤 최종 예측에서 1~5로 복원한다.

### XGBRegressor

목적: confidence를 연속값 1~5로 회귀 예측한 뒤, `clip(1, 5)` 및 round로 class화한다.

주요 설정:

- `XGBRegressor`
- `n_estimators=1600`
- `max_depth=3`
- `learning_rate=0.03`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `objective='reg:squarederror'`
- `eval_metric='mae'`
- `tree_method='hist'`
- `early_stopping_rounds=60`

회귀 모델의 MAE는 round된 class가 아니라 연속 예측 score 기준으로 계산되어 있다.

## 3. 결과 비교

### Classification 결과

| feature set | accuracy | macro precision | macro recall | macro F1 | MAE | best iteration |
|---|---:|---:|---:|---:|---:|---:|
| method2: text+audio+label+desc_len | 0.5912 | 0.5824 | 0.2978 | 0.3242 | 0.4699 | 428 |
| method1: text+audio+label | 0.5899 | 0.5781 | 0.2974 | 0.3238 | 0.4745 | 363 |

`description length`를 추가한 method2가 아주 근소하게 더 좋다. 차이는 accuracy +0.14%p, macro F1 +0.0003 수준이라 실질적으로는 거의 같은 모델로 보는 편이 안전하다.

### Regression 결과

| feature set | accuracy | macro precision | macro recall | macro F1 | MAE | best iteration |
|---|---:|---:|---:|---:|---:|---:|
| method1: text+audio+label | 0.5661 | 0.3977 | 0.2811 | 0.2786 | 0.5308 | 1035 |
| method2: text+audio+label+desc_len | 0.5625 | 0.3837 | 0.2795 | 0.2769 | 0.5325 | 1022 |

회귀 모델은 classification보다 accuracy, macro F1, MAE가 모두 낮다. 특히 round 후 class로 바꾸는 과정에서 중간값 3~4로 예측이 더 심하게 압축된다.

### 종합 순위

현재 산출물 기준 가장 좋은 모델은 `XGBClassifier + method2`다.

- accuracy 최고: 0.5912
- macro F1 최고: 0.3242
- MAE 최저: 0.4699

다만 성능 차이가 크지 않고, macro recall이 약 0.30에 머문다는 점에서 “confidence 1~5를 균형 있게 맞추는 모델”이라기보다는 “majority인 4를 중심으로 안정적으로 맞추는 모델”에 가깝다.

## 4. Loss 분석

### XGBClassifier mlogloss

`method2` loss curve를 보면 train mlogloss는 약 1.10에서 0.62 근처까지 꾸준히 감소한다. validation mlogloss는 초반 빠르게 감소한 뒤 약 0.96 근처에서 평탄해진다.

해석:

- 학습 데이터에는 계속 적합하고 있지만 validation 개선은 일찍 둔화된다.
- train/validation gap이 시간이 갈수록 커져 약한 과적합이 있다.
- early stopping으로 큰 붕괴는 막았지만, class imbalance 때문에 validation loss가 더 이상 크게 낮아지지 않는 구조다.

### XGBRegressor MAE

`method1` 회귀 loss curve는 train MAE가 약 0.63에서 0.36까지 계속 낮아지고, validation MAE는 약 0.52 근처에서 완만하게 정체된다.

해석:

- 회귀 모델도 train/validation gap이 분명하다.
- validation MAE는 classification MAE보다 나빠서, 연속 score 회귀가 confidence class 경계를 잘 학습하지 못했다.
- 회귀 objective가 “순서형 라벨”이라는 특성에는 맞지만, 라벨 불균형이 강하면 평균 쪽으로 수렴하는 경향이 커진다.

## 5. Confusion Matrix 분석

### Best classification: method2

row-normalized confusion matrix 핵심:

| true | pred 1 | pred 2 | pred 3 | pred 4 | pred 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 9.5% | 0.0% | 38.1% | 52.4% | 0.0% |
| 2 | 0.7% | 6.7% | 30.7% | 62.0% | 0.0% |
| 3 | 0.0% | 0.9% | 31.1% | 68.0% | 0.0% |
| 4 | 0.0% | 0.4% | 11.1% | 87.5% | 1.0% |
| 5 | 0.0% | 0.0% | 3.8% | 82.1% | 14.1% |

관찰:

- true 4는 87.5%로 매우 잘 맞춘다.
- true 3도 68.0%가 4로 밀려 올라간다.
- true 5는 82.1%가 4로 내려간다.
- true 1, 2는 대부분 3 또는 4로 예측된다.
- 즉, 모델이 “확신도 4”를 기본값처럼 사용하는 강한 중앙/다수 클래스 편향을 가진다.

### Best regression: method1

row-normalized confusion matrix 핵심:

| true | pred 1 | pred 2 | pred 3 | pred 4 | pred 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0% | 9.5% | 81.0% | 9.5% | 0.0% |
| 2 | 0.0% | 2.7% | 70.0% | 27.3% | 0.0% |
| 3 | 0.0% | 0.8% | 59.5% | 39.8% | 0.0% |
| 4 | 0.0% | 0.3% | 30.1% | 68.8% | 0.7% |
| 5 | 0.0% | 0.0% | 7.1% | 83.3% | 9.6% |

관찰:

- 회귀 모델은 1과 5 같은 극단 class를 거의 예측하지 않는다.
- classification보다 pred 3 비율이 커서 전체적으로 3~4 사이로 압축된다.
- true 4의 recall도 classification 87.5%보다 낮은 68.8%다.

## 6. 핵심 결론

1. 현재 목적이 confidence 1~5 class 예측이라면 `XGBClassifier method2`가 가장 낫다.
2. `description length` 추가 효과는 매우 작다. 단일 feature로는 성능을 크게 끌어올리지 못했다.
3. 전체 성능은 accuracy 기준 약 59%지만, 이는 confidence 4 쏠림의 영향이 크다.
4. macro F1이 약 0.324로 낮기 때문에 minority confidence 1, 2, 5 예측력은 부족하다.
5. 회귀 접근은 ordinal 특성을 반영한다는 장점은 있으나, 현재 결과에서는 class imbalance와 평균 회귀 현상 때문에 classification보다 불리하다.

## 7. 개선 제안

우선순위 높은 개선:

- class weight 또는 sample weight 적용: confidence 1, 2, 5 recall 개선 목적
- ordinal classification 방식 도입: 1~5 단순 softmax 대신 `P(y > k)` 형태 또는 ordinal loss 사용
- threshold 튜닝: classification probability를 argmax로만 쓰지 말고, class 5/2 쪽 decision threshold 보정
- macro F1 기준 validation selection: 현재 early stopping은 mlogloss/MAE 기준이라 minority class F1과 직접 맞지 않음

추가 실험:

- `description length` 대신 텍스트 품질 feature 추가: 단어 수, 태그 수, class-name 포함 여부, CLAP text-audio cosine similarity
- label smoothing 또는 focal loss 계열 모델과 비교
- confidence를 5-class 대신 low/mid/high 또는 clean/noisy binary로 재정의한 downstream filtering 성능 비교

현재 결과만 놓고 보면, XGBoost는 강한 baseline으로는 쓸 수 있지만 “confidence 라벨의 세밀한 1~5 구분기”로 쓰기에는 부족하다. 실사용 필터링 목적이라면 5-class 정확도보다 “낮은 confidence를 얼마나 잘 잡는가” 또는 “높은 confidence를 얼마나 안전하게 보존하는가” 쪽으로 objective를 다시 잡는 것이 좋다.
