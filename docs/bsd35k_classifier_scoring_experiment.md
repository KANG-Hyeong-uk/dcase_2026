# BSD35k-CS Classifier 기반 Label Quality 분석

작성일: 2026-05-05

## 목적

이 실험은 이미 학습된 BSD10k baseline classifier를 사용해서 BSD35k-CS의 라벨 품질을 분석한다.

중요한 구분:

- 이 실험은 새로운 confidence classifier를 학습하지 않는다.
- 기존 BSD10k class classifier를 재사용해서 BSD35k-CS에 적용한다.
- 출력값은 classifier 기반 label reliability score이며, expert `confidence` 필드를 직접 예측한 값은 아니다.

이 실험이 답하려는 질문은 다음과 같다.

- BSD10k로 학습한 모델이 BSD35k-CS에서 제공한 라벨에 동의하는가?
- 동의하지 않는다면, 같은 top-level class 안에서의 불일치인가, 아니면 top-level class 자체가 다른가?
- 어떤 class와 uploader가 학습에 위험해 보이는가?

## 사용한 모델

기존 full BSD10k baseline classifier checkpoint를 사용했다.

```text
dcase2026_task1_baseline/model_output/both/fold_0/best_model.pth
dcase2026_task1_baseline/model_output/both/fold_1/best_model.pth
dcase2026_task1_baseline/model_output/both/fold_2/best_model.pth
dcase2026_task1_baseline/model_output/both/fold_3/best_model.pth
dcase2026_task1_baseline/model_output/both/fold_4/best_model.pth
```

모델은 `both` mode이며 입력 구조는 다음과 같다.

```text
audio CLAP embedding + text CLAP embedding -> 23-class BSD classifier
```

5개 fold 모델의 softmax 출력을 평균해서 ensemble prediction으로 사용했다.

기존 baseline validation/test 요약:

```text
accuracy: 79.63% +/- 0.39%
top accuracy: 88.88% +/- 0.15%
hierarchical F1: 78.33% +/- 0.41%
```

## 사용한 데이터

BSD35k-CS metadata:

```text
data/metadata/BSD35k-CS_metadata.csv
```

BSD35k-CS embedding:

```text
data/features/BSD35k_clap_audio_embeddings
data/features/BSD35k-CS_clap_text_embeddings
```

스크립트는 `config.yaml`의 BSD35k 경로가 BSD10k embedding folder를 가리키고 있음을 감지했고, 실제 BSD35k embedding folder를 대신 사용했다.

Scoring한 row 수:

```text
31,464 / 31,464
```

## 방법

각 BSD35k-CS sample에 대해 다음 과정을 수행했다.

1. audio/text CLAP embedding을 load한다.
2. BSD10k classifier 5개 fold를 실행한다.
3. 5개 softmax probability vector를 평균한다.
4. classifier prediction과 BSD35k-CS 제공 라벨을 비교한다.

주요 출력 컬럼:

```text
provided_class
provided_top_class
predicted_class
predicted_top_class
classifier_confidence
provided_class_probability
classifier_margin
ensemble_disagreement
same_class
same_top_class
attention_audio
attention_text
prob_<class>
```

컬럼 의미:

- `classifier_confidence`: ensemble prediction의 최대 확률.
- `provided_class_probability`: BSD35k-CS 제공 라벨에 할당된 확률.
- `same_class`: `provided_class == predicted_class` 여부.
- `same_top_class`: top-level class가 같은지 여부.
- `ensemble_disagreement`: fold 간 probability 표준편차의 평균.

## 출력 파일

주요 script:

```text
dcase2026_task1_baseline/score_bsd35k_with_classifier.py
```

Notebook:

```text
notebooks/01_bsd35k_classifier_scoring_analysis.ipynb
```

생성 파일:

```text
experiments/bsd35k_scoring/BSD35k-CS_classifier_scores.csv
experiments/bsd35k_scoring/BSD35k_classifier_scores_by_class.csv
experiments/bsd35k_scoring/BSD35k_classifier_scores_by_uploader.csv
experiments/bsd35k_scoring/classifier_scoring_summary.json
```

## 주요 결과

전체 요약:

```text
rows scored: 31,464
same_class_rate: 57.02%
same_top_class_rate: 75.14%
mean_classifier_confidence: 0.764
mean_provided_class_probability: 0.506
```

해석:

- BSD35k-CS 라벨 중 약 57%만 BSD10k classifier의 exact class prediction과 일치했다.
- 약 75%는 top-level hierarchy에서 일치했다.
- BSD35k-CS에는 상당한 label noise 또는 domain shift가 포함되어 있을 가능성이 있다.
- DCASE metric은 hierarchical structure를 사용하므로 top-level mismatch가 특히 위험하다.

## 위험한 Class

Class agreement가 가장 낮은 class:

```text
fx-m   n=1542  same_class=24.25%  same_top=93.32%  provided_prob=0.218
is-p   n=2321  same_class=24.47%  same_top=31.28%  provided_prob=0.224
ss-i   n=383   same_class=24.80%  same_top=44.65%  provided_prob=0.211
is-k   n=43    same_class=27.91%  same_top=53.49%  provided_prob=0.271
sp-c   n=50    same_class=32.00%  same_top=54.00%  provided_prob=0.272
fx-ex  n=839   same_class=33.61%  same_top=60.19%  provided_prob=0.280
fx-el  n=2000  same_class=34.85%  same_top=77.40%  provided_prob=0.319
```

관찰:

- `fx-m`은 exact class agreement는 낮지만 top-level agreement는 높다. 따라서 대부분 `fx` 내부 subclass 혼동일 가능성이 있다.
- `is-p`, `ss-i`, `m-sp`, 일부 `is` class는 top-level agreement도 낮아 hierarchical evaluation에서 더 위험하다.

Class agreement가 가장 높은 class:

```text
fx-v   n=1852  same_class=83.26%  same_top=94.87%
m-m    n=2044  same_class=78.33%  same_top=91.88%
fx-o   n=4534  same_class=77.68%  same_top=95.37%
sp-s   n=980   same_class=69.90%  same_top=77.24%
fx-n   n=659   same_class=68.44%  same_top=93.02%
ss-n   n=4831  same_class=68.04%  same_top=73.13%
```

이 class들은 high-confidence BSD35k training sample 후보로 더 적합하다.

## 위험한 Uploader

sample이 10개 이상인 uploader 중 class agreement가 0%인 사례:

```text
BEZUPRECHNOST    n=15  same_class=0.00%  same_top=6.67%
brokenmachinery  n=14  same_class=0.00%  same_top=50.00%
Department64     n=13  same_class=0.00%  same_top=100.00%
derjuli          n=22  same_class=0.00%  same_top=50.00%
Pate141          n=18  same_class=0.00%  same_top=0.00%
1volcano         n=15  same_class=0.00%  same_top=0.00%
charonfaustinus  n=11  same_class=0.00%  same_top=90.91%
dOPdOPdOP        n=10  same_class=0.00%  same_top=0.00%
```

Uploader 단위 filtering 또는 down-weighting이 유용할 가능성이 있다.

## 결과 활용 방법

Classifier 기반 score로 다음과 같은 sample group을 정의할 수 있다.

```text
clean candidate:
same_class == True
provided_class_probability high

hierarchical hard candidate:
same_class == False
same_top_class == True
classifier_confidence high

dangerous/noisy candidate:
same_top_class == False
classifier_confidence high
provided_class_probability low
```

활용처:

- BSD35k-CS 학습 전 filtering
- sample weighting
- pseudo-labeling 후보 선정
- class-specific cleaning
- uploader-specific cleaning

## 이 실험이 하지 않은 것

이 실험은 BSD10k expert `confidence` 값을 직접 학습하지 않는다.

이를 위해서는 다음과 같은 별도 모델을 학습해야 한다.

```text
input: audio embedding + text embedding + optional metadata/class features
target: BSD10k confidence value
```

그 뒤 BSD35k-CS에 적용해서 다음 값을 생성할 수 있다.

```text
predicted_annotation_confidence
confidence_bucket
```

최종적으로는 두 신호를 함께 사용하는 것이 가장 좋다.

```text
classifier-based label agreement
+ predicted expert-style annotation confidence
```
