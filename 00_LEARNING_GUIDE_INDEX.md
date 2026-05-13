# DCASE 2026 Task 1 - 전체 학습 가이드 목차

## 📚 모든 가이드 모음

이 문서는 프로젝트의 모든 Python 파일에 대한 **상세한 학습 자료**에 대한 색인입니다.
각 단계를 순서대로 학습하거나, 필요한 파일만 선택적으로 학습할 수 있습니다.

---

## 🎯 권장 학습 순서

### 1단계: 프로젝트 구조 이해 (15분)
- [README.md](README.md) - 프로젝트 개요
- [10_main_guide.md](10_main_guide.md) - 전체 파이프라인

### 2단계: 설정 및 유틸리티 (30분)
- [01_config_yaml_guide.md](01_config_yaml_guide.md) - 설정 관리
- [02_utils_guide.md](02_utils_guide.md) - 헬퍼 함수들

### 3단계: 데이터 처리 (45분)
- [03_build_dataset_guide.md](03_build_dataset_guide.md) - 데이터셋 전처리
- [04_dataset_utils_guide.md](04_dataset_utils_guide.md) - 데이터 로더

### 4단계: 모델 설계 (60분)
- [05_models_guide.md](05_models_guide.md) - 신경망 아키텍처 (가장 중요!)
- [06_losses_guide.md](06_losses_guide.md) - 손실 함수

### 5단계: 훈련 및 평가 (60분)
- [07_train_test_guide.md](07_train_test_guide.md) - 훈련 루프
- [08_evaluate_guide.md](08_evaluate_guide.md) - 평가 메트릭

### 6단계: 결과 분석 (15분)
- [09_summarize_results_guide.md](09_summarize_results_guide.md) - 결과 요약

**전체 소요 시간: 약 3.5시간**

---

## 📋 파일별 가이드 상세 설명

### 1. [`01_config_yaml_guide.md`](01_config_yaml_guide.md)
**주제**: 설정 파일 관리
- ✅ 데이터셋 경로 설정
- ✅ 입출력 경로 정의
- ✅ 하이퍼파라미터 설정

**학습 내용**:
```yaml
active_dataset: BSD10k-v1.2        # 사용 데이터셋 선택
datasets:
  - 메타데이터 경로
  - 임베딩 폴더 경로
  - 클래스 정보 경로
```

**중요도**: ⭐⭐⭐ (필수)
**예상 시간**: 15분

---

### 2. [`02_utils_guide.md`](02_utils_guide.md)
**주제**: 유틸리티 함수 모음
- ✅ YAML 설정 로드
- ✅ 가중치 초기화
- ✅ 클래스 매핑 생성
- ✅ 난수 시드 설정
- ✅ 조기 종료 관리

**주요 함수**:
```python
get_subconfig()                          # 설정 로드
xavier_init(), kaiming_init()            # 가중치 초기화
build_class_to_topclass_mapping()        # 클래스 매핑
set_seed()                               # 재현성
EarlyStopping                            # 과적합 방지
```

**중요도**: ⭐⭐⭐ (필수)
**예상 시간**: 20분

---

### 3. [`03_build_dataset_guide.md`](03_build_dataset_guide.md)
**주제**: 데이터셋 전처리
- ✅ 메타데이터 로드
- ✅ 클래스 필터링
- ✅ 인덱스 재매핑
- ✅ CSV 생성

**처리 절차**:
```
원본 메타데이터
  ↓ (필터링: 상위 클래스, "-other" 제거)
필터링된 데이터
  ↓ (인덱스 재매핑: 0부터 시작)
학습용 데이터셋 (processed_dataset.csv)
```

**중요도**: ⭐⭐⭐ (필수)
**예상 시간**: 20분

---

### 4. [`04_dataset_utils_guide.md`](04_dataset_utils_guide.md)
**주제**: PyTorch 데이터 로더
- ✅ HATRDataset 클래스
- ✅ 임베딩 로드
- ✅ 온더플라이 증강
- ✅ 계층적 레이블

**주요 기능**:
```python
class HATRDataset(Dataset):
  - 음성/텍스트 임베딩 동시 제공
  - 가우시안 노이즈 + 랜덤 마스킹
  - 상위/하위 클래스 레이블
```

**중요도**: ⭐⭐⭐ (필수)
**예상 시간**: 20분

---

### 5. [`05_models_guide.md`](05_models_guide.md) ⭐ 가장 중요!
**주제**: 신경망 아키텍처
- ✅ ResidualBlock (잔차 블록)
- ✅ EmbeddingEncoder (임베딩 인코더)
- ✅ AttentionFusion (주의 기반 융합)
- ✅ BaseClassifier (메인 분류기)

**핵심 아이디어**:
```
음성 임베딩 (512D)         텍스트 임베딩 (512D)
    ↓                           ↓
[EmbeddingEncoder]       [EmbeddingEncoder]
    ↓ (256D)                ↓ (256D)
    └─────────[AttentionFusion]─────────┘
               ↓ (동적 가중치 학습)
          [분류기]
               ↓
          23개 클래스
```

**주요 개념**:
- 잔차 연결 (Residual Connection)
- 배치 정규화 (Batch Normalization)
- 멀티모달 학습 (Multimodal Learning)
- 주의 메커니즘 (Attention Mechanism)

**중요도**: ⭐⭐⭐⭐⭐ (매우 필수)
**예상 시간**: 60분

---

### 6. [`06_losses_guide.md`](06_losses_guide.md)
**주제**: 손실 함수
- ✅ Cross Entropy Loss
- ✅ Label Smoothing
- ✅ 손실값 해석

**핵심 개념**:
```
표준 CE: 모델이 출력한 확률의 음의 로그
Label Smoothing (ε=0.01):
  - 정답 클래스: 0.9896 (완벽한 1이 아님)
  - 오답 클래스: 0.0004 (0이 아님)
  - 효과: 과신 방지, 일반화 성능 향상
```

**중요도**: ⭐⭐⭐ (필수)
**예상 시간**: 15분

---

### 7. [`07_train_test_guide.md`](07_train_test_guide.md)
**주제**: 훈련 루프 및 K-Fold CV
- ✅ train_model() 함수
- ✅ K-Fold 교차 검증 (5-fold)
- ✅ 조기 종료 (Early Stopping)
- ✅ 학습률 스케줄링
- ✅ 모델 체크포인트 저장

**훈련 과정**:
```
Fold 1: train(80%) | val(10%) | test(10%)
Fold 2: train(80%) | val(10%) | test(10%)
...
Fold 5: train(80%) | val(10%) | test(10%)

각 fold에서 검증 성능이 가장 좋은 모델 저장
```

**중요도**: ⭐⭐⭐⭐ (매우 필수)
**예상 시간**: 40분

---

### 8. [`08_evaluate_guide.md`](08_evaluate_guide.md)
**주제**: 평가 메트릭
- ✅ 미시/거시 정확도
- ✅ 계층적 정확도
- ✅ 정밀도/재현율/F1
- ✅ 가중 계층적 메트릭

**메트릭 체계**:
```
단순 정확도:
  - 미시 (Micro): 전체 샘플 기준
  - 거시 (Macro): 클래스별 평균

계층적 메트릭:
  - 상위 클래스 일치는 부분 점수 (λ=0.5)
  - 경로 교집합 기반 계산
```

**중요도**: ⭐⭐⭐ (필수)
**예상 시간**: 25분

---

### 9. [`09_summarize_results_guide.md`](09_summarize_results_guide.md)
**주제**: 결과 통합 및 요약
- ✅ 파일 스캔 (os.walk)
- ✅ 정규표현식 파싱
- ✅ 통계 계산 (평균 ± 표준편차)
- ✅ 최종 보고서 생성

**출력 예시**:
```
BSD10k-v1.2_audio
  Micro_Accuracy: 82.34% ± 1.23%
  Hierarchical_Accuracy: 80.56% ± 1.12%
  ...

BSD10k-v1.2_both
  Micro_Accuracy: 85.67% ± 0.89%
  Hierarchical_Accuracy: 84.56% ± 0.95%
  ...
```

**중요도**: ⭐⭐⭐ (필수)
**예상 시간**: 15분

---

### 10. [`10_main_guide.md`](10_main_guide.md)
**주제**: 전체 파이프라인 통합 실행
- ✅ subprocess.run() 사용법
- ✅ 순차 실행
- ✅ 오류 처리
- ✅ 실행 흐름도

**실행 명령**:
```bash
python main.py  # 전체 파이프라인 자동 실행

내부 실행 순서:
1. build_dataset.py    (데이터 준비)
2. train_test.py       (모델 훈련)
3. summarize_results.py (결과 요약)
```

**중요도**: ⭐⭐⭐ (권장)
**예상 시간**: 10분

---

## 🚀 빠른 시작 (Quick Start)

### 처음 실행하는 경우

```bash
# 1. 환경 설정
conda create -n hac python=3.13
conda activate hac
pip install -r requirements.txt

# 2. 전체 파이프라인 실행
python main.py

# 3. 결과 확인
cat model_output/summary_metrics.txt
```

---

## 📊 학습 경로별 추천

### 경로 1: 완전 초보자
1. [`01_config_yaml_guide.md`](01_config_yaml_guide.md)
2. [`02_utils_guide.md`](02_utils_guide.md)
3. [`03_build_dataset_guide.md`](03_build_dataset_guide.md)
4. [`04_dataset_utils_guide.md`](04_dataset_utils_guide.md)
5. [`05_models_guide.md`](05_models_guide.md) ⭐
6. [`06_losses_guide.md`](06_losses_guide.md)
7. [`07_train_test_guide.md`](07_train_test_guide.md)
8. [`08_evaluate_guide.md`](08_evaluate_guide.md)
9. [`09_summarize_results_guide.md`](09_summarize_results_guide.md)
10. [`10_main_guide.md`](10_main_guide.md)

**소요 시간**: 3.5시간

---

### 경로 2: 머신러닝 경험자
1. [`01_config_yaml_guide.md`](01_config_yaml_guide.md) (5분 훑기)
2. [`02_utils_guide.md`](02_utils_guide.md) (10분 - 핵심만)
3. [`05_models_guide.md`](05_models_guide.md) ⭐ (60분 정독)
4. [`07_train_test_guide.md`](07_train_test_guide.md) (30분)
5. [`08_evaluate_guide.md`](08_evaluate_guide.md) (20분)

**소요 시간**: 2시간

---

### 경로 3: 깊이 있는 학습
모든 가이드 정독 + 코드 분석
**소요 시간**: 5시간

---

## 💡 가이드 활용 팁

### 1. 동시에 코드 열기
```bash
# VS Code에서
code 05_models_guide.md
code dcase2026_task1_baseline/models.py
# 가이드와 코드를 나란히 읽기
```

### 2. 개념을 실습하며 배우기
```python
# 개념 학습 후 직접 시도해보기

# 예: 클래스 매핑 학습 후
from utils import build_class_to_topclass_mapping
import json

with open('./data/class_dict.json') as f:
    class_dict = json.load(f)

# 직접 매핑 생성해보기
mapping = build_class_to_topclass_mapping(class_dict, ...)
```

### 3. 주요 개념 메모
각 가이드의 **🎯 학습 포인트** 섹션 참고

### 4. 실전 프로젝트 진행
1. 가이드 학습
2. 동일한 코드 스스로 작성
3. 프로젝트 수정/확장

---

## 🎓 심화 학습 (다음 단계)

각 가이드를 마친 후 다음을 시도해보세요:

### 모델 개선
- [ ] 다른 아키텍처 시도 (Transformer, Graph Neural Network)
- [ ] 하이퍼파라미터 튜닝
- [ ] 데이터 증강 기법 추가
- [ ] 앙상블 모델 구현

### 기능 추가
- [ ] 실시간 음성 분류
- [ ] 설명 가능성 (Explainability)
- [ ] 웹 인터페이스 구현
- [ ] 배포 (Docker, Flask)

### 논문 작성
- [ ] 실험 결과 정리
- [ ] 비교 분석 (baseline vs 개선 모델)
- [ ] 기여도 정리

---

## 🔗 추가 자료

### 공식 문서
- [DCASE Challenge 2026](https://dcase.community/challenge2026/)
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [CLAP 모델](https://github.com/LAION-AI/CLAP/)

### 관련 논문
- [HATR: Heterogeneous Audio Tagging with Residual Networks](https://arxiv.org/abs/2404.XXXXX)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

## ❓ FAQ

### Q1: 어느 가이드부터 시작해야 할까요?
A: 초보자는 순서대로, 경험자는 05_models_guide부터 시작하세요.

### Q2: 각 파일의 중요도가 어떻게 됩니까?
A: 05_models_guide가 가장 중요합니다 (⭐⭐⭐⭐⭐).

### Q3: 코드를 수정하고 싶습니다. 어떤 파일부터 보면 좋을까요?
A: 수정 대상에 따라:
- 모델 구조: 05_models_guide
- 훈련 방식: 07_train_test_guide
- 평가 메트릭: 08_evaluate_guide

### Q4: K-Fold 검증이란 무엇인가요?
A: 07_train_test_guide의 "K-Fold 교차 검증" 섹션을 참고하세요.

### Q5: 계층적 메트릭은 왜 필요할까요?
A: 08_evaluate_guide의 "배경: 계층적 분류" 섹션을 참고하세요.

---

## 📞 피드백 및 개선

가이드에 오류나 개선 사항이 있으면 공유해주세요!

---

**Last Updated**: 2026년 4월 28일
**Version**: 1.0
**Total Pages**: ~50 pages
