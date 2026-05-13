# main.py - 전체 파이프라인 통합 실행 가이드

## 📋 파일 개요
`main.py`는 전체 훈련 파이프라인을 **자동으로 순차 실행**하는 통합 진입점입니다.
복잡한 실행 순서를 단순화하고, 한 번의 명령으로 전체 작업을 수행합니다.

---

## 🔍 코드 분석

```python
import subprocess
import sys


def run(script):
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed!")

pipeline = [
    ("Dataset", "build_dataset.py"),
    ("Training", "train_test.py"),
    ("Results", "summarize_results.py"),
]

for name, script in pipeline:
    print(f"\n------- Process: {name} -------")
    print(f"\nRunning {script}...\n")
    run(script)
    print(f"\nFinished {script}")

print("Processes completed. Check the output folders for results.")
```

---

## 📊 파이프라인 구조

### 전체 실행 흐름

```
main.py 실행
    ↓
Step 1: Dataset 준비
    └─ build_dataset.py 실행
        ├─ config.yaml 읽기
        ├─ 메타데이터 로드
        ├─ 임베딩 폴더 스캔
        ├─ 필터링 (상위 클래스, "-other" 제거)
        ├─ 인덱스 재매핑
        └─ 출력 생성:
            ├─ processed_dataset.csv
            ├─ class_dict.json
            ├─ top_class_dict.json
            └─ top_class_subclass_dict.json
    ↓
Step 2: 모델 훈련
    └─ train_test.py 실행
        ├─ K-Fold 분할 (5-fold)
        ├─ 각 Fold에서:
        │   ├─ train/val/test 분할
        │   ├─ 모델 초기화
        │   ├─ 훈련 루프 (조기 종료)
        │   ├─ 최적 모델 저장
        │   └─ 테스트 평가
        └─ 출력 생성:
            └─ model_output/
                ├─ BSD10k-v1.2_audio/
                │   ├─ fold_0/
                │   │   ├─ best_model.pth
                │   │   ├─ history.json
                │   │   └─ results.txt
                │   └─ fold_1~4/
                └─ BSD10k-v1.2_both/
                    └─ fold_0~4/
    ↓
Step 3: 결과 요약
    └─ summarize_results.py 실행
        ├─ model_output/ 스캔
        ├─ 모든 fold의 results.txt 파싱
        ├─ 평균 ± 표준편차 계산
        └─ 최종 보고서 생성:
            └─ model_output/summary_metrics.txt
```

---

## 💻 subprocess.run() 상세

```python
def run(script):
    # subprocess.run(): 외부 프로세스 실행
    # [sys.executable, script]
    # ├─ sys.executable: 현재 Python 인터프리터 경로
    # └─ script: 실행할 Python 스크립트
    
    result = subprocess.run([sys.executable, script])
    
    # result.returncode
    # ├─ 0: 성공 (정상 종료)
    # └─ 0이 아님: 실패 (에러 발생)
    
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed!")
        # 스크립트 실패 시 예외 발생 → 파이프라인 중단
```

**실행 원리**:
```bash
# 내부적으로 실행되는 명령
python build_dataset.py
python train_test.py
python summarize_results.py
```

---

## 🔄 각 단계별 상세

### Step 1: 데이터셋 준비 (build_dataset.py)

**실행 명령**:
```bash
python build_dataset.py
```

**역할**:
- 원본 메타데이터 + 임베딩 → 훈련용 데이터셋
- 필터링 및 정규화
- 클래스 인덱스 매핑

**출력 파일**:
```
data/
├── processed_dataset.csv       (훈련용 메인 데이터)
├── class_dict.json            (클래스명 ↔ ID 매핑)
├── top_class_dict.json        (상위 클래스 매핑)
└── top_class_subclass_dict.json (계층 구조)
```

**소요 시간**: 약 1~5분 (데이터셋 크기에 따라)

---

### Step 2: 모델 훈련 (train_test.py)

**실행 명령**:
```bash
python train_test.py
```

**역할**:
- K-Fold 교차 검증 (5-fold)
- 각 Fold에서 훈련/검증/테스트
- 조기 종료로 훈련 시간 단축
- 최과 모델 저장

**하이퍼파라미터** (스크립트 내부에서 설정):
```python
batch_size = 64
num_epochs = 100
learning_rate = 0.001
scheduler_type = 'step'
patience = 5
early_stopping_factor = 3
k_folds = 5
modes = ['both', 'audio']  # 2가지 모드 훈련
```

**출력 파일**:
```
model_output/
├── BSD10k-v1.2_audio/       (오디오만)
│   ├── fold_0/
│   │   ├── best_model.pth
│   │   ├── history.json      (훈련 곡선)
│   │   └── results.txt       (평가 메트릭)
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
├── BSD10k-v1.2_both/        (멀티모달)
│   ├── fold_0~4/
│   ├── ...
│   └── ...
```

**소요 시간**: 약 2~4시간 (GPU 기준) / 10~20시간 (CPU 기준)

---

### Step 3: 결과 요약 (summarize_results.py)

**실행 명령**:
```bash
python summarize_results.py
```

**역할**:
- 모든 Fold 결과 수집
- 메트릭 평균 ± 표준편차 계산
- 최종 보고서 생성

**출력 파일**:
```
model_output/summary_metrics.txt

예시:
=== Experiment Summary ===

BSD10k-v1.2_audio
  Micro_Accuracy:  82.34% ± 1.23%
  Macro_Accuracy:  79.87% ± 1.45%
  ...
  runs        : 5

BSD10k-v1.2_both
  Micro_Accuracy:  85.67% ± 0.89%
  Macro_Accuracy:  83.45% ± 1.02%
  ...
  runs        : 5
```

**소요 시간**: 약 1~2분

---

## 📋 실행 예시 및 예상 출력

### 명령줄 실행

```bash
python main.py
```

### 예상 출력

```
------- Process: Dataset -------

Running build_dataset.py...

Examining original data from BSD10k-v1.2:
  Total rows: 10352
  Unique classes: 23
After filtering: 9876

Processing complete!
Generated files:
  - data/processed_dataset.csv
  - data/class_dict.json
  - data/top_class_dict.json
  - data/top_class_subclass_dict.json

Finished build_dataset.py

------- Process: Training -------

Running train_test.py...

=== Dataset: BSD10k-v1.2 full ===

=== Running experiments: Dataset=BSD10k-v1.2 full | Mode=both ===

==== Fold 0 ====
Epoch [1/100] - Val acc: 35.44%
Epoch [2/100] - Val acc: 52.12%
...
Epoch [25/100] - Val acc: 85.67%
  New best model saved
...
Epoch [45/100] - Early stopping triggered.

==== Fold 1 ====
...

=== Running experiments: Dataset=BSD10k-v1.2 full | Mode=audio ===

==== Fold 0 ====
...

Finished train_test.py

------- Process: Results -------

Running summarize_results.py...

Summary written to model_output/summary_metrics.txt

Finished summarize_results.py

Processes completed. Check the output folders for results.
```

---

## ⚠️ 오류 처리

### 파이프라인이 중단되는 경우

```python
# Build Dataset 실패
if result.returncode != 0:
    raise RuntimeError(f"build_dataset.py failed!")
    # → 훈련 단계로 진행하지 않음

# Training 실패
if result.returncode != 0:
    raise RuntimeError(f"train_test.py failed!")
    # → 결과 요약 단계로 진행하지 않음
```

**해결 방법**:
1. 개별 스크립트 직접 실행으로 오류 원인 파악
   ```bash
   python build_dataset.py  # 어디서 오류?
   ```

2. 로그 확인 및 config.yaml 경로 검증

3. 데이터 파일 존재 여부 확인

---

## 🔧 커스터마이제이션

### 특정 단계만 실행

```python
# main.py 수정 (필요한 스크립트만 선택)

pipeline = [
    # ("Dataset", "build_dataset.py"),  # 주석 처리 (스킵)
    ("Training", "train_test.py"),
    ("Results", "summarize_results.py"),
]
```

### 추가 스크립트 실행

```python
# evaluate.py도 포함
pipeline = [
    ("Dataset", "build_dataset.py"),
    ("Training", "train_test.py"),
    ("Evaluation", "evaluate.py"),  # 새 단계
    ("Results", "summarize_results.py"),
]
```

---

## 📊 실행 흐름도

```
시작
  ↓
config.yaml 읽기
  ↓
build_dataset.py 실행 (미스매칭 데이터셋 준비)
  ├─ 성공 → ⭕
  └─ 실패 → ❌ 중단
  ↓
train_test.py 실행 (모델 훈련 및 테스트)
  ├─ 5-Fold × 2 Mode = 10개 실험
  ├─ 각 실험의 best_model.pth 저장
  ├─ 성공 → ⭕
  └─ 실패 → ❌ 중단
  ↓
summarize_results.py 실행 (결과 통합)
  ├─ 모든 결과 파싱
  ├─ 평균 ± 표준편차 계산
  ├─ summary_metrics.txt 생성
  └─ 완료 → ✅
  ↓
종료
```

---

## 🎯 학습 포인트

- ✅ **subprocess.run()**: 외부 Python 스크립트 실행
- ✅ **순차 실행**: 각 단계의 출력이 다음 단계의 입력
- ✅ **오류 처리**: 실패 시 파이프라인 중단
- ✅ **단순화**: 복잡한 파이프라인을 한 명령으로 실행
- ✅ **확장성**: 필요시 스크립트 추가/제거 가능

---

## 🚀 권장 사용 방법

### 처음 실행 (전체 파이프라인)
```bash
python main.py
# 데이터 준비 → 훈련 → 결과 요약
```

### 이미 훈련된 모델이 있고, 결과만 다시 요약
```python
# main.py 수정
pipeline = [
    # ("Dataset", "build_dataset.py"),
    # ("Training", "train_test.py"),
    ("Results", "summarize_results.py"),
]
```

### 데이터만 다시 준비 (새 데이터셋)
```bash
python build_dataset.py
# build_dataset.py만 실행
```
