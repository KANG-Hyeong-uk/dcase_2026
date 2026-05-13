# summarize_results.py - 결과 요약 및 분석 가이드

## 📋 파일 개요
`summarize_results.py`는 K-Fold 교차 검증의 **결과를 통합하여 요약**합니다.
각 Fold의 결과를 수집하고 평균 ± 표준편차를 계산합니다.

---

## 🔍 주요 함수 분석

### summarize_metrics() 함수

```python
def summarize_metrics(root_dir="model_output", output_file="summary_metrics.txt"):
    """
    Scan experiment folders for results.txt files, extract metrics,
    and compute mean ± std across folds per experiment in the 'root_dir'.
    """
    
    pattern = re.compile(r'([a-zA-Z0-9_]+):\s*([\d.]+)%')
    # 정규표현식: "Metric_Name: 85.23%"를 파싱
    
    metrics_data = defaultdict(lambda: defaultdict(list))
    # 구조: {experiment_name: {metric_name: [fold1_value, fold2_value, ...]}}
```

---

## 📂 폴더 구조 예상

```
model_output/
├── BSD10k-v1.2_audio/
│   ├── fold_0/
│   │   ├── best_model.pth
│   │   ├── history.json
│   │   └── results.txt  ← 여기서 메트릭 추출
│   ├── fold_1/
│   │   └── results.txt
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   └── summary_metrics.txt  ← 생성되는 파일
│
├── BSD10k-v1.2_both/
│   ├── fold_0/
│   ├── fold_1/
│   └── results.txt (fold별)
│
└── (다른 실험들...)
```

---

## 🔄 동작 흐름

### 1️⃣ 파일 스캔

```python
for root, _, files in os.walk(root_dir):
    if "results.txt" in files:
        file_path = os.path.join(root, "results.txt")
```

**찾는 것**:
- `model_output/` 디렉토리 하위의 모든 `results.txt` 파일
- 각 Fold의 평가 결과 포함

---

### 2️⃣ 실험명 추출

```python
rel_path = os.path.relpath(root, root_dir)
# 예: "BSD10k-v1.2_both/fold_0"

parts = rel_path.split(os.sep)
# ["BSD10k-v1.2_both", "fold_0"]

parts = [p for p in parts if not p.lower().startswith("fold")]
# ["BSD10k-v1.2_both"]

exp_name = "/".join(parts) if parts else "root"
# "BSD10k-v1.2_both"
```

**목적**: Fold 이름 제거 → 실험명만 추출

---

### 3️⃣ 메트릭 파싱

```python
with open(file_path, "r") as f:
    content = f.read()

for match in pattern.finditer(content):
    name = match.group(1)      # "Micro_Accuracy"
    value = float(match.group(2))  # 85.23
    metrics_data[exp_name][name].append(value)
```

**정규표현식 패턴**:
```
r'([a-zA-Z0-9_]+):\s*([\d.]+)%'

예시 매칭:
Micro_Accuracy: 85.23%
├─ Group 1: Micro_Accuracy
├─ Group 2: 85.23
└─ 결과: ("Micro_Accuracy", 85.23)
```

**results.txt 파일 예시**:
```
Test Results for Fold 0 (mode=audio):
────────────────────────────────
Micro_Accuracy: 82.34%
Macro_Accuracy: 80.12%
Hierarchical_Accuracy: 81.45%
Hierarchical_Precision: 79.23%
Hierarchical_Recall: 80.67%
Hierarchical_F1: 79.94%
```

---

### 4️⃣ 통계 계산

```python
for exp_name, metrics in sorted(metrics_data.items()):
    out.write(f"{exp_name}\n")
    
    for metric_name, values in sorted(metrics.items()):
        if values:
            mean = np.mean(values)
            std = np.std(values)
            out.write(f"  {metric_name:12s}: {mean:.2f}% ± {std:.2f}%\n")
        else:
            out.write(f"  {metric_name:12s}: No data\n")
    
    num_runs = len(next(iter(metrics.values()), []))
    out.write(f"  runs        : {num_runs}\n\n")
```

**계산 내용**:
- **평균 (Mean)**: 5개 Fold 결과의 평균
- **표준편차 (Std)**: 결과의 변동성

**예시**:
```
Fold 값: [85.2, 84.8, 85.5, 85.1, 84.9]
평균: (85.2 + 84.8 + 85.5 + 85.1 + 84.9) / 5 = 85.1%
표준편차: sqrt(Var) ≈ 0.26%
결과: 85.1% ± 0.26%
```

---

## 📊 출력 파일 형식

### summary_metrics.txt 예시

```
=== Experiment Summary ===

BSD10k-v1.2_audio
  Micro_Accuracy:  82.34% ± 1.23%
  Macro_Accuracy:  79.87% ± 1.45%
  Hierarchical_Accuracy:  80.56% ± 1.12%
  Hierarchical_Precision:  78.45% ± 1.67%
  Hierarchical_Recall:  79.23% ± 1.34%
  Hierarchical_F1:  78.83% ± 1.51%
  runs        : 5

BSD10k-v1.2_both
  Micro_Accuracy:  85.67% ± 0.89%
  Macro_Accuracy:  83.45% ± 1.02%
  Hierarchical_Accuracy:  84.56% ± 0.95%
  Hierarchical_Precision:  82.34% ± 1.12%
  Hierarchical_Recall:  83.12% ± 0.98%
  Hierarchical_F1:  82.72% ± 1.05%
  runs        : 5

BSD35k-CS_audio
  Micro_Accuracy:  78.90% ± 2.34%
  Macro_Accuracy:  76.45% ± 2.56%
  ...
  runs        : 5
```

---

## 🧮 해석 가이드

### 평균 ± 표준편차의 의미

```
85.67% ± 0.89%

의미:
- 평균 성능: 85.67%
- 변동성: 0.89%
- 신뢰 구간 (95%): 약 83.89% ~ 87.45%

평가:
- 표준편차가 작음 → 안정적인 모델 ✓
- 표준편차가 큼 → 불안정한 모델 ✗
```

---

### 실험 비교 예시

```
오디오만 vs 멀티모달:

BSD10k-v1.2_audio:
  Micro_Accuracy: 82.34% ± 1.23%

BSD10k-v1.2_both:
  Micro_Accuracy: 85.67% ± 0.89%

결론:
- 멀티모달이 3.33% 높은 성능
- 멀티모달이 더 안정적 (표준편차 작음)
- 멀티모달 권장 ✓
```

---

## 💡 활용 방법

### 1️⃣ 실행

```bash
python summarize_results.py
```

### 2️⃣ 출력 위치

```
model_output/summary_metrics.txt
```

### 3️⃣ 결과 확인

```python
# Python에서 프로그래밍 방식 접근
from summarize_results import summarize_metrics

metrics_data = summarize_metrics(
    root_dir="model_output",
    output_file="summary_metrics.txt"
)

# metrics_data 구조:
# {
#   'BSD10k-v1.2_audio': {
#     'Micro_Accuracy': [82.1, 82.5, 82.2, 82.4, 82.3],
#     'Macro_Accuracy': [79.8, 80.0, 79.9, 80.1, 79.7],
#     ...
#   },
#   'BSD10k-v1.2_both': {
#     'Micro_Accuracy': [85.6, 85.7, 85.8, 85.5, 85.7],
#     ...
#   }
# }

# 직접 분석
for exp_name, metrics in metrics_data.items():
    micro_acc = metrics['Micro_Accuracy']
    print(f"{exp_name}")
    print(f"  평균: {np.mean(micro_acc):.2f}%")
    print(f"  표준편차: {np.std(micro_acc):.2f}%")
```

---

## 📈 고급 분석 (확장 가능)

### Fold별 성능 변동 시각화

```python
import matplotlib.pyplot as plt

# 각 Fold의 성능을 플롯
exp_name = 'BSD10k-v1.2_both'
metric_values = metrics_data[exp_name]['Micro_Accuracy']

plt.figure(figsize=(8, 4))
plt.plot(range(1, 6), metric_values, marker='o', label='Fold별 정확도')
plt.axhline(y=np.mean(metric_values), color='r', linestyle='--', label='평균')
plt.fill_between(
    range(1, 6),
    np.mean(metric_values) - np.std(metric_values),
    np.mean(metric_values) + np.std(metric_values),
    alpha=0.2, label='±1 표준편차'
)
plt.xlabel('Fold Number')
plt.ylabel('Accuracy (%)')
plt.title(f'{exp_name} - 성능 안정성')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 모든 실험 비교

```python
# 모든 실험의 micro accuracy 비교
exp_names = list(metrics_data.keys())
means = [
    np.mean(metrics_data[exp]['Micro_Accuracy'])
    for exp in exp_names
]
stds = [
    np.std(metrics_data[exp]['Micro_Accuracy'])
    for exp in exp_names
]

plt.figure(figsize=(12, 6))
plt.bar(range(len(exp_names)), means, yerr=stds, capsize=5)
plt.xticks(range(len(exp_names)), exp_names, rotation=45, ha='right')
plt.ylabel('Micro Accuracy (%)')
plt.title('모든 실험 비교')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

---

## 🔧 문제 해결

### 결과가 나타나지 않음

**확인 사항**:
1. ✓ `results.txt` 파일 존재 여부
   ```bash
   find model_output -name "results.txt"
   ```

2. ✓ 파일 형식 확인
   ```
   Metric_Name: 85.23%  ← 이 형식이어야 함
   ```

3. ✓ 정규표현식 매칭 테스트
   ```python
   import re
   pattern = re.compile(r'([a-zA-Z0-9_]+):\s*([\d.]+)%')
   test_text = "Micro_Accuracy: 85.23%"
   match = pattern.search(test_text)
   print(match.groups())  # ('Micro_Accuracy', '85.23')
   ```

---

## 🎯 학습 포인트

- ✅ **파일 스캔**: os.walk로 재귀적 검색
- ✅ **정규표현식**: 파싱으로 구조화된 데이터 추출
- ✅ **통계**: 평균과 표준편차로 성능 정량화
- ✅ **비교**: 실험 간 차이를 수치로 평가
- ✅ **재현성**: 표준편차로 불안정성 감지
