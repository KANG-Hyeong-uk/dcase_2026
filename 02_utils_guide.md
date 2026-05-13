# utils.py - 유틸리티 함수 완벽 가이드

## 📋 파일 개요
`utils.py`는 전체 프로젝트에서 **반복적으로 사용되는 도우미 함수들**을 모아둔 파일입니다.
설정 로드, 난수 시드 설정, 클래스 매핑 등의 공통 기능을 제공합니다.

---

## 🔍 함수별 상세 분석

### 1️⃣ 설정 로드 함수

#### load_config()
```python
def load_config(path='config.yaml'):
    config_dir = os.path.dirname(os.path.realpath(__file__)) 
    config_path = os.path.join(config_dir, path) 
    
    if not os.path.exists(config_path): 
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

**기능**: YAML 설정 파일을 Python 딕셔너리로 로드
- **현재 파일 위치** 기준으로 상대 경로 계산
- **존재 확인** 후 로드
- **YAML** 형식을 파이썬 dict로 변환

**사용 예시**:
```python
config = load_config()  # config.yaml 로드
print(config['active_dataset'])  # 'BSD10k-v1.2'
print(config['datasets'])         # 전체 데이터셋 설정
```

---

#### get_subconfig()
```python
def get_subconfig(section, path='config.yaml'):
    config = load_config(path)
    return config.get(section, {})
```

**기능**: 설정 파일의 특정 섹션만 추출
- **섹션 이름** 지정하면 해당 부분만 반환
- 없으면 **빈 딕셔너리** 반환

**사용 예시**:
```python
# ✅ 활성 데이터셋 이름
dataset_name = get_subconfig("active_dataset")
# 반환: "BSD10k-v1.2"

# ✅ 데이터셋 설정 전체
datasets = get_subconfig("datasets")
# 반환: {'BSD10k-v1.2': {...}, 'BSD35k-CS': {...}}

# ✅ 출력 경로
output_path = get_subconfig("output_path")
# 반환: "data"
```

---

### 2️⃣ 가중치 초기화 함수

#### xavier_init()
```python
def xavier_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)
```

**기능**: Xavier 균등 초기화 (Linear 레이어용)
- **목적**: 심층 네트워크에서 gradient vanishing/explosion 방지
- **적용**: Linear 레이어만

**이론**:
```
W ~ Uniform(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

장점:
- 활성화 함수 출력이 균일한 분포 유지
- ReLU 계열 함수에는 효과 적음
```

---

#### kaiming_init()
```python
def kaiming_init(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight)
```

**기능**: Kaiming 균등 초기화 (ReLU와 함께 사용)
- **목적**: ReLU 활성화 함수의 특성을 고려한 초기화
- **추천**: ResNet, LeakyReLU 등에 사용

**이론**:
```
W ~ Uniform(-√(6/n_in), √(6/n_in))

장점:
- ReLU의 0 으로의 bias를 고려
- 더 빠른 수렴
```

---

### 3️⃣ 클래스 매핑 함수

#### build_class_to_topclass_mapping()
```python
def build_class_to_topclass_mapping(class_dict, top_class_dict):
    class_to_topclass = {}

    for class_name, class_id in class_dict.items():
        for top_class_name, top_class_dict.items():
            if class_name.startswith(top_class_name):
                class_to_topclass[class_id] = top_class_id
                break

    return class_to_topclass
```

**기능**: 하위 클래스 → 상위 클래스 매핑 생성
- **딕셔너리 형식** 반환

**입력 예시**:
```python
class_dict = {
    'dog-barking': 0,       # 하위 클래스
    'dog-growling': 1,
    'speech-male': 2,
    'speech-female': 3
}

top_class_dict = {
    'dog': 0,               # 상위 클래스
    'speech': 1
}
```

**출력**:
```python
{
    0: 0,  # 'dog-barking' → 'dog'
    1: 0,  # 'dog-growling' → 'dog'
    2: 1,  # 'speech-male' → 'speech'
    3: 1   # 'speech-female' → 'speech'
}
```

---

#### build_class_to_topclass_tensor()
```python
def build_class_to_topclass_tensor(class_dict, top_class_dict, device):
    num_classes = len(class_dict)
    class_to_topclass = torch.zeros(num_classes, dtype=torch.long, device=device)
    
    for class_name, class_id in class_dict.items():
        for top_class_name, top_class_id in top_class_dict.items():
            if class_name.startswith(top_class_name):
                class_to_topclass[class_id] = top_class_id
                break
    
    return class_to_topclass
```

**기능**: 위와 동일하나 **PyTorch Tensor** 형식 반환
- **GPU 이동** 가능 (device 지정)
- **배치 연산** 최적화

**사용 이유**:
```python
# 수천 개 샘플에서 상위 클래스 빠르게 조회
# 딕셔너리: class_to_topclass[idx] → O(1)
# 하지만 GPU 연산이면 Tensor가 효율적

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mapping = build_class_to_topclass_tensor(class_dict, top_class_dict, device)
```

---

#### build_id_to_class_mapping()
```python
def build_id_to_class_mapping(class_dict):
    """Return a mapping from class ID to class name."""
    return {class_id: class_name for class_name, class_id in class_dict.items()}
```

**기능**: 역방향 매핑 (ID → 클래스명)
- **예측 후** 나중에 클래스명 복원할 때 사용

**예시**:
```python
# 입력
class_dict = {'dog-barking': 0, 'dog-growling': 1}

# 출력
id_to_class = {0: 'dog-barking', 1: 'dog-growling'}

# 사용
pred_id = 0
pred_class = id_to_class[pred_id]  # 'dog-barking'
```

---

### 4️⃣ 클래스명 분석 함수

#### extend_subcat()
```python
def extend_subcat(subcat):
    if "-" not in subcat:
        raise Exception("invalid subcat name: " + subcat + ", top level not found <top>-<subcat>")
    top_level = subcat.split("-")[0]
    return (top_level, subcat)
```

**기능**: 하위 클래스명을 분석해 상위+하위 튜플 반환
- **형식**: "상위-하위" (예: "dog-barking")

**사용 예시**:
```python
subcat = "dog-barking"
result = extend_subcat(subcat)
# 반환: ('dog', 'dog-barking')

top_level, full_name = extend_subcat("speech-male")
# top_level = 'speech'
# full_name = 'speech-male'
```

---

#### get_top_level()
```python
def get_top_level(subcat):
  return extend_subcat(subcat)[0]
```

**기능**: 클래스명에서만 상위 클래스 추출
- **간편함**: extend_subcat 보다 간단

**사용 예시**:
```python
get_top_level('dog-barking')      # 'dog'
get_top_level('speech-female')    # 'speech'
```

---

#### intersection()
```python
def intersection(list1, list2):
  return list(set(list1).intersection(list2))
```

**기능**: 두 리스트의 공통 원소 반환
- **용도**: 계층 구조 평가에서 pathwise 겹침 계산

**사용 예시**:
```python
pred_path = ('dog', 'dog-barking')
gt_path = ('dog', 'dog-growling')
common = intersection(pred_path, gt_path)
# 반환: ['dog']  (상위 클래스만 일치)
```

---

### 5️⃣ 재현성 함수

#### set_seed()
```python
def set_seed(seed=1821):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
```

**기능**: 모든 난수 생성기에 **동일 시드** 설정
- **목적**: 논문 재현 가능성 확보

**설정하는 것들**:
| 라이브러리 | 목적 |
|----------|------|
| random | Python 기본 난수 |
| numpy | NumPy 배열 연산 |
| torch | PyTorch CPU 연산 |
| torch.cuda | PyTorch GPU 연산 |

**사용 방법**:
```python
# 프로그램 시작 지점에서
seed = set_seed()
print(f"Seed: {seed}")  # Seed: 1821

# 같은 결과 재현
set_seed(1821)
arr1 = np.random.randn(5)

set_seed(1821)
arr2 = np.random.randn(5)

print(np.array_equal(arr1, arr2))  # True
```

**⚠️ 주의**: GPU 사용 시에도 완벽 재현 불가능 (부동소수 오차)

---

### 6️⃣ 조기 종료 클래스

#### EarlyStopping
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

**기능**: 과적합 방지를 위한 조기 종료 관리
- **patience**: 개선 없을 때 최대 에포크 수
- **min_delta**: 개선으로 인정될 최소 손실 감소량

**사용 예시**:
```python
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

for epoch in range(100):
    # 훈련 코드...
    val_loss = validate(model)
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("조기 종료!")
        break

# 로직:
# - Epoch 1: loss=0.5 (첫 값 저장)
# - Epoch 2: loss=0.45 ✓ (개선됨 → counter=0)
# - Epoch 3: loss=0.48 ✗ (악화 → counter=1)
# - Epoch 4: loss=0.49 ✗ (악화 → counter=2)
# - ...
# - Epoch 11: counter=10 → early_stop=True
```

**파라미터 선택 가이드**:
```
patience=5   → 빨리 종료 (과민함)
patience=20  → 늦게 종료 (과면함)
min_delta=0.001 → 아주 작은 개선도 인정
```

---

## 📊 전체 함수 호출 흐름

```
train_test.py 실행
  ↓
set_seed() → 모든 난수 생성기 초기화
  ↓
build_class_to_topclass_tensor() → 클래스 매핑 생성
  ↓
get_subconfig() → config.yaml에서 설정 로드
  ↓
EarlyStopping() → 조기 종료 관리자 생성
  ↓
훈련 루프
  └─ 에포크마다 EarlyStopping 체크
```

---

## 🎯 학습 포인트

- ✅ **설정 관리**: load_config, get_subconfig로 중앙화된 설정
- ✅ **초기화**: Xavier vs Kaiming은 활성화 함수에 따라 선택
- ✅ **클래스 매핑**: 계층 구조를 딕셔너리/Tensor로 변환
- ✅ **재현성**: set_seed로 동일 결과 보증 (논문 필수!)
- ✅ **조기 종료**: 훈련 시간 단축 + 과적합 방지
