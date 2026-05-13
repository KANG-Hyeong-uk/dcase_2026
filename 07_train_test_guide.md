# train_test.py - 훈련 루프 완벽 가이드

## 📋 파일 개요
`train_test.py`는 **핵심 훈련 및 테스트 파이프라인**을 구현합니다.
K-Fold 교차 검증, 하이퍼파라미터 설정, 모델 저장 등을 담당합니다.

---

## 🔄 전체 파이프라인 흐름

```
config.yaml 읽기
     ↓
processed_dataset.csv 로드
     ↓
클래스 딕셔너리 로드
     ↓
K-Fold 데이터 분할
     ↓
각 Fold별:
  ├─ 훈련/검증/테스트 분할
  ├─ 모델 초기화
  ├─ 훈련 루프 (에포크 반복)
  │  ├─ Forward pass
  │  ├─ 손실 계산
  │  ├─ Backward pass
  │  └─ 가중치 업데이트
  ├─ 검증 (조기 종료 확인)
  ├─ 최적 모델 저장
  └─ 테스트 세트 평가
     ↓
결과 요약 및 저장
```

---

## 🔍 주요 함수 상세 분석

### 1️⃣ 가중치 초기화

```python
def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out')
    elif isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)
```

**목적**: 모델의 모든 가중치를 최적 상태로 초기화

| 레이어 | 초기화 방식 | 이유 |
|-------|----------|------|
| Conv2D | Kaiming Normal | CNN의 표준 |
| Linear | Xavier Uniform | 심층 신경망 최적 |

---

### 2️⃣ JSON 직렬화

```python
def make_serializable(obj, decimals=6):
    """Recursively convert tensors, numpy arrays, and numbers to JSON-serializable types"""
    
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
        return make_serializable(obj, decimals)
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return round(float(obj), decimals)  # 스칼라
        else:
            return [make_serializable(x, decimals) for x in obj]  # 배열
    elif isinstance(obj, float):
        return round(obj, decimals)  # 소수점 6자리 반올림
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, collections.abc.Mapping):
        return {k: make_serializable(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
        return [make_serializable(x, decimals) for x in obj]
    else:
        return obj
```

**목적**: Tensor/NumPy 객체를 JSON 형식으로 저장
- 훈련 히스토리 저장
- 주의 점수 저장
- 재현성 위해 소수점 6자리로 반올림

**사용 예시**:
```python
history = {
    'train_loss': [tensor(0.5), tensor(0.3), ...],
    'val_accuracy': [0.75, 0.82, ...],
    'attention': np.array([0.7, 0.3])
}

json_safe = make_serializable(history)
# 결과:
# {
#   'train_loss': [0.5, 0.3, ...],
#   'val_accuracy': [0.75, 0.82, ...],
#   'attention': [0.7, 0.3]
# }

with open('history.json', 'w') as f:
    json.dump(json_safe, f)
```

---

### 3️⃣ train_model() - 핵심 훈련 함수

```python
def train_model(model, train_loader, val_loader, device,
                num_epochs=100, lr=0.001, classification_weight=1.0, 
                classification_criterion=None, output_dir='model_output', 
                scheduler_type='plateau', patience=10, early_stopping_factor=5):
```

**파라미터**:

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| **model** | - | 훈련할 모델 |
| **train_loader** | - | 훈련용 데이터 로더 |
| **val_loader** | - | 검증용 데이터 로더 |
| **device** | - | CPU 또는 GPU |
| **num_epochs** | 100 | 최대 훈련 에포크 |
| **lr** | 0.001 | 학습률 |
| **classification_weight** | 1.0 | 분류 손실 가중치 |
| **classification_criterion** | - | 손실 함수 |
| **output_dir** | 'model_output' | 모델 저장 폴더 |
| **scheduler_type** | 'plateau' | 'plateau' 또는 'step' |
| **patience** | 10 | 학습률 감소 대기 에포크 |
| **early_stopping_factor** | 5 | 조기 종료: patience × factor |

---

#### 훈련 루프 상세 분석

```python
# 1. 옵티마이저 설정
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# weight_decay: L2 정규화 (과적합 방지)

# 2. 학습률 스케줄러 설정
if scheduler_type == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience, verbose=True
    )
    # 검증 정확도가 개선 안 되면 학습률을 0.5배로 감소
elif scheduler_type == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )
    # 20 에포크마다 학습률을 0.5배로 감소
```

**학습률 스케줄러 비교**:

```
ReduceLROnPlateau (동적):
학습률
 ↑ ┌─────┐
   │     └───┐
   │         └──┐
   │            └────  ← 개선 없을 때 감소
   └────────────→ 에포크

StepLR (고정):
학습률
 ↑ ┌─────────────────┐
   │                 └──────────┐
   │                            └──  ← 정기적으로 감소
   └───────────────────────→ 에포크
```

**권장**:
- 동적이 필요하면 → `plateau`
- 안정성 중시 → `step`

---

#### 에포크 루프 (훈련)

```python
for epoch in range(num_epochs):
    model.train()  # 훈련 모드 (드롭아웃, 배치정규화 활성화)
    losses = defaultdict(float)
    total_samples = 0

    attn_audio_epoch = []
    attn_text_epoch = []

    # 배치 반복
    for data in train_loader:
        class_labels = data['class_idx'].to(device)
        audio_emb = data.get('audio_embedding', None)
        text_emb = data.get('text_embedding', None)
        
        if audio_emb is not None:
            audio_emb = audio_emb.to(device)
        if text_emb is not None:
            text_emb = text_emb.to(device)

        optimizer.zero_grad()  # 그래디언트 초기화
        
        # Forward pass
        z, class_logit, attn_scores = model(audio_emb, text_emb)
        
        # 주의 점수 수집
        if attn_scores is not None:
            attn_audio_epoch.append(attn_scores[:, 0].detach().cpu())
            attn_text_epoch.append(attn_scores[:, 1].detach().cpu())

        batch_size = class_labels.size(0)
        total_samples += batch_size

        # 손실 계산
        if classification_criterion is not None:
            cls_loss = classification_criterion(class_logit, class_labels)
            losses['cls'] += cls_loss.item() * batch_size
            total_loss = classification_weight * cls_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()  # 가중치 업데이트
        losses['total'] += total_loss.item() * batch_size
```

**주의 점수 수집**:
```python
# 배치마다 주의 가중치 저장
# 나중에 에포크 평균 계산

if attn_audio_epoch:
    attn_audio_epoch = torch.cat(attn_audio_epoch, dim=0)
    history["attention_audio"].append(attn_audio_epoch.mean(0).numpy())
```

---

#### 검증 단계

```python
model.eval()  # 평가 모드 (드롭아웃 비활성화)
correct = 0
total = 0

with torch.no_grad():  # 그래디언트 계산 안 함
    for data in val_loader:
        labels = data['class_idx'].to(device)
        audio_emb = data.get('audio_embedding', None)
        text_emb = data.get('text_embedding', None)
        
        if audio_emb is not None:
            audio_emb = audio_emb.to(device)
        if text_emb is not None:
            text_emb = text_emb.to(device)

        _, class_logit, _ = model(audio_emb, text_emb)
        
        _, predicted = torch.max(class_logit.data, 1)  # 최대 확률 클래스
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
```

---

#### 모델 저장 및 조기 종료

```python
if val_accuracy > best_accuracy:
    best_accuracy = val_accuracy
    
    # 모델 설정 저장
    model_config = {
        'hidden_size': hidden_size,
        'num_classes': len(class_dict),
        'emb_size_audio': emb_size_audio,
        'emb_size_text': emb_size_text,
        'dropout': dropout,
        'use_batch_norm': True,
        'mode': mode,
    }

    # 체크포인트 저장
    torch.save({
        'model_state': model.state_dict(),
        'config': model_config,
    }, os.path.join(output_dir, "best_model.pth"))
    
    print(f"  New best model saved")
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience * early_stopping_factor:
        print("Early stopping triggered.")
        break
```

**조기 종료 로직**:
```
patience = 5, early_stopping_factor = 3
조기 종료 임계값 = 5 × 3 = 15 에포크

- Epoch 1~5: 개선 있음 → counter = 0
- Epoch 6~15: 개선 없음 → counter 증가
- Epoch 15 후미: counter = 15 → 훈련 종료
```

---

### 4️⃣ 메인 실행 루프

```python
if __name__ == "__main__":
    seed = set_seed()  # 재현성 확보
    
    # 클래스 딕셔너리 로드
    with open(class_dict_json, 'r') as f:
        class_dict = json.load(f)
    with open(top_class_dict_json, 'r') as f:
        top_class_dict = json.load(f)
    
    # 하이퍼파라미터
    modes = ['both', 'audio']  # 모드별로 실험
    model_output = './model_output'
    
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    scheduler_type = 'step'
    patience = 5
    early_stopping_factor = 3
    k_folds = 5
```

---

#### K-Fold 교차 검증

```python
full_df = pd.read_csv(prepared_dataset_path)
labels = full_df["class_idx"].tolist()

# 5-겹 교차 검증 분할기
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n==== Fold {fold} ====")
    
    # 훈련+검증 데이터에서 다시 분할 (80:20)
    trainval_labels = [labels[i] for i in trainval_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx_rel, val_idx_rel = next(sss.split(np.zeros(len(trainval_labels)), trainval_labels))
    
    train_idx = [trainval_idx[i] for i in train_idx_rel]
    val_idx = [trainval_idx[i] for i in val_idx_rel]
    
    # 데이터 분할
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)
```

**분할 비율 시각화**:
```
전체 데이터: 9876개
    ↓
Fold 1 (80:20 split):
┌─────────────────┬──────────┐
│  trainval (80%) │ test (20%)│
│     7900        │   1976   │
│  ├─ train 80%   │          │
│  │   6320       │          │
│  └─ val 20%     │          │
│      1580       │          │
└─────────────────┴──────────┘

Fold 2, 3, 4, 5 반복
```

---

## 📊 데이터 흐름 예시

### 단일 Fold의 전체 과정

```
입력:
- train_df: 6320 샘플
- val_df: 1580 샘플
- test_df: 1976 샘플

1. 모델 초기화
   model = BaseClassifier(...)

2. DataLoader 생성
   train_loader: 98 배치 (batch_size=64)
   val_loader: 25 배치
   test_loader: 31 배치

3. 훈련 (100 에포크)
   Epoch 1: train loss=2.8, val_acc=35%
   Epoch 2: train loss=2.1, val_acc=52%
   ...
   Epoch 25: train loss=0.5, val_acc=85% ← 최고 모델 저장
   ...
   Epoch 45: 개선 없음 → 조기 종료

4. 테스트
   best_model.pth 로드
   테스트 정확도 계산

출력:
- model_output/fold_0/best_model.pth
- model_output/fold_0/history.json
- model_output/fold_0/results.txt
```

---

## 🧪 실행 예시

```bash
# 훈련 시작
python train_test.py

# 출력:
# ===== Dataset: BSD10k-v1.2 full | Mode=both =====
# ==== Fold 0 ====
# Epoch [1/100] - Val acc: 35.44%
# Epoch [2/100] - Val acc: 52.12%
# ...
# Epoch [25/100] - Val acc: 85.67%
#   New best model saved
# ...
# Early stopping triggered at Epoch 45
# Fold 0 completed!
#
# ==== Fold 1 ====
# ...
```

---

## 🎯 학습 포인트

- ✅ **K-Fold CV**: 데이터 활용 극대화 + 안정적인 평가
- ✅ **조기 종료**: 훈련 시간 절약 + 과적합 방지
- ✅ **학습률 스케줄링**: 수렴 안정성 향상
- ✅ **주의 점수 기록**: 모달리티 기여도 분석
- ✅ **체크포인트 저장**: 최적 모델만 보존
- ✅ **계층적 전략**: train/val/test 명확 분리
