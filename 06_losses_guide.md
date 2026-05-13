# losses.py - 손실 함수 완벽 가이드

## 📋 파일 개요
`losses.py`는 모델 훈련 중 사용되는 **손실 함수**를 정의합니다.
현재는 CrossEntropyLoss with label smoothing을 구현하고 있습니다.

---

## 🔍 CrossEntropyLoss 상세 분석

### 클래스 구조
```python
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.01)

    def forward(self, logits, labels):
        return self.cross_entropy(logits, labels)
```

---

## 📚 Cross Entropy Loss 이론

### 기본 개념

**Cross Entropy Loss의 정의**:
```
CE = -∑ y * log(p)

여기서:
- y: 정답 레이블 (원-핫 인코딩)
- p: 모델이 예측한 확률
```

**구체적 예시**:
```
정답 클래스: 3번 (dog-barking)
모델이 예측한 로짓: [-1.2, 0.5, -0.3, 2.1, ...]  (23차원)

단계 1: Softmax로 확률로 변환
probs = softmax(logits) 
      ≈ [0.01, 0.05, 0.03, 0.85, ...] (합=1)

단계 2: 정답 클래스의 음의 로그 확률 계산
loss = -log(p[3])
     = -log(0.85)
     ≈ 0.16
```

**해석**:
- 모델이 정답을 높은 확률로 예측 → loss는 작음 ✓
- 모델이 정답을 낮은 확률로 예측 → loss는 큼 ✗

---

## 🎯 Label Smoothing

### 문제점: 과신 (Overconfidence)

표준 Cross Entropy Loss에서:
```
정답: [0, 0, 1, 0, ...]  (원-핫)
model이 예측: 정답 클래스 확률 = 0.99

손실: -log(0.99) ≈ 0.01  (매우 작음!)
```

**문제**: 
- 모델이 너무 자신감 있게 학습
- 과적합 위험 증가
- 새로운 데이터에 일반화 어려움

### 해결책: Label Smoothing (ε = 0.01)

```
기존 정답:        [0.00, 0.00, 1.00, 0.00, ...]
Label Smoothing:  [0.0004, 0.0004, 0.9896, 0.0004, ...]
                                    ↑ 약간 감소
                  ↑ 다른 클래스에 작은 확률 할당
```

**수식**:
```
y'_i = (1 - ε) * y_i + ε / K

여기서:
- ε: smoothing 계수 (0.01)
- K: 클래스 수 (23)
- y_i: 원본 레이블

예시:
- 정답 클래스: (1 - 0.01) * 1 + 0.01/23 = 0.9896
- 오답 클래스: (1 - 0.01) * 0 + 0.01/23 = 0.0004
```

**효과**:
```
1. 모델이 과도한 자신감 피함
2. 예측 확률이 더 균형잡힘
3. 일반화 성능 증가 (특히 소규모 데이터셋)
4. 불확실성을 모델이 더 정직하게 표현
```

---

## 📊 Label Smoothing의 시각화

### 수렴 곡선 비교

```
손실
 ↑
 │      Label Smoothing
 │  ✓   /
 │     /
 │    /    표준 CS
 │   /   /
 │  /  /
 │ / /
 │ X____________
 └─────────────→ 에포크
```

- **표준**: 빠르게 떨어지지만, 과적합 발생
- **Smoothing**: 천천히 떨어지지만, 안정적인 수렴

### 확률 분포 비교

```
표준 Cross Entropy:
클래스  확률
 0:    0.001%
 1:    0.001%
 2:    99.9%   ← 과신
 3:    0.001%
 ...

Label Smoothing (ε=0.01):
클래스  확률
 0:    0.04%
 1:    0.04%
 2:    98.96%  ← 적절한 자신감
 3:    0.04%
 ...
```

---

## 💻 Using CrossEntropyLoss in Training

### 기본 사용법

```python
from losses import CrossEntropyLoss
import torch

# 손실 함수 생성
criterion = CrossEntropyLoss()

# 배치 데이터
logits = torch.randn(64, 23)    # (배치크기=64, 클래스=23)
labels = torch.randint(0, 23, (64,))  # (배치크기=64,)

# 손실 계산
loss = criterion(logits, labels)
print(loss)  # 스칼라 값 (예: tensor(2.8234))

# 역전파
loss.backward()
```

---

### 훈련 루프에서의 사용

```python
def train_model(model, train_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    
    for epoch in range(100):
        total_loss = 0
        
        for batch in train_loader:
            audio_emb = batch['audio_embedding'].to(device)
            text_emb = batch['text_embedding'].to(device)
            labels = batch['class_idx'].to(device)
            
            # Forward pass
            _, logits, _ = model(audio_emb, text_emb)
            
            # 손실 계산
            loss = criterion(logits, labels)  # ← 여기서 사용
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

---

## 📈 Label Smoothing의 영향 분석

### ε 값의 선택

| ε 값 | 특징 | 추천 상황 |
|------|------|---------|
| 0.00 | 표준 CE (smoothing 없음) | 큰 데이터셋 |
| 0.01 | 약한 smoothing (권장) | 중간 크기 데이터셋 (기본값) |
| 0.05 | 중정도 smoothing | 작은 데이터셋, 노이즈 많음 |
| 0.10 | 강한 smoothing | 극도로 작은 데이터셋 |

---

### ε 값별 확률 분포 변화 (K=23)

```python
K = 23  # 클래스 수

# 정답 클래스 확률
eps_values = [0.00, 0.01, 0.05, 0.10]

for eps in eps_values:
    correct_prob = (1 - eps) * 1 + eps / K
    wrong_prob = (1 - eps) * 0 + eps / K
    
    print(f"ε={eps:0.2f}")
    print(f"  정답 클래스:  {correct_prob:.4f}")
    print(f"  오답 클래스:  {wrong_prob:.4f}")
    print()

# 출력:
# ε=0.00
#   정답 클래스:  1.0000
#   오답 클래스:  0.0000
#
# ε=0.01
#   정답 클래스:  0.9896
#   오답 클래스:  0.0004
#
# ε=0.05
#   정답 클래스:  0.9478
#   오답 클래스:  0.0022
#
# ε=0.10
#   정답 클래스:  0.8957
#   오답 클래스:  0.0043
```

---

## 🔄 다른 손실 함수 설계 아이디어

현재 프로젝트에서는 기본 CE를 사용하지만, 다음과 같은 확장 가능성이 있습니다:

### 1. 계층적 손실 (Hierarchical Loss)
```python
class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.01)
        self.alpha = alpha  # 손실 가중치
    
    def forward(self, sub_logits, top_logits, sub_labels, top_labels):
        # 하위 클래스 손실
        sub_loss = self.ce_loss(sub_logits, sub_labels)
        
        # 상위 클래스 손실 (더 약함)
        top_loss = self.ce_loss(top_logits, top_labels)
        
        # 결합
        total_loss = sub_loss + self.alpha * top_loss
        return total_loss
```

**이유**: 계층 구조 활용 → 더 나은 학습

### 2. 초점 손실 (Focal Loss)
```python
# 드물거나 어려운 샘플에 높은 가중치
gamma = 2  # 집중도 조절

focal_loss = -alpha * (1 - p_t)^gamma * log(p_t)
```

**이유**: 불균형 데이터셋 대비

### 3. BCE Loss (이진 분류)
```python
# 클래스별로 다중 이진 분류
per_class_loss = nn.BCEWithLogitsLoss()
```

**이유**: 다중 레이블 분류

---

## 🧪 디버깅: 손실값 확인

### 손실이 발산하는 경우

```python
criterion = CrossEntropyLoss()

# 테스트 케이스 1: 모든 예측이 같은 경우
logits = torch.ones(4, 23)
labels = torch.tensor([0, 1, 2, 3])
loss = criterion(logits, labels)
print(f"모두 같은 로짓: {loss:.4f}")  # 약 3.14 (log(23))

# 테스트 케이스 2: 완벽한 예측
logits = torch.zeros(4, 23)
logits[0, 0] = 100
logits[1, 1] = 100
logits[2, 2] = 100
logits[3, 3] = 100
labels = torch.tensor([0, 1, 2, 3])
loss = criterion(logits, labels)
print(f"완벽한 예측: {loss:.6f}")  # 약 0.01 (smoothing 기인)
```

### 손실값 정상 범위

```
초기 에포크: loss ≈ log(num_classes) ≈ 3.14 (23개 클래스)
중간 에포크: loss ≈ 1.0 ~ 2.0
최종 에포크: loss ≈ 0.1 ~ 0.5 (좋은 훈련)
```

---

## 🎯 학습 포인트

- ✅ **CrossEntropyLoss**: 분류 분제의 표준 손실 함수
- ✅ **Label Smoothing**: 과신 방지 + 일반화 성능 향상
- ✅ **ε=0.01**: 프로젝트의 권장 smoothing 계수
- ✅ **계층 구조 활용**: 향후 개선 아이디어
- ✅ **안정적인 훈련**: Loss curve 모니터링 필수
