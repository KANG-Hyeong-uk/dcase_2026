# models.py - 신경망 모델 완벽 가이드

## 📋 파일 개요
`models.py`는 멀티모달 음향 분류 모델의 **핵심 신경망 아키텍처**를 정의합니다.
잔차 블록, 임베딩 인코더, 주의메커니즘, 분류기로 구성됩니다.

---

## 🏗️ 전체 모델 구조

```
음성 임베딩 (512)         텍스트 임베딩 (512)
     ↓                        ↓
┌────────────────┐      ┌────────────────┐
│ EmbeddingEnc   │      │ EmbeddingEnc   │
│  (512 → 256)   │      │  (512 → 256)   │
└────────────────┘      └────────────────┘
     ↓                        ↓
┌────────────────────────────────────────┐
│     AttentionFusion (주의 가중치)        │
│  (동적으로 음성/텍스트 비율 학습)        │
└────────────────────────────────────────┘
     ↓
┌────────────────┐
│ LatentProj     │ (특성 추출)
│  (256 → 128)   │
└────────────────┘
     ↓
┌────────────────┐
│ ResidualBlock  │ (비선형 변환)
│  ×2 중첩        │
└────────────────┘
     ↓
┌────────────────┐
│ ClassPredictor │ (최종 분류)
│  (128 → 23)    │
└────────────────┘
     ↓
클래스 확률 (23가지)
```

---

## 🔍 각 컴포넌트 상세 분석

### 1️⃣ ResidualBlock - 잔차 블록

```python
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2, use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(hidden_size)
            self.norm2 = nn.BatchNorm1d(input_size)
```

**구조 다이어그램**:
```
입력 x (256D)
 ↓
├─→ 선형층 1 (256→512)
    ↓
    배치 정규화 (선택사항)
    ↓
    LeakyReLU (α=0.01)
    ↓
    드롭아웃 (p=0.2)
    ↓
    선형층 2 (512→256)  ←┐
    ↓                  │
    배치 정규화         │
    ↓                  │
기본 경로 (x) →ㅤ─────┤
                      ↓
                    덧셈 (잔차 연결)
                    ↓
                    LeakyReLU
                    ↓
                   출력 (256D)
```

**Forward 함수**:
```python
def forward(self, x):
    residual = x  # 원본 입력 저장 (잔차)
    
    out = self.linear1(x)         # 256 → 512
    if self.use_batch_norm:
        out = self.norm1(out)     # 정규화
    out = self.activation(out)    # 비선형 활성화
    out = self.dropout(out)       # 정규화 (과적합 방지)
    
    out = self.linear2(out)       # 512 → 256
    if self.use_batch_norm:
        out = self.norm2(out)
    
    out += residual               # ★ 잔차 연결 (핵심!)
    out = self.activation(out)
    
    return out
```

**핵심 개념 - 잔차 연결 (Residual Connection)**:
```
문제점: 깊은 신경망은 gradient vanishing 발생
해결책: f(x) + x 형태로 직접 경로 제공

수학: y = f(x) + x (원본을 직접 더함)
장점:
- 더 깊은 네트워크 가능
- 학습 안정성 증가
- 초기 에포크에 빠른 수렴
```

**LeakyReLU vs ReLU**:
```python
ReLU(x) = max(0, x)           # x < 0일 때 항상 0 (dead ReLU)
LeakyReLU(x) = max(0.01x, x)  # x < 0일 때도 작은 기울기 통과
```

---

### 2️⃣ EmbeddingEncoder - 임베딩 인코더

```python
class EmbeddingEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2, 
                 use_batch_norm=True, num_residual_blocks=3):
        super().__init__()
        
        hidden_size = max(input_size, output_size * 2)
        # 예: input=512, output=256 → hidden=512
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, hidden_size * 2, dropout, use_batch_norm)
            for _ in range(num_residual_blocks)
        ])
        # 3개의 잔차 블록 (512→512)
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
```

**구조**:
```
입력 임베딩 (512D)
    ↓
┌─────────────────────────┐
│ 입력 프로젝션            │ (512 → 512)
│ Linear + LeakyReLU      │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 잔차 블록 ×3             │ (512 → 512)
│ (점진적 특성 학습)      │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 출력 프로젝션            │ (512 → 256)
│ Linear + LeakyReLU      │
└─────────────────────────┘
    ↓
→ 고차원 특성 벡터 (256D)
```

**Forward 함수**:
```python
def forward(self, x):
    if self.use_batch_norm:
        x = self.input_norm(x)
    
    # 입력 프로젝션: 임베딩 차원 조정
    x = self.input_projection(x)  # 512 → 512
    
    # 잔차 블록: 깊은 비선형 특성 추출
    for block in self.residual_blocks:
        x = block(x)  # 512 → 512
    
    # 출력 프로젝션: 최종 차원 감소
    x = self.output_projection(x)  # 512 → 256
    
    if self.use_batch_norm:
        x = self.output_norm(x)
    
    return x  # (배치크기, 256)
```

---

### 3️⃣ AttentionFusion - 주의 기반 융합

```python
class AttentionFusion(nn.Module):
    def __init__(self, feature_size, dropout=0.2):
        super().__init__()
        
        # 주의 점수 계산 네트워크
        self.attention = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),  # 512 → 256
            nn.Tanh(),                                   # 비선형 활성화
            nn.Linear(feature_size, 2),                 # 256 → 2 (음성, 텍스트 스코어)
            nn.Softmax(dim=-1)                          # 확률로 정규화 [0, 1]
        )
        self.dropout = nn.Dropout(dropout)
```

**구조 및 동작**:
```
음성 특성 (256D)     텍스트 특성 (256D)
     ↓                    ↓
  ┌──────────────────────────────┐
  │ 연결 (Concatenate)            │ 512D
  └──────────────────────────────┘
         ↓
  ┌──────────────────────────────┐
  │ 선형층 (512 → 256)            │
  │ Tanh 활성화                   │
  │ 선형층 (256 → 2)             │
  │ Softmax (정규화)              │
  └──────────────────────────────┘
         ↓
  주의 가중치: [w_audio, w_text]
  예: [0.7, 0.3]  (음성 70%, 텍스트 30%)
```

**Forward 함수**:
```python
def forward(self, audio_features, text_features):
    # 1. 음성과 텍스트 연결
    combined = torch.cat([audio_features, text_features], dim=-1)  # (배치, 512)
    
    # 2. 주의 점수 계산
    attention_weights = self.attention(combined)  # (배치, 2)
    # 예: [[0.7, 0.3], [0.6, 0.4], ...]
    
    # 3. 가중 합산
    weighted_audio = audio_features * attention_weights[:, 0:1]  # 첫 번째 가중치
    weighted_text = text_features * attention_weights[:, 1:2]    # 두 번째 가중치
    
    fused = weighted_audio + weighted_text  # 융합
    
    return self.dropout(fused), attention_weights
```

**핵심 개념**:
```
멀티모달 학습의 문제:
- 음성과 텍스트를 같은 비율로 섞으면 최적이 아닐 수 있음
- 어떤 태스크에서는 음성이 중요, 다른 크는 텍스트가 중요

주의 해결책:
- 각 샘플마다 최적의 음성/텍스트 비율을 학습
- 동적으로 가중치 조정 (훈련 중)

예:
음성이 깨끗한 경우: w = [0.8, 0.2] (음성 우선)
배경음이 많은 경우: w = [0.3, 0.7] (텍스트 우선)
```

---

### 4️⃣ BaseClassifier - 메인 분류 모델

```python
class BaseClassifier(nn.Module):
    def __init__(self, hidden_size=256, num_classes=10, 
                 emb_size_audio=0, emb_size_text=0, 
                 dropout=0.2, use_batch_norm=True, 
                 mode="both", num_residual_blocks=3, 
                 use_attention_fusion=True):
        super().__init__()
```

**파라미터 설명**:

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| **hidden_size** | 256 | 임베딩 인코더 출력 차원 |
| **num_classes** | 10 | 분류할 클래스 수 (23) |
| **emb_size_audio** | 0 | 입력 음성 임베딩 차원 (512) |
| **emb_size_text** | 0 | 입력 텍스트 임베딩 차원 (512) |
| **dropout** | 0.2 | 드롭아웃 비율 |
| **use_batch_norm** | True | 배치 정규화 사용 |
| **mode** | "both" | "audio", "text", "both" 중 선택 |
| **num_residual_blocks** | 3 | 잔차 블록 개수 |
| **use_attention_fusion** | True | 주의 기반 융합 사용 |

---

#### 모드별 모델 구성

**mode="both" (멀티모달, 기본값)**:
```
음성 임베딩          텍스트 임베딩
(512D)             (512D)
  ↓                  ↓
음성 인코더      텍스트 인코더
(512 → 256)     (512 → 256)
  ↓                  ↓
┌─────────────────────────┐
│ AttentionFusion         │
│ (1개 추가 스트림)       │
└─────────────────────────┘
  ↓
분류기 (→ 23 클래스)
```

**mode="audio" (음성만)**:
```
음성 임베딩
(512D)
  ↓
음성 인코더 (512 → 256)
  ↓
분류기 (→ 23 클래스)
```

**mode="text" (텍스트만)**:
```
텍스트 임베딩
(512D)
  ↓
텍스트 인코더 (512 → 256)
  ↓
분류기 (→ 23 클래스)
```

---

#### 생성자의 인코더 생성

```python
if self.mode in ["audio", "both"]:
    self.audio_emb_extractor = EmbeddingEncoder(
        input_size=emb_size_audio,      # 512
        output_size=hidden_size,        # 256
        ...
    )

if self.mode in ["text", "both"]:
    self.text_emb_extractor = EmbeddingEncoder(
        input_size=emb_size_text,       # 512
        output_size=hidden_size,        # 256
        ...
    )
```

---

#### 결합 방식 선택

```python
if self.mode == "both":
    if self.use_attention_fusion:
        combined_size = hidden_size  # 256
        self.fusion = AttentionFusion(hidden_size, dropout)
    else:
        combined_size = hidden_size * 2  # 512 (단순 연결)
else:
    combined_size = hidden_size  # 256
```

**두 가지 방식 비교**:

| 방식 | 코드 | 차원 | 장점 |
|------|------|------|------|
| **Attention Fusion** | 권장 | 256D | 동적 가중치, 학습 유연성 |
| **Concatenation** | 대안 | 512D | 모든 정보 보존, 계산 간단 |

---

#### 분류기 구성

```python
self.latent_projector = nn.Sequential(
    nn.Linear(combined_size, hidden_size * 2),  # 256 → 512
    nn.LeakyReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size * 2, hidden_size),    # 512 → 256
    nn.LeakyReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, hidden_size // 2),   # 256 → 128
    nn.LeakyReLU(),
    nn.Dropout(dropout / 2)
)
# 잠재 공간 표현 생성 (128D)

self.residual_classifier = nn.ModuleList([
    ResidualBlock(hidden_size // 2, hidden_size, ...)
    for _ in range(2)
])
# 2개의 잔차 블록으로 복잡한 의사결정 경계 학습

self.class_predictor = nn.Sequential(
    nn.Linear(hidden_size // 2, hidden_size // 4),  # 128 → 64
    nn.LeakyReLU(),
    nn.Dropout(dropout / 4),
    nn.Linear(hidden_size // 4, num_classes)        # 64 → 23
)
# 최종 클래스 로짓
```

---

#### Forward 함수

```python
def forward(self, audio_emb=None, text_emb=None):
    features = []
    
    # 1. 각 모드별 인코딩
    if self.mode in ["audio", "both"]:
        audio_features = self.audio_emb_extractor(audio_emb)
        features.append(audio_features)  # (배치, 256)
    
    if self.mode in ["text", "both"]:
        text_features = self.text_emb_extractor(text_emb)
        features.append(text_features)   # (배치, 256)
    
    # 2. 특성 융합
    if len(features) > 1:
        if self.use_attention_fusion:
            combined_features, attn_scores = self.fusion(
                features[0], features[1]
            )  # (배치, 256) + 주의 가중치
        else:
            combined_features = torch.cat(features, dim=-1)
            # (배치, 512)
            attn_scores = None
    else:
        combined_features = features[0]  # (배치, 256)
        attn_scores = None
    
    # 3. 잠재 표현 추출
    z = self.latent_projector(combined_features)  # (배치, 128)
    
    # 4. 잔차 블록을 통한 깊은 처리
    for block in self.residual_classifier:
        z = block(z)  # (배치, 128)
    
    # 5. 최종 분류
    class_logit = self.class_predictor(z)  # (배치, 23)
    
    return z, class_logit, attn_scores
```

**반환값**:
```python
z: (배치크기, 128)               # 잠재 표현
class_logit: (배치크기, 23)      # 로짓 (분류 전 점수)
attn_scores: (배치크기, 2) 또는 None  # 주의 가중치
```

---

## 🧪 모델 테스트 코드

```python
if __name__ == "__main__":
    # 모델 생성
    model = BaseClassifier(
        hidden_size=128, 
        num_classes=23, 
        emb_size_audio=512, 
        emb_size_text=512, 
        dropout=0.2, 
        use_batch_norm=False, 
        mode="both"
    )
    
    # 더미 입력
    audio = torch.randn(1, 512)  # 배치크기=1, 임베딩 차원=512
    text = torch.randn(1, 512)
    
    # Forward pass
    z, class_logit, attn_scores = model(audio_emb=audio, text_emb=text)
    
    # 출력 확인
    print("Latent representation shape:", z.shape)  # (1, 128)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Attention scores:", attn_scores)  # [[0.6, 0.4]] 같은 형태
```

---

## 📊 전체 모델 파라미터 수 계산

```
입력층:
- 음성 인코더: 512 → 256 (약 130K)
- 텍스트 인코더: 512 → 256 (약 130K)

융합층:
- AttentionFusion: (약 20K)

분류기:
- 잠재 프로젝터: (약 150K)
- 잔차 블록 ×2: (약 100K)
- 클래스 예측기: (약 5K)

총합: 약 535K 파라미터
```

---

## 🎯 학습 포인트

- ✅ **잔차 연결**: Gradient flow 개선 + 깊은 네트워크 가능
- ✅ **배치 정규화**: 학습 안정성 + 수렴 속도 증가
- ✅ **멀티모달**: 음성과 텍스트를 동시에 활용
- ✅ **주의 메커니즘**: 각 샘플마다 최적의 모달리티 가중치 학습
- ✅ **모드 선택**: 단일 모달 또는 멀티모달 실험 가능
