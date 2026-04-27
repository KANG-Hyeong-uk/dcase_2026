# DCASE 2026 Task 1 — 베이스라인 완전 학습 가이드

> AI에 대한 사전 지식이 없어도 이해할 수 있도록 처음부터 설명합니다.

---

## 목차

1. [이 대회가 뭘 하는 건가요?](#1-이-대회가-뭘-하는-건가요)
2. [AI 학습이란 무엇인가요? (기초)](#2-ai-학습이란-무엇인가요-기초)
3. [왜 오디오 파일이 아닌 임베딩을 쓰나요?](#3-왜-오디오-파일이-아닌-임베딩을-쓰나요-핵심-질문)
4. [CLAP 모델이란 무엇인가요?](#4-clap-모델이란-무엇인가요)
5. [데이터셋 구조](#5-데이터셋-구조)
6. [베이스라인 전체 파이프라인](#6-베이스라인-전체-파이프라인)
7. [모델 구조 상세 설명](#7-모델-구조-상세-설명)
8. [학습 과정 단계별 설명](#8-학습-과정-단계별-설명)
9. [코드 흐름 전체 요약](#9-코드-흐름-전체-요약)
10. [이 접근법이 올바른가요?](#10-이-접근법이-올바른가요)
11. [성능을 높이려면 어떻게 해야 하나요?](#11-성능을-높이려면-어떻게-해야-하나요)

---

## 1. 이 대회가 뭘 하는 건가요?

**DCASE 2026 Task 1 — Heterogeneous Audio Classification**

- **목표**: 다양한 출처(이종, Heterogeneous)에서 온 소리 파일을 듣고, "이 소리는 무슨 소리인가?"를 분류하는 AI를 만드는 것
- **분류 대상 예시**: 음악, 악기 샘플, 음성(말소리), 효과음, 자연 소리 등 23개 세부 카테고리
- **데이터**: [Freesound](https://freesound.org)에서 수집한 수천 개의 오디오 클립

**클래스 계층 구조 (2단계)**

```
상위(top) 클래스       세부(sub) 클래스
─────────────────────────────────────
Music (m)         →  m-sp, m-si, m-m, m-other
Instrument (is)   →  is-p, is-s, is-w, is-k, is-e, is-other
Speech (sp)       →  sp-s, sp-c, sp-p, sp-other
FX (fx)           →  fx-o, fx-m, fx-h, fx-other
Soundscape (ss)   →  ss-n, ss-u, ss-other
```

예를 들어 `is-w`는 "Instrument Sample / Wind" → 플루트, 색소폰 같은 관악기 샘플입니다.

---

## 2. AI 학습이란 무엇인가요? (기초)

### 2-1. 분류 문제란?

컴퓨터에게 "이 소리를 듣고 어느 카테고리인지 맞혀봐"라고 시키는 것입니다.

```
입력: 소리 데이터
  ↓
[AI 모델]
  ↓
출력: "이건 관악기 샘플(is-w)일 확률 95%, 현악기(is-s)일 확률 3%, ..."
```

### 2-2. 학습(Training)이란?

AI는 처음에 아무것도 모릅니다. 다음 과정을 수만 번 반복하며 "공부"합니다:

```
1. 데이터 입력 → 모델이 예측
2. 정답과 비교 → 얼마나 틀렸는지 계산 (= Loss)
3. 틀린 만큼 모델의 파라미터(가중치)를 조금 수정
4. 반복 → 점점 정확해짐
```

**파라미터(가중치)**란? 모델 내부의 수많은 숫자들입니다. 이 숫자들이 학습을 통해 조정되면서 AI가 똑똑해집니다. 이 베이스라인 모델은 약 **수십만 개**의 파라미터를 가집니다.

### 2-3. Epoch, Batch, Learning Rate

| 용어 | 의미 | 이 베이스라인 설정 |
|------|------|------------------|
| **Epoch** | 전체 데이터를 한 바퀴 다 보는 것 | 100 epoch |
| **Batch** | 한 번에 처리하는 데이터 묶음 크기 | 64개 |
| **Learning Rate (lr)** | 한 번 수정할 때 얼마나 크게 바꿀지 | 0.001 |

---

## 3. 왜 오디오 파일이 아닌 임베딩을 쓰나요? (핵심 질문)

이것이 가장 중요한 질문입니다.

### 3-1. 오디오 파일을 직접 쓰지 않는 이유

오디오 파일(.wav, .mp3 등)은 "원시 파형" — 공기의 진동을 숫자로 기록한 것입니다.

```
원시 파형: [-0.0012, 0.0034, -0.0008, 0.0021, ...]  (1초에 44,100개 숫자!)
```

이 데이터를 AI에 직접 넣으면:
- 데이터가 너무 길고 복잡해서 학습이 느림
- 어떤 부분이 중요한지 AI가 스스로 찾아야 함 → 더 많은 데이터와 시간 필요
- 노이즈, 볼륨 차이 등에 취약

### 3-2. 임베딩(Embedding)이란?

임베딩은 "소리의 핵심 특징을 뽑아낸 숫자 벡터"입니다.

```
원시 오디오 (수만 개 숫자)
      ↓ [CLAP 같은 사전학습 모델로 처리]
임베딩 (512개 숫자)  ← 소리의 의미가 압축되어 있음!
```

예를 들어:
- 플루트 소리의 임베딩: `[0.12, -0.34, 0.87, ...]` (512차원)
- 색소폰 소리의 임베딩: `[0.15, -0.31, 0.85, ...]` (비슷한 값 → 둘 다 관악기)
- 개 짖는 소리의 임베딩: `[-0.78, 0.23, -0.45, ...]` (완전히 다른 값)

**핵심**: 비슷한 소리는 임베딩 공간에서도 가까이 위치합니다.

### 3-3. 임베딩을 미리 뽑아두는 이유 (Precomputed Embeddings)

```
방법 A (직접 학습):  오디오 → [CLAP 전체 모델] → 임베딩 → [분류기]
                    매 epoch마다 CLAP 전체를 돌림 → 매우 느림, GPU 메모리 많이 필요

방법 B (베이스라인): 미리 임베딩 계산 → .npy 파일로 저장
                    학습 시: .npy 파일 로드 → [분류기만 학습]
                    → 훨씬 빠름! 작은 GPU로도 가능!
```

**베이스라인이 `.npy` 파일만 쓰는 이유**: CLAP 모델을 매번 돌리면 RTX 3060으로는 감당이 안 됩니다. 임베딩을 한 번 뽑아두면 이후 학습은 가볍게 할 수 있습니다.

---

## 4. CLAP 모델이란 무엇인가요?

**CLAP (Contrastive Language-Audio Pretraining)**

OpenAI의 CLIP(이미지-텍스트 대조학습)을 오디오에 적용한 모델입니다.

### 4-1. CLAP이 하는 일

```
오디오 "개 짖는 소리" ──→ [CLAP Audio Encoder] ──→ 512차원 벡터
텍스트 "A dog barking"  ──→ [CLAP Text Encoder]  ──→ 512차원 벡터
                                                       ↑
                                          두 벡터가 같은 공간에 위치!
```

CLAP은 수백만 개의 (오디오, 텍스트 설명) 쌍으로 학습된 모델입니다:
- "개 짖는 소리" 오디오 ↔ "A dog barking" 텍스트 → 같은 공간
- "피아노 연주" 오디오 ↔ "Piano playing" 텍스트 → 같은 공간

### 4-2. 왜 텍스트 임베딩도 사용하나요?

Freesound 데이터에는 각 오디오마다 **텍스트 설명**(제목, 태그, 설명)이 있습니다:

```
sound_id: 185755
오디오: 전자레인지 소리
텍스트: "microwave oven operating, door closing, typical bell sound"
```

CLAP으로 텍스트도 임베딩하면:
- 오디오 임베딩 (소리 자체의 특징) + 텍스트 임베딩 (설명 정보) = 더 풍부한 정보

**즉, 이 베이스라인은 두 종류의 정보를 함께 씁니다:**
1. `clap_audio_embeddings/` — 소리를 들었을 때 뽑은 특징
2. `clap_text_embeddings/` — 소리에 대한 텍스트 설명에서 뽑은 특징

---

## 5. 데이터셋 구조

### 5-1. 파일 구조

```
data/
├── metadata/
│   ├── BSD10k_metadata.csv    ← 각 오디오의 정보 (클래스, 신뢰도, 업로더 등)
│   └── BST_description.csv    ← 클래스 이름과 설명
└── features/
    ├── clap_audio_embeddings/
    │   ├── 185755.npy         ← sound_id 185755번 오디오의 임베딩 (512차원)
    │   ├── 358405.npy
    │   └── ...
    └── clap_text_embeddings/
        ├── 185755.npy         ← 같은 오디오의 텍스트 설명 임베딩
        └── ...
```

### 5-2. BSD10k_metadata.csv 구조

```
sound_id | class | class_idx | class_top | confidence | ...
185755   | fx-o  | 11        | fx        | 4          | ...
358405   | is-w  | 5         | is        | 5          | ...
```

- `sound_id`: 오디오 파일의 ID (Freesound 번호)
- `class`: 세부 클래스 (예: fx-o = 효과음/물체)
- `class_idx`: 클래스 번호 (숫자)
- `class_top`: 상위 클래스 (예: fx)
- `confidence`: 라벨 신뢰도 (1~5, 5가 가장 확실)

### 5-3. `.npy` 파일이란?

NumPy 배열을 저장한 파일입니다. 각 파일은 `[512]` 크기의 숫자 배열:

```python
import numpy as np
emb = np.load("185755.npy")  # shape: (512,)
# [0.123, -0.456, 0.789, ...]  ← 512개 숫자
```

---

## 6. 베이스라인 전체 파이프라인

```
[데이터 준비 단계]
BSD10k_metadata.csv + .npy 파일들
         ↓
   build_dataset.py 실행
         ↓
  processed_dataset.csv 생성
  (각 샘플의 npy 경로 + 클래스 레이블)

[학습 단계]
  processed_dataset.csv 읽기
         ↓
  5-Fold 교차검증으로 분할
  (Train 64% / Val 16% / Test 20%)
         ↓
  HATRDataset → DataLoader
  (배치 단위로 .npy 로드)
         ↓
  BaseClassifier 모델에 입력
  (audio_emb 512차원 + text_emb 512차원)
         ↓
  Loss 계산 (CrossEntropy)
  ↓
  역전파(Backpropagation)로 파라미터 업데이트
         ↓
  100 epoch 반복 → 가장 좋은 모델 저장

[평가 단계]
  best_model.pth 로드
         ↓
  Test set으로 정확도 측정
  (세부 클래스 정확도 + 상위 클래스 정확도)
```

---

## 7. 모델 구조 상세 설명

[models.py](dcase2026_task1_baseline/models.py)에 정의된 `BaseClassifier`의 구조입니다.

### 7-1. 전체 구조도

```
audio_emb (512차원)   text_emb (512차원)
       ↓                      ↓
[EmbeddingEncoder]    [EmbeddingEncoder]
  (512 → 128차원)       (512 → 128차원)
       ↓                      ↓
         [AttentionFusion]
         (128 + 128 → 128차원)
         어떤 정보를 더 믿을지 가중치 결정
                ↓
        [latent_projector]
        (128 → 256 → 128 → 64차원)
                ↓
        [residual_classifier]
        (64 → 64차원, 2번 반복)
                ↓
        [class_predictor]
        (64 → 23차원)
                ↓
        최종 예측: 23개 클래스 중 하나
```

### 7-2. EmbeddingEncoder 상세

512차원 임베딩을 128차원으로 압축하면서 중요한 특징을 추출합니다.

```
입력 (512차원)
  ↓ BatchNorm (정규화)
  ↓ Linear (512 → 1024)
  ↓ LeakyReLU (활성화 함수)
  ↓ Dropout (과적합 방지)
  ↓ [ResidualBlock × 3] (특징 정제)
  ↓ Linear (1024 → 512 → 128)
  ↓ BatchNorm
출력 (128차원)
```

**ResidualBlock이란?**: 입력을 변환한 결과에 원래 입력을 더하는 구조.
잔차(Residual) 연결을 통해 정보 손실을 방지합니다.
```
x ──→ [Linear → BN → ReLU → Dropout → Linear → BN] ──→ + ──→ ReLU
│                                                        ↑
└────────────────── (skip connection) ──────────────────┘
```

### 7-3. AttentionFusion 상세

오디오 정보와 텍스트 정보 중 어느 것을 더 신뢰할지 동적으로 결정합니다.

```python
# 두 특징을 합쳐서 (128 + 128 = 256차원)
combined = [audio_features, text_features]  # 256차원

# 가중치 계산
attention_weights = Softmax(Linear(256 → 128 → 2))
# 결과: [α_audio, α_text]  (합이 1이 되도록)
# 예: [0.7, 0.3] → 오디오를 70%, 텍스트를 30% 신뢰

# 가중 합산
fused = audio_features × α_audio + text_features × α_text
```

이 가중치는 학습 중에 자동으로 최적화됩니다!

### 7-4. 모드(mode) 설정

베이스라인은 두 가지 모드로 실험합니다:

| 모드 | 사용 입력 | 설명 |
|------|-----------|------|
| `both` | audio + text | 오디오와 텍스트 임베딩 모두 사용 |
| `audio` | audio만 | 오디오 임베딩만 사용 (추론 시 텍스트 없을 때 대비) |

---

## 8. 학습 과정 단계별 설명

### 8-1. K-Fold 교차검증 (K=5)

한 번만 학습하면 운이 좋거나 나쁠 수 있어서, 5번 나눠서 실험합니다:

```
전체 데이터 (100%)
├── Fold 0: [Test 20%] [Train+Val 80%]
├── Fold 1: [Test 20%] [Train+Val 80%]
├── Fold 2: [Test 20%] [Train+Val 80%]
├── Fold 3: [Test 20%] [Train+Val 80%]
└── Fold 4: [Test 20%] [Train+Val 80%]

각 폴드에서:
  Train+Val 80% → Train 64% / Val 16% 로 추가 분할
  (StratifiedShuffleSplit 사용)
```

**Stratified**란? 각 클래스의 비율이 분할 후에도 유지되도록 함.
→ 희귀 클래스도 모든 폴드에 골고루 들어가게 됨.

코드에서의 위치: [train_test.py:241-258](dcase2026_task1_baseline/train_test.py#L241-L258)

### 8-2. 데이터 증강 (Data Augmentation)

훈련 데이터에만 적용하는 변형으로, 모델이 더 강건해지게 합니다:

```python
# 1. 가우시안 노이즈 추가 (아주 작은 랜덤 변화)
emb = emb + torch.randn_like(emb) * 0.0001

# 2. 랜덤 마스킹 (일부 차원을 0으로)
# mask_pct=0.7 → 최대 70%까지 랜덤하게 0으로 만듦
emb = _rand_mask(emb)
```

왜 하나요? "이 임베딩의 일부가 없어도 정답을 맞힐 수 있어야 해"라고 학습시켜서 더 견고한 모델을 만들기 위해서입니다.

코드: [dataset_utils.py:20-52](dcase2026_task1_baseline/dataset_utils.py#L20-L52)

### 8-3. Loss 함수 (CrossEntropy)

모델의 예측이 얼마나 틀렸는지 측정하는 함수입니다.

```
모델 출력 (확률): [0.05, 0.02, 0.85, 0.03, 0.05, ...]  (23개 클래스)
                                 ↑
                          (is-w 클래스)
정답:            [  0,    0,   1,    0,    0, ...]  (is-w가 정답)

CrossEntropy = -log(0.85) ≈ 0.16  ← 작을수록 좋음
CrossEntropy = -log(0.01) ≈ 4.60  ← 완전히 틀렸을 때
```

### 8-4. Optimizer (Adam)

Loss를 줄이는 방향으로 파라미터를 업데이트하는 알고리즘입니다.

```python
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

- `lr=0.001`: 한 번에 파라미터를 0.001만큼 조정
- `weight_decay=1e-5`: 파라미터가 너무 커지지 않도록 제약 (정규화)

### 8-5. Learning Rate Scheduler

학습이 진행될수록 학습률을 점점 낮춥니다:

```python
# StepLR: 20 epoch마다 lr을 절반으로
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# 변화: 0.001 → 0.0005 (20epoch) → 0.00025 (40epoch) → ...
```

왜 낮추나요? 초반엔 크게 조정해서 빠르게 학습하고, 후반엔 작게 조정해서 정밀하게 조정합니다.

### 8-6. Early Stopping

더 이상 성능이 오르지 않으면 학습을 조기 종료합니다:

```python
patience = 5             # 5 epoch 기다림
early_stopping_factor = 3  # patience × factor = 15 epoch 동안 개선 없으면 종료
```

### 8-7. 학습 1 epoch의 흐름

```
for data in train_loader:  ← 배치 단위로 데이터 로드
    1. audio_emb, text_emb 로드
    2. model(audio_emb, text_emb) → 예측값
    3. loss = CrossEntropy(예측값, 정답)
    4. loss.backward() ← 역전파: 각 파라미터의 gradient 계산
    5. optimizer.step() ← 파라미터 업데이트
    6. optimizer.zero_grad() ← gradient 초기화

→ 다음 배치로 반복
→ 모든 배치가 끝나면 1 epoch 완료
→ Validation set으로 정확도 측정
→ 최고 정확도이면 모델 저장
```

---

## 9. 코드 흐름 전체 요약

### Step 1: build_dataset.py 실행

```
BSD10k_metadata.csv 읽기
  ↓
각 sound_id에 대해:
  - audio_emb_filepath 확인 (없으면 스킵)
  - text_emb_filepath 확인 (없으면 스킵)
  ↓
processed_dataset.csv 저장:
  index | audio_emb_filepath | text_emb_filepath | class | class_idx | top_class | top_class_idx
  28570 | .../28570.npy      | .../28570.npy     | is-w  | 5         | is        | 2
  ...
```

코드: [build_dataset.py](dcase2026_task1_baseline/build_dataset.py)

### Step 2: train_test.py 실행

```python
# 1. 데이터 로드
full_df = pd.read_csv("processed_dataset.csv")

# 2. 5-Fold 분할
for fold in range(5):
    train_df, val_df, test_df = split(full_df)
    
    # 3. Dataset & DataLoader 생성
    train_loader = DataLoader(HATRDataset(train_df, aug=True), batch_size=64)
    
    # 4. 모델 생성
    model = BaseClassifier(hidden=128, num_classes=23, ...)
    
    # 5. 학습
    for epoch in range(100):
        for batch in train_loader:
            # 예측 → Loss → 역전파 → 업데이트
        # Validation 정확도 측정
        # 최고이면 저장
    
    # 6. 테스트
    metrics = evaluate_model(best_model, test_loader)
```

---

## 10. 이 접근법이 올바른가요?

**결론: 매우 올바른 접근법입니다.** 그 이유를 설명합니다.

### 10-1. 왜 오디오 파일을 직접 안 쓰나요? (다시 정리)

| 비교 항목 | 직접 오디오 학습 | 임베딩 학습 (베이스라인) |
|-----------|-----------------|------------------------|
| GPU 메모리 | 수십 GB 필요 | 몇 GB로 충분 |
| 학습 시간 | 며칠~주 | 30분~1시간 |
| 필요 데이터량 | 수십만 개 이상 | 수천 개로 가능 |
| 성능 | 데이터가 충분하면 높을 수 있음 | 충분히 경쟁력 있음 |

### 10-2. 이미 존재하는 강력한 모델을 활용

CLAP은 이미 수백만 개 데이터로 학습된 강력한 모델입니다. 이것을 특징 추출기로 쓰고, 그 위에 작은 분류기만 학습하는 방식은 **Transfer Learning (전이학습)**의 전형적인 패턴입니다:

```
[CLAP - 수백만 개로 이미 학습됨, 고정] → [작은 분류기 - 우리가 학습]
```

이 방식은 컴퓨터 비전, NLP 등 모든 AI 분야에서 표준으로 사용됩니다.

### 10-3. 텍스트 임베딩을 추가하는 것도 올바릅니다

Freesound의 텍스트 태그/설명은 매우 풍부한 정보를 담고 있습니다. 이것을 버리는 것은 손해입니다. 베이스라인은 두 정보를 Attention으로 결합해서 시너지를 냅니다.

---

## 11. 성능을 높이려면 어떻게 해야 하나요?

베이스라인 위에서 시도할 수 있는 개선 방향입니다:

### 11-1. 데이터 관련

```python
# 1. 신뢰도 필터링 (현재 주석처리됨)
# train_test.py:226-230 의 주석을 해제하고
# confidence >= 4 인 데이터만 사용
full_df = full_df[full_df['confidence'] >= 4]

# 2. 더 큰 데이터셋 사용
ACTIVE_DATASET = 'BSD35k-CS'  # BSD10k 대신 BSD35k 사용
```

### 11-2. 모델 관련

```python
# hidden_size 키우기 (모델 용량 증가)
model = BaseClassifier(hidden_size=256, ...)  # 128 → 256

# ResidualBlock 더 많이 쌓기
model = BaseClassifier(num_residual_blocks=5, ...)  # 3 → 5
```

### 11-3. 학습 관련

```python
# Learning rate 조정
lr = 0.0005  # 0.001 → 0.0005

# Cosine Annealing Scheduler 사용 (더 부드러운 lr 감소)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

### 11-4. 앙상블

5-fold의 5개 모델을 모두 사용해서 다수결/평균으로 최종 예측:

```python
# 각 fold의 모델이 예측한 확률을 평균
final_prob = (prob_fold0 + prob_fold1 + prob_fold2 + prob_fold3 + prob_fold4) / 5
prediction = argmax(final_prob)
```

---

## 핵심 정리 (한 문장씩)

1. **임베딩을 쓰는 이유**: CLAP이 이미 잘 추출한 512차원 특징을 그대로 활용해서 빠르고 효율적으로 학습
2. **텍스트 임베딩을 같이 쓰는 이유**: 오디오 소리 + 텍스트 설명 두 정보를 합치면 더 정확한 분류 가능
3. **K-Fold를 쓰는 이유**: 어떤 데이터로 나누느냐에 따라 결과가 달라지는 것을 방지, 신뢰할 수 있는 평균 성능 측정
4. **Attention Fusion을 쓰는 이유**: 오디오와 텍스트 정보의 중요도를 데이터에 따라 동적으로 조절
5. **이 접근법이 옳은 이유**: Transfer Learning + Multimodal Fusion = 현재 AI 연구의 표준적이고 강력한 방법

---

*작성일: 2026-04-14 | 베이스라인 코드: github.com/MTG/dcase2026_task1_baseline*
