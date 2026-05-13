# dataset_utils.py - 데이터 로더 완벽 가이드

## 📋 파일 개요
`dataset_utils.py`는 **PyTorch Dataset 클래스**인 `HATRDataset`을 정의합니다.
임베딩 로드, 데이터 증강, 계층적 레이블 처리를 담당합니다.

---

## 🔍 HATRDataset 클래스 상세 분석

### 클래스 구조
```python
class HATRDataset(Dataset):
    """
    Dataset for precomputed multimodal (audio + text) embeddings.
    Hierarchical labels (2-level, use only parent and leafs if more).
    Augmentation (optional): Gaussian noise + random zeroing on embeddings.
    CSV columns are hardcoded for BST datasets, change if necessary.
    """
```

**특징**:
- ✅ 멀티모달 (음성 + 텍스트)
- ✅ 계층적 레이블 (2단계)
- ✅ 온더플라이 증강
- ✅ 사전 계산된 임베딩 사용 (실시간 추출 안 함)

---

### 1️⃣ 생성자 (__init__)

```python
def __init__(self, dataframe, aug=True, mask_pct=0.2):
    self.dataframe = dataframe    # processed_dataset.csv의 결과물
    self.aug = aug                # 증강 활성화 여부
    self.mask_pct = mask_pct      # 마스킹 비율 (0~1)
```

**파라미터**:

| 파라미터 | 타입 | 설명 | 예시 |
|---------|------|------|------|
| **dataframe** | pd.DataFrame | processed_dataset.csv 로드 | df (9876행 × 8컬럼) |
| **aug** | bool | 데이터 증강 활성화 | True |
| **mask_pct** | float | 마스킹할 임베딩 비율 | 0.2 (20%) |

---

### 2️⃣ 마스킹 함수 (_rand_mask)

```python
def _rand_mask(self, emb):
    max_mask = int(emb.shape[0] * self.mask_pct)
    # 마스킹할 최대 개수 계산
    # emb.shape[0]: 임베딩 차원 (보통 512)
    # 예: 512 * 0.2 = 102.4 → 102
    
    num_to_mask = random.randint(1, max_mask)
    # 1~102 사이 임의의 개수 선택 (매번 다름)
    
    mask_indices = torch.randperm(emb.shape[0])[:num_to_mask]
    # 0~511 중 임의로 102개 인덱스 선택
    
    mask = torch.ones_like(emb)
    # 모두 1.0으로 초기화 (1.0 = keep, 0.0 = mask)
    
    mask[mask_indices] = 0.0
    # 선택된 위치만 0.0으로 설정
    
    return emb * mask
    # 마스킹된 임베딩 반환 (선택된 위치는 0)
```

**시각적 예시**:
```
임베딩: [0.1, 0.2, 0.3, 0.4, 0.5, ..., 0.512]
마스크: [1.0, 0.0, 1.0, 0.0, 1.0, ..., 1.0]  (임의 선택)
결과:  [0.1,  0.0, 0.3,  0.0, 0.5, ..., 0.512]
```

**목적**: 
- 🎯 **임베딩 노이즈 시뮬레이션** (손상된 입력 대비)
- 🎯 **모델 견고성** 증가 (과적합 방지)
- 🎯 **데이터 증강** 효과

---

### 3️⃣ 클래스 목록 구하기 (get_classes)

```python
def get_classes(self):
    return self.dataframe['class'].unique()
```

**사용 목적**:
```python
dataset = HATRDataset(df)
classes = dataset.get_classes()
# 반환: array(['dog-barking', 'dog-growling', 'speech-male', ...])

num_classes = len(classes)
# 모델의 num_classes 파라미터 설정
```

---

### 4️⃣ 길이 반환 (__len__)

```python
def __len__(self):
    return len(self.dataframe)
```

**용도**: DataLoader가 배치 구성할 때 전체 크기 파악

```python
dataset = HATRDataset(df)
print(len(dataset))  # 9876

# DataLoader 구성
loader = DataLoader(dataset, batch_size=64)
# 총 156 배치 (9876 / 64 ≈ 155)
```

---

### 5️⃣ 데이터 샘플 추출 (__getitem__)

```python
def __getitem__(self, idx):
    # 1. 행 추출
    sample = self.dataframe.iloc[idx]
    
    # 2. 메타데이터 읽기
    sound_id = sample['index']                           # 예: 1
    class_name = sample['class']                         # 예: 'dog-barking'
    top_class_name = sample['top_class']                 # 예: 'dog'
    class_idx = sample['class_idx']                      # 예: 0
    top_class_idx = sample['top_class_idx']              # 예: 0
    
    # 3. 음성 임베딩 로드
    emb_path = sample['audio_emb_filepath']
    # 예: './data/BSD10k-v1.1/features/clap_audio_embeddings/123456.npy'
    emb = torch.tensor(np.load(emb_path), dtype=torch.float32)
    # 반환: (512,) 형태 텐서

    # 4. 텍스트 임베딩 로드
    text_path = sample['text_emb_filepath']
    # 예: './data/BSD10k-v1.1/features/clap_text_embeddings/dog-barking.npy'
    text_emb = torch.tensor(np.load(text_path), dtype=torch.float32)
    # 반환: (512,) 형태 텐서

    # 5. 증강 적용
    if self.aug:
        # 가우시안 노이즈 추가
        emb = emb + torch.randn_like(emb) * 0.0001
        # 작은 노이즈로 미세한 변화 추가
        
        # 랜덤 마스킹
        emb = self._rand_mask(emb)
        
        # 텍스트도 동일하게 증강
        text_emb = text_emb + torch.randn_like(text_emb) * 0.0001
        text_emb = self._rand_mask(text_emb)

    # 6. 딕셔너리로 반환
    sample_data = {
        'sound_id': sound_id,               # 원본 ID
        'audio_embedding': emb,             # (512,) 음성 임베딩
        'text_embedding': text_emb,         # (512,) 텍스트 임베딩
        'class': class_name,                # 'dog-barking'
        'class_idx': class_idx,             # 0
        'top_class': top_class_name,        # 'dog'
        'top_class_idx': top_class_idx,     # 0
    } 
    
    return sample_data
```

---

## 📊 데이터 흐름 및 형태

### 입력 (processed_dataset.csv)
```
index | sound_id | class        | class_idx | top_class | top_class_idx | audio_emb_filepath | text_emb_filepath
------|----------|--------------|-----------|-----------|---------------|-------------------|------------------
1     | 123456   | dog-barking  | 0         | dog       | 0             | ./data/.../123456  | ./data/.../dog-barking
2     | 123457   | speech-male  | 3         | speech    | 1             | ./data/.../123457  | ./data/.../speech-male
```

### 출력 (HATRDataset[idx])
```json
{
  "sound_id": 1,
  "audio_embedding": tensor([0.123, 0.456, ..., 0.789]),  // 형태: (512,)
  "text_embedding": tensor([0.234, 0.567, ..., 0.890]),   // 형태: (512,)
  "class": "dog-barking",
  "class_idx": 0,
  "top_class": "dog",
  "top_class_idx": 0
}
```

---

## 🔄 DataLoader와 배치 구성

### 기본 사용법
```python
from torch.utils.data import DataLoader

# 데이터셋 생성
train_df = pd.read_csv('./data/processed_dataset.csv')
train_dataset = HATRDataset(train_df, aug=True, mask_pct=0.2)

# DataLoader 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0  # Windows에서는 0 권장
)

# 배치 반복
for batch in train_loader:
    audio_embs = batch['audio_embedding']      # (64, 512)
    text_embs = batch['text_embedding']        # (64, 512)
    class_ids = batch['class_idx']             # (64,)
    
    # 모델 입력
    outputs = model(audio_embs, text_embs)
    loss = criterion(outputs, class_ids)
    loss.backward()
```

---

## ⚙️ 증강 메커니즘 상세

### 1. 가우시안 노이즈 추가
```python
emb = emb + torch.randn_like(emb) * 0.0001
```

**효과**:
```
원본 임베딩:     [0.500000, 0.100000, ..., 0.300000]
노이즈 (×0.0001): [0.000034, -0.000052, ..., 0.000018]
결과:           [0.500034, 0.099948, ..., 0.300018]
```

**목적**: 미세한 변화로 모델의 견고성 증가

### 2. 랜덤 마스킹
```python
emb = self._rand_mask(emb)  # 20% 마스킹
```

**효과**:
```
mask_pct=0.2 (20%)
512 차원 중 약 102개 차원을 0으로 설정

시각화:
[0.5, 0.0, 0.3, 0.0, 0.2, ...] (0.0 = 마스킹된 부분)
```

**목적**: 불완전한 입력에 대한 견고성

### 3. 증강 비활성화
```python
# 테스트/검증 시에는 증강 비활성화
val_dataset = HATRDataset(val_df, aug=False)
```

---

## 🧪 실습 예제

### 데이터셋 시각화
```python
import pandas as pd
from dataset_utils import HATRDataset

# 데이터 로드
df = pd.read_csv('./data/processed_dataset.csv')
print(f"전체 샘플: {len(df)}")
print(f"클래스 수: {df['class'].nunique()}")

# 데이터셋 생성
dataset = HATRDataset(df, aug=True, mask_pct=0.2)

# 샘플 1개 추출
sample = dataset[0]
print("\n샘플 정보:")
print(f"  Sound ID: {sample['sound_id']}")
print(f"  Class: {sample['class']}")
print(f"  Top Class: {sample['top_class']}")
print(f"  Audio embedding shape: {sample['audio_embedding'].shape}")
print(f"  Text embedding shape: {sample['text_embedding'].shape}")

# 샘플 2번 추출 (증강 적용됨)
sample2 = dataset[0]
print("\n같은 샘플 재추출 (증강으로 인해 값이 다름):")
diff = (sample['audio_embedding'] - sample2['audio_embedding']).abs().mean()
print(f"  임베딩 평균 차이: {diff:.6f}")
```

---

## 🎯 학습 포인트

- ✅ **멀티모달**: 음성과 텍스트 둘 다 처리
- ✅ **온더플라이 증강**: 배치마다 다른 증강 적용
- ✅ **계층적 레이블**: 상위/하위 클래스 동시 제공
- ✅ **마스킹**: 불완전한 입력 시뮬레이션
- ✅ **노이즈**: 미세한 변화로 견고성 증가
- ✅ **사전 계산 임베딩**: 실시간 추출보다 빠름
