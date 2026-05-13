# build_dataset.py - 데이터셋 전처리 완벽 가이드

## 📋 파일 개요
`build_dataset.py`는 **원본 메타데이터와 임베딩을 처리**하여 모델 훈련용 데이터셋을 준비하는 파일입니다.
클래스 필터링, 인덱스 재매핑, 경로 연결 등의 전처리 작업을 수행합니다.

---

## 🔄 전체 프로세스 흐름

```
config.yaml 읽기
      ↓
메타데이터 CSV 로드
      ↓
임베딩 폴더 스캔
      ↓
필터링 (상위 클래스, "-other" 제거)
      ↓
클래스 인덱스 재매핑 (0부터 시작)
      ↓
processed_dataset.csv 생성
+ class_dict.json
+ top_class_dict.json  
+ top_class_subclass_dict.json
```

---

## 🔍 코드 상세 분석

### 1️⃣ 파일 경로 설정

```python
# --- Filepaths ---
dataset_name = get_subconfig("active_dataset")
# 반환: "BSD10k-v1.2"

metadata_csv = get_subconfig("datasets")[dataset_name]["metadata_csv"]
# 반환: "./data/BSD10k-v1.1/metadata/BSD10k_metadata.csv"

audio_emb_folder = get_subconfig("datasets")[dataset_name]["audio_emb_folder"]
# 반환: "./data/BSD10k-v1.1/features/clap_audio_embeddings"

text_emb_folder = get_subconfig("datasets")[dataset_name]["text_emb_folder"]
# 반환: "./data/BSD10k-v1.1/features/clap_text_embeddings"

# 출력 파일 경로들
output_path = get_subconfig("output_path")  # "data"
processed_dataset_csv = os.path.join(output_path, get_subconfig("processed_dataset_csv"))
class_dict_json = os.path.join(output_path, get_subconfig("class_dict_json"))
top_class_dict_json = os.path.join(output_path, get_subconfig("top_class_dict_json"))
top_class_subclass_dict_json = os.path.join(output_path, get_subconfig("top_class_subclass_dict_json"))
```

**출력 파일 위치**:
```
data/
├── processed_dataset.csv          (모델용 데이터셋)
├── class_dict.json                (클래스명→ID)
├── top_class_dict.json            (상위클래스명→ID)
└── top_class_subclass_dict_json   (계층 구조)
```

---

### 2️⃣ 메타데이터 로드 및 기본 정보 출력

```python
# --- Load metadata and build class mappings ---
df = pd.read_csv(metadata_csv)

# sound_id 정리 (공백 제거)
df['sound_id'] = df['sound_id'].astype(str).str.strip()

print(f"Examining original data from {dataset_name}:")
print(f"  Total rows: {len(df)}")
print(f"  Unique classes: {df['class'].nunique()}")
```

**출력 예시**:
```
Examining original data from BSD10k-v1.2:
  Total rows: 10352
  Unique classes: 23
```

**메타데이터 CSV 구조** (예시):
```
sound_id,class,class_idx,confidence,...
123456,dog-barking,0502,0.95,...
123457,dog-growling,0501,0.92,...
123458,speech-male,1201,0.88,...
```

---

### 3️⃣ 필터링: 상위 클래스와 "-other" 제거

```python
# Discard top-level classes and classes that belong to "-other" category
s = df['class_idx'].astype(str)

# 필터링 조건
# class_idx가 5자리이고, 끝이 99 또는 00이면 제거
df = df[~((s.str.len() == 3) & (s.str.endswith('99') | s.str.endswith('00')))].copy()

print("After filtering:", len(df))
```

**필터링 규칙 설명**:

#### class_idx 구조 (5자리):
```
0502
├─ 05: 상위 클래스 인덱스
└─ 02: 하위 클래스 인덱스
```

#### 제거 대상:
| 형태 | 예시 | 이유 |
|------|------|------|
| `00000` | 0000 | 상위 클래스 (최상위) |
| `0199` | 0199 | "-other" 카테고리 |
| `0100` | 0100 | "-other" 카테고리 |
| `0299` | 0299 | "-other" 카테고리 |

#### 유지 대상:
```
0501  ✓ 유지 (dog-growling)
0502  ✓ 유지 (dog-barking)
1201  ✓ 유지 (speech-male)
1202  ✓ 유지 (speech-female)
```

**코드 분석**:
```python
s.str.len() == 3
# class_idx를 문자열로 변환했을 때 길이가 3인가?
# "502" (5자리를 문자열로 변환 시 앞의 0이 제거될 수 있음)

s.str.endswith('99') | s.str.endswith('00')
# 끝이 99 또는 00인가?
```

---

### 4️⃣ 원본 인덱스 저장

```python
df['original_class_idx'] = df['class_idx']
```

**목적**: 원본 인덱스 보존 (나중에 검증/추적용)

---

### 5️⃣ 클래스 인덱스 재매핑 (0부터 시작)

```python
# --- Map class_idx → 0..N for training ---
original_indices = sorted(df['original_class_idx'].unique())
# 예: [501, 502, 503, ..., 2301]

index_mapping = {orig: new for new, orig in enumerate(original_indices)}
# 예: {501: 0, 502: 1, 503: 2, ..., 2301: 22}

df['class_idx'] = df['original_class_idx'].map(index_mapping)
# class_idx를 0~22 범위로 변환
```

**필요한 이유**:
```python
# 원본 인덱스 (sparse, 불규칙)
original: [501, 502, 504, 506, 1201, 1202, ...]
          ↓ (매핑)
# 재매핑된 인덱스 (dense, 연속)
new:      [0,   1,   2,   3,   4,    5,   ...]

# PyTorch CrossEntropyLoss는 0부터 시작하는 연속 인덱스 필요!
# 만약 original을 사용하면 num_classes가 너무 커짐 (2301)
```

---

### 6️⃣ 상위 클래스 추출

```python
# --- top class ---
df['class_top'] = df['class'].apply(lambda x: x.split("-")[0] if isinstance(x, str) else None)

df_sorted = df.sort_values('original_class_idx')
```

**처리 내용**:
```python
# 입력
df['class'] = ['dog-barking', 'speech-male', 'dog-growling', ...]

# 출력
df['class_top'] = ['dog', 'speech', 'dog', ...]
```

**에러 처리**: `isinstance(x, str)` 체크로 None/NaN 대비

---

### 7️⃣ 클래스 딕셔너리 생성 (이후 코드에서)

```python
# 아래는 일반적인 후속 작업 (코드에 명시 안 됨)
# build_dataset.py는 여기까지만 진행

class_dict = {}
top_class_dict = {}

# 추출한 고유 클래스명
unique_classes = sorted(df['class'].unique())
for idx, class_name in enumerate(unique_classes):
    class_dict[class_name] = idx

# 상위 클래스
unique_top_classes = sorted(df['class_top'].unique())
for idx, top_class_name in enumerate(unique_top_classes):
    top_class_dict[top_class_name] = idx

# JSON 저장
with open(class_dict_json, 'w') as f:
    json.dump(class_dict, f)
with open(top_class_dict_json, 'w') as f:
    json.dump(top_class_dict, f)
```

**생성되는 JSON 파일 형식**:

**class_dict.json**:
```json
{
  "dog-barking": 0,
  "dog-growling": 1,
  "dog-whimpering": 2,
  "speech-male": 3,
  "speech-female": 4,
  ...
}
```

**top_class_dict.json**:
```json
{
  "dog": 0,
  "speech": 1,
  "car": 2,
  "music": 3,
  "environmental": 4
}
```

---

## 📊 출력 파일 상세 설명

### processed_dataset.csv

**컬럼 구조**:
```csv
index,sound_id,class,class_idx,top_class,top_class_idx,audio_emb_filepath,text_emb_filepath
1,123456,dog-barking,0,dog,0,./data/BSD10k-v1.1/features/clap_audio_embeddings/123456.npy,./data/BSD10k-v1.1/features/clap_text_embeddings/dog-barking.npy
2,123457,dog-growling,1,dog,0,./data/BSD10k-v1.1/features/clap_audio_embeddings/123457.npy,./data/BSD10k-v1.1/features/clap_text_embeddings/dog-growling.npy
3,123458,speech-male,3,speech,1,./data/BSD10k-v1.1/features/clap_audio_embeddings/123458.npy,./data/BSD10k-v1.1/features/clap_text_embeddings/speech-male.npy
```

| 컬럼 | 예시 | 설명 |
|------|------|------|
| **index** | 1 | 행 번호 (데이터셋 내 ID) |
| **sound_id** | 123456 | 원본 음원 ID |
| **class** | dog-barking | 하위 클래스 이름 |
| **class_idx** | 0 | 재매핑된 클래스 ID |
| **top_class** | dog | 상위 클래스 이름 |
| **top_class_idx** | 0 | 상위 클래스 ID |
| **audio_emb_filepath** | ./data/.../123456.npy | 음성 임베딩 경로 |
| **text_emb_filepath** | ./data/.../dog-barking.npy | 텍스트 임베딩 경로 |

---

## 🔗 다음 단계

`build_dataset.py` 완료 후 → `train_test.py` 사용 가능

```python
# train_test.py에서
full_df = pd.read_csv(prepared_dataset_path)  # processed_dataset.csv 로드

with open(class_dict_json, 'r') as f:
    class_dict = json.load(f)  # {'dog-barking': 0, ...}
```

---

## ⚡ 실행 예시

### 명령줄에서 실행
```bash
python build_dataset.py
```

### 출력 로그
```
Examining original data from BSD10k-v1.2:
  Total rows: 10352
  Unique classes: 23
After filtering: 9876
Processing complete!
Generated files:
  - data/processed_dataset.csv
  - data/class_dict.json
  - data/top_class_dict.json
  - data/top_class_subclass_dict_json.json
```

---

## 🎯 학습 포인트

- ✅ **필터링**: 상위 클래스와 "-other"를 제거해 22개 하위 클래스만 유지
- ✅ **재매핑**: 원본 인덱스 (501~2301)를 연속 인덱스 (0~22)로 변환
- ✅ **경로 연결**: 메타데이터와 임베딩 폴더 경로를 결합
- ✅ **JSON 생성**: 클래스명과 ID의 양방향 매핑 저장
- ✅ **재현성**: 원본 인덱스 보존 + 정렬된 순서 유지
