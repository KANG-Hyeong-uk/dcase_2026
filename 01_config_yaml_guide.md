# config.yaml - 설정 파일 완벽 가이드

## 📋 파일 개요
`config.yaml`은 전체 프로젝트의 **중앙 설정 파일**입니다. 데이터셋 경로, 출력 경로, 메타데이터 위치 등 모든 주요 설정을 정의합니다.

---

## 🔍 주요 섹션 상세 분석

### 1️⃣ active_dataset
```yaml
active_dataset: BSD10k-v1.2  # 현재 사용할 데이터셋 선택
```
- **의미**: 사용할 데이터셋 이름 (BSD10k-v1.2 또는 BSD35k-CS 중 선택)
- **용도**: 아래 `datasets` 섹션에서 대응하는 설정을 자동으로 로드
- **변경 시기**: 다른 데이터셋을 실험하려고 할 때

---

### 2️⃣ datasets 섹션
데이터셋별 경로와 파일 설정을 정의합니다.

#### **BSD10k-v1.2** 설정
```yaml
BSD10k-v1.2:
  metadata_csv: "./data/BSD10k-v1.1/metadata/BSD10k_metadata.csv"
  audio_emb_folder: "./data/BSD10k-v1.1/features/clap_audio_embeddings"
  text_emb_folder: "./data/BSD10k-v1.1/features/clap_text_embeddings"
  class_names: "./data/BSD10k-v1.1/metadata/BST_description.csv"
```

| 항목 | 설명 | 파일 형식 |
|------|------|---------|
| **metadata_csv** | 음원의 메타데이터 (ID, 레이블, 신뢰도 등) | CSV 파일 |
| **audio_emb_folder** | CLAP 모델로 추출한 음성 임베딩 | .npy 파일 폴더 |
| **text_emb_folder** | 클래스명의 텍스트 임베딩 | .npy 파일 폴더 |
| **class_names** | Broad Sound Taxonomy (BST) 분류 정보 | CSV 파일 |

#### **BSD35k-CS** 설정
```yaml
BSD35k-CS:
  metadata_csv: "./data/BSD35k-CS/metadata/BSD35k-CS_metadata.csv"
  audio_emb_folder: "./data/BSD35k-CS/features/clap_audio_embeddings"
  text_emb_folder: "./data/BSD35k-CS/features/clap_text_embeddings"
  class_names: "./data/BSD35k-CS/metadata/BST_description.csv"
```
- **차이점**: BSD35k-CS는 더 크고, 크라우드소싱 레이블을 포함한 노이즈 데이터셋

---

### 3️⃣ INPUT PATHS 설정

#### metadata_csv 형식 예시
```
sound_id,class,class_idx,top_class,confidence,...
123456,dog-barking,0502,dog,0.95,...
123457,speech-male,1201,speech,0.88,...
```

#### class_names (BST_description.csv) 형식 예시
```
class_idx,class_name,top_class,description
0101,dog,dog,"Animal sounds from dogs"
0102,cat,dog,"Animal sounds from cats"
1201,speech-male,speech,"Male speech"
1202,speech-female,speech,"Female speech"
```

**구조 설명**:
- **class_idx**: 5자리 숫자 코드 (예: 0502)
  - 첫 2자리: 상위 클래스 인덱스
  - 마지막 3자리: 하위 클래스 인덱스
- **계층 구조**: 5개 상위 클래스 × 23개 하위 클래스

---

### 4️⃣ OUTPUT PATHS 설정

```yaml
output_path: "data"  # 모든 처리된 데이터 저장 폴더
processed_dataset_csv: "processed_dataset.csv"
class_dict_json: "class_dict.json"
top_class_dict_json: "top_class_dict.json"
top_class_subclass_dict_json: "top_class_subclass_dict.json"
```

| 파일 | 생성 시점 | 내용 |
|------|---------|------|
| **processed_dataset.csv** | `build_dataset.py` 실행 후 | 필터링된 데이터 + 임베딩 경로 |
| **class_dict.json** | `build_dataset.py` 실행 후 | `{"class_name": id, ...}` |
| **top_class_dict.json** | `build_dataset.py` 실행 후 | 상위 클래스만 매핑 |
| **top_class_subclass_dict_json** | `build_dataset.py` 실행 후 | 계층 구조 매핑 |

---

## 🔗 다른 파일에서의 사용

### utils.py에서
```python
def get_subconfig(section, path='config.yaml'):
    config = load_config(path)
    return config.get(section, {})

# 사용 예시
dataset_name = get_subconfig("active_dataset")
datasets = get_subconfig("datasets")
```

### build_dataset.py에서
```python
dataset_name = get_subconfig("active_dataset")
metadata_csv = get_subconfig("datasets")[dataset_name]["metadata_csv"]
audio_emb_folder = get_subconfig("datasets")[dataset_name]["audio_emb_folder"]
```

---

## 🛠️ 커스터마이제이션 가이드

### 새로운 데이터셋 추가하기

1. **config.yaml 수정**:
```yaml
active_dataset: MyDataset

datasets:
  MyDataset:
    metadata_csv: "./data/MyDataset/metadata.csv"
    audio_emb_folder: "./data/MyDataset/features/audio"
    text_emb_folder: "./data/MyDataset/features/text"
    class_names: "./data/MyDataset/metadata/classes.csv"
```

2. **메타데이터 CSV 형식 확인** (필수 컬럼):
   - `sound_id`: 음원 고유 ID
   - `class`: 하위 클래스명 (예: "dog-barking")
   - `class_idx`: 클래스 인덱스
   - 기타: 신뢰도, 공급자 등

---

## ⚠️ 주의사항

1. **경로는 상대 경로 권장**
   - 프로젝트 루트 기준 상대 경로 사용
   - 절대 경로로 변경 가능

2. **폴더 구조 확인 필수**
   ```
   data/
   ├── BSD10k-v1.1/
   │   ├── features/
   │   │   ├── clap_audio_embeddings/  (*.npy 파일들)
   │   │   └── clap_text_embeddings/   (*.npy 파일들)
   │   └── metadata/
   │       ├── BSD10k_metadata.csv
   │       └── BST_description.csv
   ```

3. **피하야 할 실수**
   - ❌ 경로 끝에 `/` 추가
   - ❌ 대소문자 혼용 (파일시스템에 따라 다를 수 있음)
   - ❌ 존재하지 않는 폴더 지정

---

## 📊 데이터 흐름

```
config.yaml
    ↓
build_dataset.py (config 읽기)
    ↓
임베딩 폴더에서 .npy 파일 로드
메타데이터 CSV 파일 로드
    ↓
processed_dataset.csv 생성 (+ JSON 파일들)
    ↓
train_test.py (생성된 CSV 사용)
```

---

## 📝 예제: 전체 프로세스

```bash
# 1. config.yaml에서 활성 데이터셋 확인
active_dataset: BSD10k-v1.2

# 2. build_dataset.py 실행
python build_dataset.py
# → data/processed_dataset.csv 생성

# 3. train_test.py가 processed_dataset.csv 사용
python train_test.py
# → 모델 훈련 시작
```

---

## 🎯 학습 포인트

- ✅ config.yaml은 데이터셋 선택과 경로를 중앙에서 관리
- ✅ `get_subconfig()` 함수로 YAML 파일 로드
- ✅ 출력 파일들은 자동 생성되는 중간 처리 결과물
- ✅ 새 데이터셋 추가는 섹션만 추가하면 됨
