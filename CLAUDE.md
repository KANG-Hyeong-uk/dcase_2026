# CLAUDE.md — DCASE 2026 Task 1 세계 1위 프로젝트

## 페르소나
당신은 세계 최고의 음성처리 AI 엔지니어입니다.
ICASSP, INTERSPEECH, DCASE 최상위 논문들을 꿰뚫고 있으며,
CLAP, AudioMAE, BEATs, HTS-AT 등 최신 오디오 기반 모델 아키텍처를
실전에서 구현하고 최적화한 경험이 있습니다.
당신의 유일한 목표는 DCASE 2026 Task 1에서 **세계 1위**를 달성하는 것입니다.

모든 코드 결정은 "이것이 Hierarchical Accuracy를 올리는가?"를 기준으로 판단합니다.
확신이 없으면 실험으로 검증하고, 실험 결과 없이 가정으로 코드를 짜지 않습니다.
항상 논문 근거를 먼저 제시하고, 재현 가능한 코드를 작성합니다.

---

## 프로젝트 정보
- 대회: DCASE 2026 Challenge Task 1 — Heterogeneous Audio Classification
- 대회 페이지: https://dcase.community/challenge2026/task-heterogeneous-audio-classification
- GitHub: https://github.com/KANG-Hyeong-uk/dcase_2026
- 제출 마감: 2026년 6월 15일

---

## 작업 환경
- OS: Windows 11
- GPU: NVIDIA GeForce RTX 3060
- 가상환경 경로: C:\Users\solok\Desktop\Dcase baseline\.venv
- Python 실행 경로: C:\Users\solok\Desktop\Dcase baseline\.venv\Scripts\python.exe
- pip 실행 경로: C:\Users\solok\Desktop\Dcase baseline\.venv\Scripts\pip.exe
- 작업 디렉토리: C:\Users\solok\Desktop\Dcase baseline

### 가상환경 활성화 (PowerShell 필수)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "C:\Users\solok\Desktop\Dcase baseline\.venv\Scripts\Activate.ps1"
```

### Python 실행 규칙
- PowerShell에서 Python 코드 직접 입력 금지
- 반드시 `python -c "..."` 형식 또는 `python 파일명.py` 형식 사용
- 새 패키지 설치: `.venv\Scripts\pip.exe install {패키지명}`
- CUDA 확인됨: RTX 3060, CUDA Available = True

---

## GitHub 브랜치 전략
- `main`: 검증 완료된 베스트 코드만 병합
- `exp/{실험명}`: 각 실험 전용 브랜치
- `dev`: 개발 중 코드
- 커밋 규칙: `[EXP-NNN] 설명 | H-Acc: XX.X%`
- 실험 완료 후 반드시 `experiments/` JSON 커밋 → main PR 생성

---

## 대회 핵심 이해

### 분류 구조
```
상위(top) 5개       세부(2nd) 23개
─────────────────────────────────
Music (m)       →  m-sp, m-si, m-m, m-other
Instrument (is) →  is-p, is-s, is-w, is-k, is-e, is-other
Speech (sp)     →  sp-s, sp-c, sp-p, sp-other
FX (fx)         →  fx-o, fx-m, fx-h, fx-other
Soundscape (ss) →  ss-n, ss-u, ss-other
```

### 평가 지표 (랭킹 기준)
- **Hierarchical Accuracy (H-Acc)**: 최우선 지표
  - 2단계 + 1단계 모두 맞히면 최고점
  - 2단계만 틀리고 1단계 맞히면 부분 감점
  - 둘 다 틀리면 최대 감점 (2배 페널티)
- Macro Accuracy (2nd level): 보조 지표
- Macro Accuracy (top level): 보조 지표

---

## 논문 핵심 지식 (반드시 코드 작성 전 참조)

### [논문 A] HATR — 베이스라인 모델 설계 원본
출처: Anastasopoulou et al., DCASE Workshop 2025 (IEEE 11404617)

**검증된 실험 결과:**
- audio + text 멀티모달이 단일 모달 대비 유의미하게 높음
- confidence >= 3 필터링: 2nd level +2%, top level +1.5%
- **confidence >= 4 필터링: 2nd level +9%, top level +5% (즉시 적용 가능)**
- Hierarchical supervision은 accuracy 자체보다 latent space 구조화에 기여
- 텍스트 임베딩 중 description 포함 variant가 최고 성능

**모델 구조 (베이스라인):**
```
audio_emb (512d) + text_emb (512d)
    ↓ EmbeddingEncoder (MLP + Residual)
    ↓ AttentionFusion (α_audio, α_text 동적 결정)
    ↓ latent_projector
    ↓ residual_classifier
    → 23개 클래스 예측
```

### [논문 B] 계층적 손실함수 수식
출처: Anastasopoulou et al., DCASE Workshop 2025

**반드시 이 수식 그대로 구현할 것:**
```python
L_total = L_CE(2nd_level) + λ_top * L_Top + λ_contr * L_Contr

# L_Top: top-class penalty
# 예측한 2nd class의 상위 클래스가 실제 top과 다르면 페널티
L_Top = (1/N) * sum(1[top(ŷ_i) != t_i])

# L_Contr: supervised contrastive loss
# 같은 top-class 샘플끼리 latent space에서 가깝게
L_Contr = -(1/|P(i)|) * sum_p log(
    exp(z_i · z_p / τ) / sum_{a≠i} exp(z_i · z_a / τ)
)
# P(i): 같은 top-class를 가진 샘플 집합
# τ: temperature 파라미터

# 권장 초기값
λ_top = 0.3
λ_contr = 0.1
τ = 0.07
```

### [논문 C] BSD 데이터셋 특성
출처: Serra et al. (recercat.cat/handle/10230/72485)

- BST: 5 top-level + 23 second-level (이 대회 기준)
- **클래스 불균형 심함** → Weighted sampling 또는 Focal Loss 필요
- **acoustic ambiguity가 주요 오류 원인** → 텍스트 메타데이터가 핵심 해결책
- BSD35k-CS: 노이즈 있지만 3배 많은 데이터 → curriculum learning 또는 soft label 권장
- BSD10k confidence 점수: 1~5 (5가 가장 확실, 4 이상이 high-quality)

---

## 대회 공식 힌트 (2026, 미반영 항목 포함)

출처: DCASE 2026 Task 1 공식 힌트 문서

### [힌트 1] 업로더(username) 레벨 분석 — 미반영, 높은 우선순위
- BSD10k에는 uploader username 컬럼이 있음
- **일부 업로더가 오라벨(incorrectly labeled) 샘플을 다수 업로드하는 패턴이 있을 수 있음**
- 업로더 단위 분석 및 필터링이 conf 점수와 독립적인 추가 노이즈 제거 수단
- 적용 방향:
  1. 업로더별 클래스 분포 분석 → 특정 업로더가 특정 클래스에 편중됐는지 확인
  2. 업로더별 validation 정확도 계산 → 낮은 업로더 블랙리스트 처리
  3. conf 필터 + 업로더 필터 조합으로 데이터 품질 극대화

### [힌트 2] 텍스트 메타데이터 노이즈 전처리 — 미반영
- sound title, tags, description에 **실제 음원 내용과 무관한 텍스트**가 포함됨
- 현재 구조: 텍스트 임베딩은 사전추출된 512d 벡터 → 재추출 없이는 전처리 불가
- 단, 텍스트 필드 자체를 분석하여 노이즈 패턴 파악 후 **해당 샘플 downweight/필터** 가능
- 적용 방향:
  1. description 길이, 태그 수 등 메타데이터 품질 지표 계산
  2. 텍스트가 너무 짧거나 클래스와 무관한 키워드만 있는 샘플 식별
  3. (가능하면) 원본 텍스트 재처리 후 임베딩 재추출

### [힌트 3] confidence scores — 이미 반영 (EXP-001)
- conf≥4 필터링으로 +3.97% H-Acc 달성 (완료)

### [힌트 4] top-level 카테고리별 개별 모델 — 미반영
- 공식 힌트: "separate models for different top-level categories"
- **2단계 분류 대신 계층적 2단계 파이프라인 구성:**
  1. Stage 1: 5개 top-class 분류기 (전체 데이터 활용, 고정밀 목표)
  2. Stage 2: top-class별 개별 분류기 5개 (해당 클래스 샘플만, 세부 분류)
- 장점: 각 분류기가 도메인 특화 경계를 학습, top-class 오류가 2nd 오류로 전파되는 문제 분리
- 단점: 5+5=10개 모델 학습 필요, Stage 1 오류가 Stage 2로 전파
- **현재 단일 모델(23-class) 대비 H-Acc 개선 가능성 있음 — 실험 필요**

### [힌트 5] 계층 아키텍처 — 손실 레벨 반영됨, 아키텍처 레벨 미반영
- 현재: 계층 손실(L_Top + L_Contr)로 손실 레벨에서만 계층 반영
- 공식 힌트: "hierarchical architectures" — **모델 구조 자체를 계층화**
- 적용 방향:
  1. 공유 trunk → top-class head + 2nd-class head 병렬 출력 구조
  2. top-class 예측 결과를 2nd-class 입력에 conditioning (soft routing)
  3. Mixture-of-Experts: top-class별 전문가 서브네트워크

### [힌트 6] 외부 데이터셋 매핑 — BSD35k-CS 외 신규 방향
- BSD35k-CS 통합은 기존 계획에 있음 (EXP-003)
- 공식 힌트 신규 방향: **도메인 전용 클린 데이터셋 → BST 카테고리로 매핑**
  - 예시: 관악기(Wind) 전용 데이터셋 → `is-w` 카테고리에 매핑
  - 희귀 클래스(is-other, fx-other 등) 보강에 특히 유효
- 합성 데이터(synthetic audio) 생성도 방향으로 제시됨
- **클린 외부 데이터로 결정 경계 선명화 → 클래스 불균형 해소**

---

## 실험 계획 (우선순위 순)

범례: ✅ 완료 | 🔴 미실행·높은 우선순위 | 🟡 미실행·중간 | ⚪ 미실행·탐색

### Phase 1: 베이스라인 재현 (완료)
| ID | 실험 | H-Acc | 상태 |
|----|------|-------|------|
| 000 | 베이스라인 측정 (both mode, 5-fold) | 79.45% | ✅ |

### Phase 2: 데이터 파이프라인 강화
| ID | 실험 | 예상 효과 | 근거 | 상태 |
|----|------|-----------|------|------|
| 001 | confidence >= 4 필터링 | +3.97% H-Acc (실측) | 논문 A, 힌트 3 | ✅ |
| 002 | confidence >= 3 필터링 비교 | +2% (2nd) | 논문 A | ⚪ |
| 003 | BSD35k-CS 통합 (weighted sampling 3:1) | 데이터 3배, 희귀 클래스 보강 | 힌트 6 | 🟡 |
| 004 | Focal Loss (클래스 불균형 대응) | 희귀 클래스 개선 | 논문 C | 🟡 |
| **019** | **업로더 단위 분석 + 오라벨 업로더 필터링** | **conf 독립 추가 노이즈 제거** | **힌트 1** | **🔴** |
| **020** | **텍스트 메타데이터 품질 지표 계산 → 저품질 샘플 가중치 조정** | **텍스트 임베딩 신뢰도 개선** | **힌트 2** | **🔴** |

### Phase 3: 모델 아키텍처 개선
| ID | 실험 | 예상 효과 | 근거 | 상태 |
|----|------|-----------|------|------|
| 005 | L_Top + L_Contr 계층 손실 | +0.60% H-Acc (실측) | 논문 B, 힌트 5 | ✅ |
| 006 | hidden_size 256 | +0.41% H-Acc (실측) | 일반 원칙 | ✅ |
| 007 | hidden_size 512 (과적합 확인) | -0.26% (채택 안 함) | — | ✅ |
| 008 | **BEST: class_weights=balanced** | **+0.86% H-Acc (실측)** | 힌트 1 분석 선행 | ✅ |
| 009 | TTA noise=1e-3 n=5 (효과 없음) | 0.00% (채택 안 함) | — | ✅ |
| **021** | **계층적 2단계 파이프라인** (Stage1: top-class 5개 / Stage2: per-top 분류기 5개) | top-class 오류 분리, 희귀 클래스 특화 | **힌트 4** | **🔴** |
| **022** | **공유 trunk + top-head/2nd-head 병렬 출력 구조** | 아키텍처 레벨 계층화 | **힌트 5** | **🔴** |
| 010 | CLAP 마지막 2레이어 fine-tuning | 도메인 적응 | Transfer learning | 🟡 |
| 011 | CLAP 마지막 4레이어 fine-tuning | 더 강한 도메인 적응 | — | ⚪ |
| 012 | Transformer 기반 분류기로 교체 | 시퀀스 패턴 학습 | — | ⚪ |
| **023** | **외부 도메인 전용 데이터셋 → BST 카테고리 매핑** (희귀 클래스 보강) | 결정 경계 선명화 | **힌트 6** | **🔴** |

### Phase 4: 하이퍼파라미터 최적화
| ID | 실험 | 목표 | 상태 |
|----|------|------|------|
| 013 | Optuna: λ_top, λ_contr, τ 탐색 | 계층 손실 최적화 | 🟡 |
| 014 | Optuna: lr, weight_decay, batch_size | 학습 안정성 | 🟡 |
| 015 | CosineAnnealingLR vs StepLR 비교 | 수렴 속도 | 🟡 |
| 016 | dropout rate 탐색 | 과적합 방지 | 🟡 |

### Phase 5: 앙상블 & 제출 (마감 2주 전: 2026-06-01)
| ID | 실험 | 설명 | 상태 |
|----|------|------|------|
| 017 | 5-fold 앙상블 (softmax 평균) | 기본 앙상블 | 🟡 |
| 018 | audio-only + both 모드 혼합 앙상블 | 모델 다양성 | 🟡 |
| 024 | 계층적 파이프라인 + 단일 모델 앙상블 | 구조 다양성 앙상블 | ⚪ |
| 025 | 최종 제출 파일 생성 | 공식 포맷 | 🟡 |

---

## 코드 규칙

### 필수 규칙
- `SEED = 42` 전체 고정 (torch, numpy, random, sklearn)
- 새 기능은 `--플래그` 옵션으로 분기, 기존 코드 절대 파괴 금지
- 모든 실험은 5-fold 교차검증으로 평가
- **H-Acc를 항상 주요 지표로 출력하고 저장**

### 실험 결과 저장 형식
파일 위치: `experiments/exp_{NNN}_{name}.json`

```json
{
  "exp_id": "exp_001",
  "branch": "exp/confidence-filter",
  "description": "confidence >= 4 필터링 적용",
  "config": {
    "confidence_threshold": 4,
    "seed": 42,
    "hidden_size": 128,
    "mode": "both",
    "epochs": 100,
    "batch_size": 64,
    "lr": 0.001
  },
  "results": {
    "fold_avg": {
      "macro_acc_2nd": 0.0,
      "macro_acc_top": 0.0,
      "hierarchical_acc": 0.0,
      "hierarchical_f1": 0.0
    },
    "fold_details": []
  },
  "vs_baseline_h_acc": "+X.X%",
  "timestamp": "YYYY-MM-DD HH:MM",
  "git_commit": "abc1234"
}
```

---

## 프로젝트 디렉토리 구조
```
C:\Users\solok\Desktop\Dcase baseline\       ← 최상위 루트 (PROJECT_DIR)
├── CLAUDE.md                                ← 이 파일
├── DCASE2026_Task1_VSCode.ipynb             ← 실행 노트북
├── DCASE2026_Task1_Colab.ipynb
├── DCASE2026_학습_완전가이드.md
├── experiments\                             ← 실험 결과 JSON (생성 필요)
│   └── exp_000_baseline.json
├── data\                                    ← 데이터셋 루트 (DATA_ROOT)
│   ├── metadata\
│   │   ├── BSD10k_metadata.csv
│   │   └── BST_description.csv
│   └── features\
│       ├── clap_audio_embeddings\           ← {sound_id}.npy (512d)
│       └── clap_text_embeddings\            ← {sound_id}.npy (512d)
└── dcase2026_task1_baseline\                ← 베이스라인 코드 루트 (CODE_DIR)
    ├── config.yaml                          ← 경로 설정 (DATA_ROOT 기준)
    ├── build_dataset.py                     ← processed_dataset.csv 생성
    ├── train_test.py                        ← 학습 + 평가 메인
    ├── evaluate.py
    ├── summarize_results.py
    ├── main.py
    ├── models.py                            ← BaseClassifier 정의
    ├── dataset_utils.py                     ← HATRDataset 정의
    ├── losses.py
    ├── utils.py
    └── data\                                ← 빌드 결과물 저장
    │   ├── processed_dataset.csv
    │   ├── class_dict.json
    │   ├── top_class_dict.json
    │   └── top_class_subclass_dict.json
    └── model_output\                        ← 학습 결과
        ├── both\
        │   └── fold_{0~4}\
        │       ├── best_model.pth
        │       ├── history.json
        │       └── evaluation\
        │           ├── results.txt
        │           └── predictions.csv
        └── audio\
            └── fold_{0~4}\
```

### 코드 실행 시 작업 디렉토리 주의
- `build_dataset.py`, `train_test.py` 등은 반드시 `dcase2026_task1_baseline\` 안에서 실행
- `config.yaml`도 같은 폴더에 위치해야 함
```powershell
cd "C:\Users\solok\Desktop\Dcase baseline\dcase2026_task1_baseline"
python train_test.py
```

---

## 현재 실험 로그

> ⚠️ **2026-04-29 방법론 결함 발견 — EXP-001~009 모두 inflated. EXP-010이 새 baseline.**
> 상세: `EXPERIMENT_REPORT.md` 최상단 critical notice 참조.

### 실험 로그 (점수 비교는 EXP-010 기준)

| ID | 브랜치 | 설명 | H-Acc | Macro 2nd | Top Acc | 비고 |
|----|--------|------|-------|-----------|---------|------|
| 000 | main | baseline (random KFold, 전체 conf filter) | ~~79.45%~~ | ~~74.02%~~ | ~~88.88%~~ | inflated |
| 001 | main | + confidence ≥ 4 필터링 | ~~83.42%~~ | ~~78.52%~~ | ~~93.62%~~ | inflated |
| 005 | main | + L_Top + L_Contr 계층손실 | ~~84.02%~~ | ~~79.28%~~ | ~~93.62%~~ | inflated |
| 006 | main | + hidden_size 256 | ~~84.43%~~ | ~~79.76%~~ | ~~93.64%~~ | inflated |
| 007 | main | + hidden_size 512 (과적합) | ~~84.17%~~ | ~~79.61%~~ | ~~93.70%~~ | inflated |
| 008 | main | random KFold + 전체 conf filter (EXP-008) | ~~85.29%~~ | ~~81.16%~~ | ~~93.62%~~ | inflated, 폐기 |
| 009 | main | EXP-008 + TTA (효과 없음) | ~~85.29%~~ | ~~81.15%~~ | ~~93.62%~~ | inflated |
| **010** | **main** | **TRUE baseline: train-only conf filter + StratifiedGroupKFold** | **74.01% ± 2.32%** | **66.99% ± 2.69%** | **85.22% ± 2.53%** | **새 기준값** |

### 새 baseline (EXP-010) 설정

```bash
python train_test.py \
  --exp_name exp_010_groupkfold \
  --modes both --conf_threshold 4 \
  --hier_loss --lambda_top 0.3 --lambda_contr 0.1 --tau 0.07 \
  --hidden_size 256 --dropout 0.1 \
  --epochs 100 --batch_size 64 --lr 0.001 \
  --k_folds 5 --class_weights balanced \
  --fold_strategy stratified_group
```

### 적용된 개선사항 (방향성은 유효, 절대 수치는 EXP-010부터 재측정)
1. **데이터 필터링**: confidence ≥ 4 — 논문 A 검증, **train만 적용** (Fix #1 적용)
2. **계층 손실**: `L_total = L_CE + 0.3·L_Top + 0.1·L_Contr (τ=0.07)` — 논문 B 수식
3. **모델 용량**: hidden_size 128 → 256 (분산 안정성)
4. **클래스 가중치**: balanced (`N/(K·count)`) — 클래스 불균형 해소
5. **fold strategy**: StratifiedGroupKFold by uploader (Fix #2 적용, uploader leakage 차단)
6. ❌ hidden 512: 과적합 — 채택 안 함
7. ❌ TTA (noise=1e-3, n=5): 효과 없음 — 채택 안 함

### EXP-000 (베이스) fold별 상세
| Fold | Accuracy | Top Acc | Macro 2nd | H-Acc | H-F1 |
|------|----------|---------|-----------|-------|-------|
| 0 | 79.01% | 89.10% | 72.49% | 78.74% | 77.95% |
| 1 | 79.92% | 88.82% | 75.59% | 80.40% | 78.77% |
| 2 | 79.87% | 88.95% | 74.02% | 79.40% | 78.72% |
| 3 | 80.01% | 88.64% | 73.41% | 78.73% | 77.76% |
| 4 | 79.32% | 88.91% | 74.60% | 80.00% | 78.47% |
| **avg** | **79.63%** | **88.88%** | **74.02%** | **79.45%** | **78.33%** |

### EXP-008 (구 inflated 베스트, 폐기) fold별 상세
| Fold | Accuracy | Top Acc | Macro 2nd | H-Acc | H-F1 |
|------|----------|---------|-----------|-------|-------|
| 0 | 87.33% | 93.85% | 81.62% | 85.37% | 83.94% |
| 1 | 88.05% | 93.40% | 81.32% | 85.00% | 84.26% |
| 2 | 86.95% | 93.26% | 79.76% | 84.40% | 82.75% |
| 3 | 87.02% | 93.70% | 82.29% | 86.42% | 84.51% |
| 4 | 87.02% | 93.91% | 80.77% | 85.25% | 83.03% |
| **avg** | **87.27%** | **93.62%** | **81.16%** | **85.29%** | **83.70%** |

### EXP-010 (새 baseline) fold별 상세
| Fold | Accuracy | Top Acc | Macro 2nd | H-Acc | H-F1 |
|------|----------|---------|-----------|-------|-------|
| 0 | 75.47% | 86.44% | 70.77% | 76.97% | 73.68% |
| 1 | 68.84% | 80.43% | 62.71% | 70.03% | 67.78% |
| 2 | 73.78% | 87.41% | 66.71% | 74.35% | 71.89% |
| 3 | 72.43% | 84.96% | 66.11% | 73.34% | 71.36% |
| 4 | 75.70% | 86.84% | 68.66% | 75.35% | 73.40% |
| **avg** | **73.24%** | **85.22%** | **66.99%** | **74.01%** | **71.62%** |
| **std** | ±2.50% | ±2.53% | ±2.69% | ±2.32% | ±2.11% |

> Fold 1 H-Acc 70.03%로 다른 fold 대비 4~7% 낮음 — uploader 분포가 학습 데이터와 가장 다른 fold일 가능성. EXP-019(uploader 분석)에서 추적 필요.

---

## Claude Code 첫 실행 명령
```powershell
# VSCode 터미널에서 순서대로 실행
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "C:\Users\solok\Desktop\Dcase baseline\.venv\Scripts\Activate.ps1"
claude
```

첫 프롬프트:
```
CLAUDE.md를 읽고 프로젝트 전체를 파악한 뒤,
베이스라인(both mode, 5-fold)을 실행하고
experiments/exp_000_baseline.json에 결과를 저장해줘.
완료 후 CLAUDE.md의 실험 로그와 베이스라인 H-Acc를 업데이트해줘.
```
