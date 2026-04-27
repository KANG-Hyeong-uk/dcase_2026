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

## 실험 계획 (우선순위 순)

### Phase 1: 베이스라인 재현 (즉시)
| ID | 실험 | 예상 효과 | 근거 |
|----|------|-----------|------|
| 000 | 베이스라인 측정 (both mode, 5-fold) | 기준값 측정 | - |

### Phase 2: 데이터 파이프라인 강화 (1~3일)
| ID | 실험 | 예상 효과 | 근거 |
|----|------|-----------|------|
| 001 | confidence >= 4 필터링 | **+9% (2nd), +5% (top)** | 논문 A 검증 |
| 002 | confidence >= 3 필터링 비교 | +2% (2nd), +1.5% (top) | 논문 A 검증 |
| 003 | BSD35k-CS 통합 (weighted sampling 3:1) | 데이터 3배 | - |
| 004 | Focal Loss 적용 (클래스 불균형 대응) | 희귀 클래스 개선 | 논문 C |

### Phase 3: 모델 아키텍처 개선 (1~2주)
| ID | 실험 | 예상 효과 | 근거 |
|----|------|-----------|------|
| 005 | L_Top + L_Contr 계층적 손실함수 | H-Acc 직접 최적화 | 논문 B 수식 |
| 006 | hidden_size 128 → 256 | 모델 용량 증가 | 일반 원칙 |
| 007 | hidden_size 256 → 512 | 모델 용량 증가 | - |
| 008 | CLAP 마지막 2레이어 fine-tuning | 도메인 적응 | Transfer learning |
| 009 | CLAP 마지막 4레이어 fine-tuning | 더 강한 도메인 적응 | - |
| 010 | Transformer 기반 분류기로 교체 | 시퀀스 패턴 학습 | - |

### Phase 4: 하이퍼파라미터 최적화 (1주)
| ID | 실험 | 목표 |
|----|------|------|
| 011 | Optuna: λ_top, λ_contr, τ 탐색 | 계층적 손실 최적화 |
| 012 | Optuna: lr, weight_decay, batch_size | 학습 안정성 |
| 013 | CosineAnnealingLR vs StepLR 비교 | 수렴 속도 |
| 014 | dropout rate 탐색 | 과적합 방지 |

### Phase 5: 앙상블 & 제출 (마감 2주 전)
| ID | 실험 | 설명 |
|----|------|------|
| 015 | 5-fold 앙상블 (softmax 평균) | 기본 앙상블 |
| 016 | audio-only + both 모드 혼합 앙상블 | 모델 다양성 |
| 017 | TTA (가우시안 노이즈 5회 평균) | 추론 안정성 |
| 018 | 최종 제출 파일 생성 | 공식 포맷 |

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
| ID | 브랜치 | 설명 | H-Acc | Macro 2nd | Top Acc | vs baseline |
|----|--------|------|-------|-----------|---------|-------------|
| 000 | main | baseline (both mode, 5-fold) | 79.45% | 74.02% | 88.88% | - (기준값) |
| 001 | main | + confidence ≥ 4 필터링 | 83.42% | 78.52% | 93.62% | +3.97% |
| 005 | main | + L_Top + L_Contr 계층손실 | 84.02% | 79.28% | 93.62% | +4.57% |
| 006 | main | + hidden_size 256 | 84.43% | 79.76% | 93.64% | +4.98% |
| 007 | main | + hidden_size 512 (과적합) | 84.17% | 79.61% | 93.70% | +4.72% |
| 008 | main | **BEST = 006 + class_weights=balanced** | **85.29%** | **81.16%** | 93.62% | **+5.84%** |
| 009 | main | EXP-008 + TTA (noise=1e-3, n=5) | 85.29% | 81.15% | 93.62% | +5.84% |

### 베스트 설정 (EXP-008)
```bash
python train_test.py \
  --exp_name exp_008_best_classweights \
  --modes both --conf_threshold 4 \
  --hier_loss --lambda_top 0.3 --lambda_contr 0.1 --tau 0.07 \
  --hidden_size 256 --dropout 0.1 \
  --epochs 100 --batch_size 64 --lr 0.001 \
  --k_folds 5 --class_weights balanced
```

### 적용된 개선사항 (누적, 효과 검증됨)
1. **데이터 필터링**: confidence ≥ 4 (10,956 → 6,821 샘플) — 논문 A 검증, +3.97% H-Acc
2. **계층 손실**: `L_total = L_CE + 0.3·L_Top + 0.1·L_Contr (τ=0.07)` — 논문 B 수식, +0.60%
3. **모델 용량**: hidden_size 128 → 256 (분산 ±1.21 → ±0.29 4배 안정), +0.41%
4. **클래스 가중치**: balanced (`N/(K·count)`) — conf 필터 후 11.2x 불균형 해소, +0.86%
5. ❌ hidden 512: 과적합 (4k train으로 부족), -0.26% — 채택 안 함
6. ❌ TTA (noise=1e-3, n=5): 효과 없음 (모델이 이미 robust) — 채택 안 함

### EXP-000 (베이스) fold별 상세
| Fold | Accuracy | Top Acc | Macro 2nd | H-Acc | H-F1 |
|------|----------|---------|-----------|-------|-------|
| 0 | 79.01% | 89.10% | 72.49% | 78.74% | 77.95% |
| 1 | 79.92% | 88.82% | 75.59% | 80.40% | 78.77% |
| 2 | 79.87% | 88.95% | 74.02% | 79.40% | 78.72% |
| 3 | 80.01% | 88.64% | 73.41% | 78.73% | 77.76% |
| 4 | 79.32% | 88.91% | 74.60% | 80.00% | 78.47% |
| **avg** | **79.63%** | **88.88%** | **74.02%** | **79.45%** | **78.33%** |

### EXP-008 (베스트) fold별 상세
| Fold | Accuracy | Top Acc | Macro 2nd | H-Acc | H-F1 |
|------|----------|---------|-----------|-------|-------|
| 0 | 87.33% | 93.85% | 81.62% | 85.37% | 83.94% |
| 1 | 88.05% | 93.40% | 81.32% | 85.00% | 84.26% |
| 2 | 86.95% | 93.26% | 79.76% | 84.40% | 82.75% |
| 3 | 87.02% | 93.70% | 82.29% | 86.42% | 84.51% |
| 4 | 87.02% | 93.91% | 80.77% | 85.25% | 83.03% |
| **avg** | **87.27%** | **93.62%** | **81.16%** | **85.29%** | **83.70%** |

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
