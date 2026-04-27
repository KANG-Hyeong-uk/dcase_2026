# DCASE 2026 Task 1 실험 보고서

**베이스라인 79.45% → 베스트 85.29% (+5.84% H-Acc), 7개 누적 개선 실험**

---

## TL;DR

| 구분 | H-Acc | Macro 2nd | Top Acc | 주요 변경 |
|------|-------|-----------|---------|-----------|
| 1차 (베이스라인) | 79.45% | 74.02% | 88.88% | HATR 원본 재현, CrossEntropy, hidden=128 |
| 2차 베스트 (EXP-008) | **85.29%** | **81.16%** | **93.62%** | conf≥4 필터 + 계층 손실 + hidden=256 + class_weights |
| **개선폭** | **+5.84%** | **+7.14%** | **+4.74%** | — |

---

## 1. 프로젝트 배경

- **대회**: DCASE 2026 Challenge Task 1 — Heterogeneous Audio Classification
- **데이터**: BSD10k — 10,956개 sound clip, 23개 2nd-level 클래스, 5개 top-level 클래스 (Music, Instrument, Speech, FX, Soundscape)
- **입력**: 사전추출된 CLAP audio 임베딩 + text 임베딩 (각 512d, L2-norm=1.0 정규화 완료)
- **평가지표**: Hierarchical Accuracy (H-Acc) — 메인 랭킹 지표. 2nd-level 정답이면 최고점, 2nd 틀리고 top 맞으면 부분 점수(λ=0.5), 둘 다 틀리면 0점
- **검증**: 5-fold StratifiedKFold cross-validation (SEED=42)
- **하드웨어**: NVIDIA RTX 3060

---

## 2. 1차 실험: 베이스라인 재현

### 목표

논문(Anastasopoulou et al., DCASE Workshop 2025)의 HATR 모델을 Windows 환경에서 재현하고 기준값을 측정한다. 이 단계에서는 성능 개선보다 환경 세팅과 학습 파이프라인 검증 자체가 목표였다.

### 모델 구조

```
audio_emb (512d) ─┐
                   ├─ EmbeddingEncoder (MLP + Residual×3) × 2
text_emb  (512d) ─┘
                   ↓
            AttentionFusion (α_audio, α_text 동적 결정)
                   ↓
           latent_projector
                   ↓
         ResidualBlock × 2
                   ↓
          23-class predictor
```

### 설정

| 항목 | 값 |
|------|----|
| 손실 | CrossEntropy (label_smoothing=0.01) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| 스케줄러 | StepLR (step=20, γ=0.5) |
| Augmentation | Gaussian noise σ=1e-4 + random feature masking 70% |
| hidden_size | 128 |
| dropout | 0.1 |
| batch_size | 64 |
| epochs | 100 (early stop 5×3) |
| 기타 | Windows 호환: num_workers=0, config.yaml 빈 키 패치 |

### 결과 (5-fold avg)

| H-Acc | Macro 2nd | Top Acc | Accuracy | H-F1 |
|-------|-----------|---------|----------|-------|
| 79.45% | 74.02% | 88.88% | 79.63% | 78.33% |

fold별로는 78.73%~80.40% 범위로 비교적 안정적이었으나 개선 여지가 많았다.

### 한계

- 클래스 불균형(7.0x) 미대응 — 손실이 다수 클래스에 지배됨
- BSD10k confidence 점수 미활용 — 신뢰도 낮은 라벨이 노이즈로 작용
- 계층 구조(top ↔ 2nd 관계)를 손실에 반영하지 않음 — H-Acc가 아닌 단순 accuracy를 최적화
- 모델 capacity가 작을 수 있음 (hidden=128)

---

## 3. 2차 실험: 데이터 분석

개선 실험을 시작하기 전에 데이터셋 자체를 먼저 분석했다. 어디서 개선 여지가 있는지 파악해야 가설을 제대로 세울 수 있기 때문이다.

| 분석 항목 | 발견 |
|----------|------|
| Confidence 분포 | conf=4가 전체의 55%, conf=5는 7%. conf≥4가 62% (6,821개) 보존 — sweet spot |
| 클래스 불균형 (raw) | 7.0x (fx-o 1,204개 vs fx-a 171개) |
| 클래스 불균형 (conf≥4 필터 후) | **11.2x** — 필터가 희귀 클래스에서 더 많이 제거, 오히려 불균형 악화 |
| 임베딩 정합성 | L2-norm=1.0, mean≈0, std≈0.0442 → 추가 정규화 불필요 |

conf≥4 필터가 불균형을 악화(7.0x → 11.2x)시킨다는 발견은 이후 class_weights 실험(EXP-008)의 직접적인 근거가 됐다.

---

## 4. 2차 실험: 7개 누적 개선 실험

각 실험은 직전 베스트 설정에 변경을 하나씩 추가하는 누적 방식으로 진행했다. EXP-007처럼 regression이 발생한 경우 해당 변경을 채택하지 않고 이전 설정으로 돌아갔다.

### EXP-000: 베이스라인 재현 (기준값)

**가설**: 1차 실험 결과를 공식 평가 파이프라인(`train_test.py`)으로 재현하여 기준값을 확정한다.

**변경점**: 없음. CLAUDE.md 파이프라인 기준으로 동일 설정 재실행.

**결과**:

| H-Acc | std | vs 베이스 |
|-------|-----|-----------|
| 79.45% | — | 0.00% |

**해석**: 1차 실험과 동일한 수치가 나와 재현성 확인 완료. 이 값을 기준점으로 사용한다.

---

### EXP-001: confidence ≥ 4 필터링

**가설**: Anastasopoulou et al. (DCASE 2025)에 따르면 BSD10k의 저신뢰(conf 1~3) 라벨은 노이즈로 작용한다. conf≥4(high-quality) 샘플만 사용하면 라벨 노이즈가 줄어 H-Acc가 크게 오를 것이다. 논문에서 conf≥4 적용 시 2nd-level +9%, top-level +5% 개선을 보고했다.

**변경점**: 학습 데이터를 conf≥4로 필터링 (10,956 → 6,821개, 62% 보존). `--conf_threshold 4` 플래그 추가.

**결과**:

| H-Acc | std | vs 베이스 | vs EXP-000 |
|-------|-----|-----------|-----------|
| 83.42% | ±1.33 | +3.97% | +3.97% |

**해석**: 가설 일치. 단일 변경으로 +3.97%라는 가장 큰 개선을 달성했다. 다만 논문 보고값(+9%)보다는 낮은데, 이는 데이터셋 분할 방식이나 사전추출 임베딩 품질 차이 때문으로 보인다. 이후 모든 실험의 기본 설정으로 채택.

---

### EXP-005: 계층 손실 (L_Top + L_Contr)

**가설**: 표준 CrossEntropy는 23개 클래스를 독립적으로 다루지만, H-Acc는 top-class 계층 정보까지 평가한다. 계층 구조를 손실에 직접 반영하면 latent space가 top-class 기준으로 구조화되어 H-Acc가 오를 것이다.

- **L_Top**: fine-class softmax 출력 → membership matrix로 top-class 확률 집계 → top label에 대한 NLL (미분가능 surrogate)
- **L_Contr** (Khosla 2020 SupCon): 같은 top-class 샘플을 positive로 묶는 supervised contrastive loss

```
L_total = L_CE + 0.3·L_Top + 0.1·L_Contr (τ=0.07)

L_Contr = -(1/|P(i)|) · Σ_{p∈P(i)} log( exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) )
```

**변경점**: `--hier_loss --lambda_top 0.3 --lambda_contr 0.1 --tau 0.07` 추가. EXP-001 설정 위에 누적.

**결과**:

| H-Acc | std | vs 베이스 | vs EXP-001 |
|-------|-----|-----------|-----------|
| 84.02% | ±1.21 | +4.57% | +0.60% |

**해석**: 가설 일치. +0.60%로 의미 있는 개선. std는 ±1.33 → ±1.21로 소폭 줄었다. 계층 손실이 모델을 H-Acc 방향으로 직접 유도하는 효과가 확인됐다.

---

### EXP-006: hidden_size 256

**가설**: EXP-005에서 계층 손실을 추가했음에도 std가 여전히 크다(±1.21). hidden=128이 6,821개 샘플 + 계층 손실의 복잡도를 처리하기 부족할 수 있다. hidden을 2배로 늘려 모델 capacity를 키우면 더 잘 학습될 것이다.

**변경점**: `--hidden_size 256`. EXP-005 설정 위에 누적.

**결과**:

| H-Acc | std | vs 베이스 | vs EXP-005 |
|-------|-----|-----------|-----------|
| 84.43% | **±0.29** | +4.98% | +0.41% |

**해석**: 가설 일치. H-Acc +0.41% 개선보다 더 주목할 부분은 std가 ±1.21 → **±0.29**로 4배 이상 안정화됐다는 점이다. hidden=128이 underfit 상태였음을 시사. 이후 hidden=512 실험(EXP-007)의 비교 기준이 됨.

---

### EXP-007: hidden_size 512

**가설**: hidden=256이 좋았으니, 512로 더 키우면 추가 이득이 있을 것이다. dropout을 0.10 → 0.15로 올려 과적합을 방지한다.

**변경점**: `--hidden_size 512 --dropout 0.15`. EXP-006 대비 변경.

**결과**:

| H-Acc | std | vs 베이스 | vs EXP-006 |
|-------|-----|-----------|-----------|
| 84.17% | ±1.23 | +4.72% | **-0.26%** |

**해석**: 가설 불일치 — regression 발생. conf≥4 필터 후 train set이 ~4,000개 수준인데, hidden=512는 그 규모에 비해 capacity 과잉이다. dropout을 높였음에도 std가 ±0.29 → ±1.23으로 다시 불안정해진 것이 과적합의 증거. **EXP-006(hidden=256)이 sweet spot** — 채택하지 않고 이전으로 복귀.

---

### EXP-008: class_weights balanced (베스트)

**가설**: 데이터 분석에서 conf≥4 필터 후 클래스 불균형이 7.0x → 11.2x로 악화됨을 확인했다. 손실이 다수 클래스에 지배되어 희귀 클래스 성능이 저하되고 있을 것이다. balanced class_weights (`N/(K·count)`)를 CrossEntropy에 적용하면 희귀 클래스를 더 잘 학습하고 macro 2nd가 오를 것이다.

**변경점**: `--class_weights balanced`. EXP-006 설정 위에 누적.

**결과**:

| H-Acc | std | vs 베이스 | vs EXP-006 |
|-------|-----|-----------|-----------|
| **85.29%** | **±0.63** | **+5.84%** | +0.86% |

**해석**: 가설 일치. H-Acc +0.86%, 특히 Macro 2nd가 79.76% → **81.16%** (+1.40%)로 크게 올랐다 — 불균형이 진짜 병목이었음을 확인. std도 ±0.29 → ±0.63으로 소폭 증가했지만 여전히 충분히 안정적이다. **최종 베스트 설정으로 채택.**

---

### EXP-009: TTA (Test-Time Augmentation)

**가설**: 추론 시 Gaussian noise를 5회 적용하고 평균 내면 예측이 더 안정적으로 개선될 것이다.

**변경점**: 추론 단계에서 Gaussian noise σ=1e-3를 5회 적용, softmax 출력 평균. 학습 설정은 EXP-008과 동일.

**결과**:

| H-Acc | std | vs 베이스 | vs EXP-008 |
|-------|-----|-----------|-----------|
| 85.29% | ±0.66 | +5.84% | 0.00% |

**해석**: 가설 불일치 — 개선 없음. CLAP 임베딩이 이미 L2-norm=1.0으로 정규화돼 있어 모델이 충분히 robust한 상태다. 노이즈를 추가해도 임베딩 공간에서 크게 이동하지 않아 TTA 효과가 없다. **채택하지 않음.**

---

## 5. 핵심 인사이트

**1. 라벨 노이즈 제거가 최고 ROI**

conf≥4 필터링 하나로 +3.97%를 얻었다. 데이터 품질이 모델 아키텍처 개선보다 훨씬 효과적이었다. DCASE 이후 대회에서도 confidence/annotation quality를 먼저 분석하는 것이 우선순위가 되어야 한다.

**2. hidden=256이 현재 데이터 규모의 sweet spot**

128(underfit, std ±1.21) ↔ **256(best, std ±0.29)** ↔ 512(overfit, std ±1.23). conf 필터 후 train ~4,000개 규모에서 512는 과적합을 일으키고 오히려 불안정해진다. 데이터가 더 늘어나면(BSD35k-CS 통합 후) 512를 재시도할 가치가 있다.

**3. 필터링이 불균형을 악화시킨 사실이 EXP-008의 근거**

conf≥4 필터가 희귀 클래스에서 더 많이 제거되어 불균형이 7.0x → 11.2x로 악화됐다는 사전 분석이 없었다면 EXP-008은 시도하지 않았을 것이다. 데이터 분석 → 가설 → 실험 사이클이 실제로 작동했다.

**4. 계층 손실의 진짜 가치는 안정성**

H-Acc 자체 개선(+0.60%)보다 std 감소와 latent space 구조화가 더 중요한 효과다. 이후 BSD35k-CS 같은 노이즈 있는 데이터를 추가할 때 계층 구조가 정규화 역할을 할 가능성이 높다.

**5. TTA는 임베딩 기반 모델에 부적합**

raw waveform 모델에서 TTA는 효과적이지만, L2-정규화된 사전추출 임베딩 위에서는 작은 perturbation이 결정 경계를 바꾸지 못한다. 임베딩 기반 파이프라인에서 TTA를 쓰려면 오디오 레벨(원본 파일)에서 augmentation 후 임베딩을 재추출해야 의미가 있다.

---

## 6. 최종 베스트 모델 (EXP-008)

### fold별 상세 결과

| Fold | Accuracy | Top Acc | Macro 2nd | H-Acc | H-F1 |
|------|----------|---------|-----------|-------|-------|
| 0 | 87.33% | 93.85% | 81.62% | 85.37% | 83.94% |
| 1 | 88.05% | 93.40% | 81.32% | 85.00% | 84.26% |
| 2 | 86.95% | 93.26% | 79.76% | 84.40% | 82.75% |
| 3 | 87.02% | 93.70% | 82.29% | 86.42% | 84.51% |
| 4 | 87.02% | 93.91% | 80.77% | 85.25% | 83.03% |
| **avg** | **87.27%** | **93.62%** | **81.16%** | **85.29%** | **83.70%** |

H-Acc 범위: 84.40% ~ 86.42% (std ±0.63) — fold 간 분산이 매우 작아 안정적인 설정이다.

### 실행 명령

```bash
cd "C:\Users\solok\Desktop\Dcase baseline\dcase2026_task1_baseline"
python train_test.py \
  --exp_name exp_008_best_classweights \
  --modes both \
  --conf_threshold 4 \
  --hier_loss \
  --lambda_top 0.3 \
  --lambda_contr 0.1 \
  --tau 0.07 \
  --hidden_size 256 \
  --dropout 0.1 \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001 \
  --k_folds 5 \
  --class_weights balanced
```

### 적용된 개선 누적 (EXP-008 기준)

| # | 변경 | H-Acc 기여 |
|---|------|-----------|
| 1 | conf≥4 필터링 | +3.97% |
| 2 | 계층 손실 (λ_top=0.3, λ_contr=0.1, τ=0.07) | +0.60% |
| 3 | hidden_size 128 → 256 | +0.41% |
| 4 | class_weights=balanced | +0.86% |
| **합계** | | **+5.84%** |

---

## 7. 다음 단계 (Phase 4-5 미실행 항목)

현재까지 Phase 2~3 초반까지 완료됐다. 아래는 우선순위 순으로 정리한 미실행 항목이다.

### 하이퍼파라미터 최적화 (Phase 4)

**Optuna로 계층 손실 파라미터 탐색 (EXP-011)**

λ_top, λ_contr, τ를 고정값(0.3 / 0.1 / 0.07)으로 쓰고 있는데, 이 값이 현재 데이터셋에 최적인지 확인이 안 됐다. Optuna로 탐색하면 +0.3~0.5% 추가 이득이 가능하다.

```python
# 탐색 범위 예시
lambda_top:   [0.1, 0.5]
lambda_contr: [0.05, 0.3]
tau:          [0.05, 0.2]
```

**학습 안정성 파라미터 탐색 (EXP-012)**

lr, weight_decay, batch_size의 최적 조합을 탐색한다. 현재 lr=1e-3은 초기 실험 기본값 그대로다.

**CosineAnnealingLR 적용 (EXP-013)**

현재 StepLR(step=20, γ=0.5)을 CosineAnnealingLR로 교체하면 수렴 안정성이 개선될 수 있다.

### 데이터 파이프라인 강화 (Phase 2 미완)

**Focal Loss (EXP-004)**

class_weights로 불균형에 대응했지만, Focal Loss는 hard example에 추가 집중하는 효과가 있다. balanced weights와 결합하거나 대체하는 실험이 필요하다.

**BSD35k-CS 통합 (EXP-003)**

3배 많은 데이터지만 노이즈가 많다. weighted sampling (BSD10k:BSD35k = 3:1) + curriculum learning(초반에는 BSD10k만, 후반에 BSD35k 점진 추가) 조합으로 실험할 가치가 있다. 데이터 규모가 늘면 hidden=512도 재시도 가능하다.

### 모델 아키텍처 개선 (Phase 3 미완)

**CLAP 마지막 레이어 fine-tuning (EXP-008~009 원래 계획)**

현재 임베딩은 완전히 고정(frozen)된 상태다. CLAP의 마지막 2레이어를 낮은 lr(1e-5)로 fine-tuning하면 BSD 도메인에 적응할 수 있다. 다만 RTX 3060 메모리(12GB) 한계를 확인하고 시도해야 한다.

**Mixup augmentation**

임베딩 공간에서 두 샘플을 보간(α=0.2)하는 Mixup은 계층 손실과 함께 latent space 구조화에 시너지가 예상된다.

### 앙상블 및 최종 제출 (Phase 5)

- **5-fold 앙상블**: softmax 출력 평균 — fold 간 분산(std ±0.63)을 줄여 추가 +0.2~0.5% 기대
- **audio-only + both 모드 혼합**: 두 모달리티 조합의 다양성 활용
- **오디오 레벨 TTA**: 원본 파일에서 augmentation 후 임베딩 재추출 (현재 frozen 방식으로는 효과 없음)
- **공식 포맷 제출 파일 생성 (EXP-018)**: 마감(2026-06-15) 2주 전 시작 필요

---

*작성일: 2026-04-27 | 현재 베스트: EXP-008, H-Acc 85.29%*
