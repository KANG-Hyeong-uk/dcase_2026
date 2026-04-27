# DCASE 2026 Task 1 — Experiment Log & Documentation

이 폴더는 모든 실험 결과(`exp_NNN_*.json`)와 그 의도/근거를 기록합니다.

> **목표 지표**: Hierarchical Accuracy (H-Acc, λ=0.5 partial credit) — 대회 랭킹 1순위 메트릭.
> **베이스라인**: EXP-000 H-Acc **79.45%** (5-fold StratifiedKFold, both mode).

---

## 데이터셋 분석 결과 (2026-04-27)

### Confidence 분포 (BSD10k_metadata.csv, n=10,956)
| conf | count | 누적비율 |
|------|-------|---------|
| 1 | 106 | 1.0% |
| 2 | 749 | 7.8% |
| 3 | 3,280 | 37.7% |
| 4 | 6,045 | 92.9% |
| 5 | 776 | 100.0% |

→ **conf=4가 가장 흔함 (55%)**. conf>=5는 776개로 너무 작음(7%). **conf>=4 (6,821 샘플 보존)** 이 sweet spot.

### 클래스 불균형
- **원본 (10,956개)**: 1,204(fx-o) vs 171(fx-a) → **7.0x 비율**
- **conf>=4 적용 후 (6,821개)**: 795(fx-o) vs 71(ss-i) → **11.2x 비율** ⚠️
- conf 필터가 희귀 클래스에서 더 많이 제거함 → **클래스 가중치 또는 weighted sampler 필요**

### 상위 클래스 분포
| top_class | count | 비율 |
|-----------|-------|------|
| fx | 4,003 | 36.5% |
| is | 2,368 | 21.6% |
| m  | 1,716 | 15.7% |
| ss | 1,526 | 13.9% |
| sp | 1,343 | 12.3% |

### 임베딩 정합성 검사
- audio/text 모두 **shape (512,), float32, L2-norm = 1.0000** (이미 정규화됨)
- mean ≈ 0, std ≈ 0.0442 (= 1/√512)
- → **추가 정규화/표준화 전처리 불필요**

---

## 실험 목록

### Phase 1 — 베이스라인 (완료)

#### EXP-000: 베이스라인 재현
- **설정**: 기본 코드 그대로, both mode, 5-fold, hidden=128, dropout=0.1, CE loss
- **결과**: H-Acc **79.45%** (Macro 2nd 74.02%, Top 88.88%)
- **파일**: `exp_000_baseline.json`
- **의의**: 모든 실험의 기준점.

---

### Phase 2 — 데이터 파이프라인 강화

#### EXP-001: confidence ≥ 4 필터링
- **가설** (논문 A 검증): 저신뢰 라벨 제거 → 2nd level +9%, top level +5%
- **변경점**: `--conf_threshold 4` → 10,956 → 6,821 샘플 (62%)
- **나머지 동일**: hidden=128, plain CE, 5-fold
- **예상 H-Acc**: 84-87%
- **파일**: `exp_001_conf4.json`

---

### Phase 3 — 모델/손실 아키텍처 개선

#### EXP-005: + Hierarchical Loss (L_Top + L_Contr)
- **가설** (논문 B): `L_total = L_CE + λ_top·L_Top + λ_contr·L_Contr` 가 latent space를 구조화하여 H-Acc를 직접 최적화
- **L_Top**: 미분가능 surrogate — fine-class 확률을 top-class membership matrix로 집계 후 top label에 대한 NLL
- **L_Contr**: Khosla 2020 SupCon — L2-정규화된 z에서 동일 top-class 샘플끼리 가깝게
- **하이퍼파라미터** (논문 권장값): λ_top=0.3, λ_contr=0.1, τ=0.07
- **EXP-001 대비 추가 효과 예상**: +1-3% H-Acc
- **파일**: `exp_005_conf4_hier.json`

#### EXP-006: + hidden_size 256
- **가설**: 모델 용량 2x → 더 나은 표현력. dropout 동일 유지.
- **파일**: `exp_006_conf4_hier_h256.json`

#### EXP-007: + hidden_size 512
- **가설**: 4x 용량. 4k train split에서 과적합 가능성 — dropout 0.15로 약간 증가.
- **파일**: `exp_007_conf4_hier_h512.json`

---

### Phase 4 — 데이터 분석 기반 추가 실험

#### EXP-008: BEST + 클래스 가중치 (계획)
- **데이터 분석 발견**: conf>=4 이후 11.2x 불균형 → 베이스라인 macro 74%의 주범
- **변경점**: `class_weight = total / (n_classes * class_count)` (sklearn balanced 방식). CE 및 L_CE에 적용.
- **EXP-005~007 중 최고 모델에 적용**.
- **예상 효과**: macro 2nd-level +2-4%, H-Acc +0.5-1.5%
- **파일**: `exp_008_best_classweights.json` (계획)

#### EXP-009: 5-fold Ensemble (계획)
- **방식**: 4개 실험 중 베스트의 5개 fold 체크포인트 → softmax 평균 → 최종 예측 (re-training 없음, inference-only)
- **예상 효과**: H-Acc +0.5-1.0% (앙상블의 일반적 이득)
- **파일**: `exp_009_ensemble.json` (계획)

---

## 코드 구조

| 파일 | 역할 |
|------|------|
| `dcase2026_task1_baseline/train_test.py` | 메인 학습 진입점 (argparse) |
| `dcase2026_task1_baseline/losses.py` | `CrossEntropyLoss`, `HierarchicalLoss` |
| `dcase2026_task1_baseline/models.py` | `BaseClassifier` (HATR 아키텍처) |
| `dcase2026_task1_baseline/evaluate.py` | H-Acc, hierarchical PRF |
| `dcase2026_task1_baseline/dataset_utils.py` | `HATRDataset` |
| `DCASE2026_World1st.ipynb` | **단일 자체완결 노트북** (위 모든 코드를 셀로 합친 deliverable) |

### CLI 사용법
```bash
cd dcase2026_task1_baseline
python train_test.py \
  --exp_name exp_005_conf4_hier \
  --modes both \
  --conf_threshold 4 \
  --hier_loss \
  --lambda_top 0.3 --lambda_contr 0.1 --tau 0.07 \
  --hidden_size 128 --dropout 0.1 \
  --epochs 100 --batch_size 64 --lr 0.001 \
  --k_folds 5
```

### 주요 플래그
| 플래그 | 설명 | 기본값 |
|--------|------|--------|
| `--exp_name` | 실험 이름 (output 디렉토리 분리) | `baseline` |
| `--modes` | 학습할 모드 (`both`, `audio`, `text`) | `both audio` |
| `--conf_threshold` | confidence 최솟값 (None=필터링 안함) | None |
| `--hier_loss` | HierarchicalLoss 사용 | False |
| `--lambda_top` | L_Top 가중치 | 0.3 |
| `--lambda_contr` | L_Contr 가중치 | 0.1 |
| `--tau` | SupCon temperature | 0.07 |
| `--hidden_size` | 모델 hidden dim | 128 |
| `--dropout` | dropout rate | 0.1 |
| `--smoke_test` | 1-fold × 2-epoch 빠른 검증 | False |

---

## 결과 요약

| ID | 설명 | H-Acc | std | Macro 2nd | Top | vs 베이스 |
|----|------|-------|-----|-----------|-----|-----------|
| 000 | baseline | 79.45% | — | 74.02% | 88.88% | 0.00% |
| 001 | + conf≥4 | 83.42% | ±1.33 | 78.52% | 93.62% | +3.97% |
| 005 | + hier loss | 84.02% | ±1.21 | 79.28% | 93.62% | +4.57% |
| 006 | + hidden 256 | 84.43% | ±0.29 | 79.76% | 93.64% | +4.98% |
| 007 | + hidden 512 | 84.17% | ±1.23 | 79.61% | 93.70% | +4.72% (regression) |
| **008** | **+ class_weights=balanced (BEST)** | **85.29%** | **±0.63** | **81.16%** | 93.62% | **+5.84%** |
| 009 | EXP-008 + TTA (noise=1e-3, n=5) | 85.29% | ±0.66 | 81.15% | 93.62% | +5.84% (no gain) |

## 핵심 인사이트

1. **confidence ≥ 4 필터링이 가장 큰 단일 기여 (+3.97%)** — 라벨 노이즈 제거가 효과적.
2. **hidden_size 256 sweet spot** — 128(under) ↔ 256(best) ↔ 512(overfit). std는 ±1.21 → ±0.29 4배 안정.
3. **계층 손실(+0.60%) + 클래스 가중치(+0.86%) 시너지** — 두 가지 모두 규모는 작지만 macro 2nd-level 메트릭에 직접적 기여.
4. **TTA는 효과 없음** — 임베딩이 이미 L2 정규화되어 있고 모델이 충분히 robust.
5. **데이터 전처리 불필요** — CLAP 임베딩이 이미 L2-norm=1.0, mean≈0, std≈0.0442 정규화 상태.

## 다음 단계 (Phase 4-5 미실행 항목)

향후 추가 가능한 실험 (Phase 4-5에서 효과 클 가능성):
- **EXP-010+**: Optuna로 λ_top, λ_contr, τ 탐색 (현재는 논문 권장값 그대로)
- **EXP-011**: focal loss (γ=2.0) — 클래스 가중치보다 더 강한 hard-example mining
- **EXP-012**: mixup at embedding level (α=0.2) — train 다양성 확장
- **EXP-013**: 다른 seed로 EXP-008 재학습 → ensemble (현재는 1821 단일)
- **EXP-014**: BSD35k-CS curriculum learning
- **EXP-015**: CLAP 마지막 2-4레이어 fine-tuning (도메인 적응)
