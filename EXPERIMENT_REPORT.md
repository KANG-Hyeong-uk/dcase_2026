# DCASE 2026 Task 1 실험 보고서

**진정한 baseline: EXP-010, H-Acc 74.01% ± 2.32%** (5-fold StratifiedGroupKFold + train-only conf filter)

> ## ⚠️ 방법론 결함 발견 및 수정 (2026-04-29)
>
> 초기 실험(EXP-000 ~ EXP-009)에서 두 가지 평가 방법론 결함이 발견되어 모든 점수가 inflated였음. 본 문서는 **EXP-010(두 결함 모두 수정한 진정한 baseline)** 중심으로 재구성됨.
>
> ### 발견된 결함 2건
>
> 1. **Confidence filter scope 오류**: `--conf_threshold` 필터가 fold split *전*에 전체 데이터에 적용 → val/test도 conf≥4 깨끗한 샘플만 평가 → 실제 운영 분포 미반영
> 2. **Uploader leakage**: random `StratifiedKFold` 사용 시 같은 uploader의 비슷한 녹음이 train/test 양쪽에 포함 → 1,806 uploader 중 77.6%가 단일 클래스만 업로드해 leakage 영향 큼
>
> ### 폐기된 점수 (참고용)
>
> | EXP | 설명 (구) | inflated H-Acc |
> |-----|-----------|---------------|
> | 000 | baseline (random KFold) | ~~79.45%~~ |
> | 001 | + conf≥4 필터 (전체) | ~~83.42%~~ |
> | 005 | + 계층 손실 | ~~84.02%~~ |
> | 006 | + hidden_size 256 | ~~84.43%~~ |
> | 007 | + hidden_size 512 (regression) | ~~84.17%~~ |
> | 008 | + class_weights=balanced (구 베스트) | ~~85.29%~~ |
> | 009 | + TTA (효과 없음) | ~~85.29%~~ |
>
> **개선 방향성** (어떤 변경이 H-Acc를 올리는지)은 여전히 참고 가치 있으나, **절대 수치**는 모두 EXP-010 baseline에서 재측정 필요. 본 문서에서는 narrative를 제거하고 핵심 발견만 보존.

---

## 1. 프로젝트 배경

- **대회**: DCASE 2026 Challenge Task 1 — Heterogeneous Audio Classification
- **데이터**: BSD10k — 10,956개 sound clip, 23개 2nd-level 클래스, 5개 top-level 클래스 (Music, Instrument, Speech, FX, Soundscape)
- **입력**: 사전추출된 CLAP audio 임베딩 + text 임베딩 (각 512d, L2-norm=1.0 정규화 완료)
- **평가지표**: Hierarchical Accuracy (H-Acc) — 메인 랭킹 지표. 2nd-level 정답이면 최고점, 2nd 틀리고 top 맞으면 부분 점수, 둘 다 틀리면 0점
- **검증**: 5-fold StratifiedGroupKFold cross-validation, group=uploader (SEED=1821)
- **하드웨어**: NVIDIA RTX 3060
- **마감**: 2026-06-15

---

## 2. 데이터 분석 (방법론 결함과 무관, 사실 유효)

| 분석 항목 | 발견 |
|----------|------|
| Confidence 분포 | conf=4가 전체의 55%, conf=5는 7%. conf≥4가 62% (6,821개) 보존 |
| 클래스 불균형 (raw) | 7.0x (fx-o 1,204개 vs fx-a 171개) |
| 클래스 불균형 (conf≥4 필터 후) | **11.2x** — 필터가 희귀 클래스에서 더 많이 제거, 오히려 불균형 악화 |
| 임베딩 정합성 | L2-norm=1.0, mean≈0, std≈0.0442 → 추가 정규화 불필요 |
| Unique uploaders (conf≥4) | 1,806명 |
| 단일 클래스만 업로드한 uploader | **77.6%** — uploader = 클래스 정보 누설, GroupKFold 필수 근거 |
| Top uploader 점유율 | MTG 374 samples = 5.5% |

이 분석 결과들은 모두 **데이터셋 자체의 사실**로 fold split 방식과 무관. EXP-010 이후에도 그대로 유효.

---

## 3. 방법론 결함 진단 및 수정 (2026-04-29)

### Fix #1: confidence filter scope (train-only)

**Before (잘못됨):**
```python
# fold split 이전에 전체 필터
if args.conf_threshold is not None:
    full_df = full_df[full_df['index'].isin(high_conf['sound_id'])].reset_index(drop=True)
    # 10,956 -> 6,821
# 이후 5-fold split: train/val/test 모두 conf>=4 깨끗한 샘플만 → 평가 inflated
```

**After (수정됨):**
```python
# 1) high_conf_ids set만 미리 로드 (full_df는 그대로 10,956 유지)
high_conf_ids = None
if args.conf_threshold is not None:
    high_conf_ids = set(conf_df.loc[conf_df['confidence'] >= args.conf_threshold, 'sound_id'])

# 2) fold split (전체 10,956 기준)
# 3) fold loop 안에서 train만 필터
if high_conf_ids is not None:
    train_df = train_df[train_df['index'].astype(str).isin(high_conf_ids)].reset_index(drop=True)
    # val_df, test_df는 손대지 않음
```

**Per-fold log 검증:**
```
[conf-filter PREP] threshold>=4 | 6821 high-conf sound_ids loaded (6821/10956 samples).
                   Will be applied to TRAIN ONLY in fold loop.
[Fold 0] Train: 6476 -> 3925 (conf>=4 필터, train만)
           Val:   2075 (필터 없음, 모두 유지)
           Test:  2405 (필터 없음, 모두 유지)
```

### Fix #2: StratifiedGroupKFold by uploader

**진단:** conf≥4 필터 후 1,806 unique uploaders 중 77.6%가 단일 클래스만 업로드. random KFold 사용 시 같은 uploader가 train/test 양쪽에 들어가 평가 inflated.

**Code:**
```python
# 새 옵션
--fold_strategy {random, group, stratified_group}

# stratified_group 선택 시
meta_df = pd.read_csv(dataset_path)
full_df = full_df.merge(meta_df[['sound_id', 'uploader']], left_on='index', right_on='sound_id', how='left')
groups = full_df['uploader'].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
all_splits = list(sgkf.split(np.zeros(len(labels)), labels, groups=groups))
# inner train/val split도 GroupShuffleSplit으로 group-aware 처리
```

**검증:** 5 fold 모두 uploader overlap=0, 23 클래스 모두 보존.

---

## 4. EXP-010 — 진정한 baseline

### 모델 아키텍처

```
audio_emb (512d) ─┐
                   ├─ EmbeddingEncoder (MLP + Residual×3) × 2
text_emb  (512d) ─┘
                   ↓
            AttentionFusion (α_audio, α_text 동적 결정)
                   ↓
           latent_projector (hidden=256)
                   ↓
         ResidualBlock × 2
                   ↓
          23-class predictor
```

### 학습 설정

| 항목 | 값 | 비고 |
|------|----|------|
| 손실 | `L_total = L_CE + 0.3·L_Top + 0.1·L_Contr` (τ=0.07) | 계층 손실 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) | |
| 스케줄러 | StepLR (step=20, γ=0.5) | |
| Augmentation | Gaussian noise σ=1e-4 + random feature masking 70% (train only) | |
| hidden_size | 256 | |
| dropout | 0.1 | |
| batch_size | 64 | |
| epochs | 100 (early stop 5×3) | |
| class_weights | balanced (`N/(K·count)`) | 클래스 불균형 대응 |
| **conf_threshold** | **4 (train만 적용)** | **Fix #1** |
| **fold_strategy** | **stratified_group (uploader)** | **Fix #2** |

### 실행 명령

```bash
cd "C:\Users\solok\Desktop\Dcase baseline\dcase2026_task1_baseline"
python train_test.py \
  --exp_name exp_010_groupkfold \
  --modes both \
  --conf_threshold 4 \
  --hier_loss --lambda_top 0.3 --lambda_contr 0.1 --tau 0.07 \
  --hidden_size 256 --dropout 0.1 \
  --epochs 100 --batch_size 64 --lr 0.001 \
  --k_folds 5 --class_weights balanced \
  --fold_strategy stratified_group
```

### 결과 (5-fold avg ± std)

| 지표 | 값 |
|---|---|
| **Hierarchical Accuracy** | **74.01% ± 2.32%** |
| Top-level Accuracy | 85.22% ± 2.53% |
| Macro 2nd Accuracy | 66.99% ± 2.69% |
| Macro Top Accuracy | 81.02% ± 2.01% |
| Hierarchical F1 | 71.62% ± 2.11% |
| Hierarchical Precision | 72.57% ± 1.63% |
| Hierarchical Recall | 72.25% ± 2.41% |

### Fold별 상세

| Fold | Train (filtered) | Val | Test | Accuracy | Top Acc | Macro 2nd | H-Acc | H-F1 |
|------|------------------|-----|------|----------|---------|-----------|-------|-------|
| 0 | 8,551 → ~4,900 | ~1,200 | 2,405 | 75.47% | 86.44% | 70.77% | **76.97%** | 73.68% |
| 1 | 8,713 → ~5,000 | ~1,200 | 2,243 | 68.84% | 80.43% | 62.71% | **70.03%** | 67.78% |
| 2 | 8,851 → ~5,100 | ~1,200 | 2,105 | 73.78% | 87.41% | 66.71% | **74.35%** | 71.89% |
| 3 | 8,881 → ~5,100 | ~1,200 | 2,075 | 72.43% | 84.96% | 66.11% | **73.34%** | 71.36% |
| 4 | 8,828 → ~4,900 | ~1,100 | 2,128 | 75.70% | 86.84% | 68.66% | **75.35%** | 73.40% |
| **avg** | | | | **73.24%** | **85.22%** | **66.99%** | **74.01%** | **71.62%** |
| **std** | | | | ±2.50% | ±2.53% | ±2.69% | ±2.32% | ±2.11% |

### 핵심 관찰

**1. 두 fix 합산 효과: H-Acc inflation 약 11%p**

이전 inflated 점수(EXP-008 85.29%) 대비 EXP-010 74.01% — 차이 −11.28%. 이 격차의 원천:
- Test set이 conf<4 노이즈 샘플 포함 (~1.6배 커짐) → 실제 운영 환경 반영
- Uploader leakage 차단 → 모델이 외운 uploader 패턴이 평가에 새지 않음

**2. fold 분산이 4배 증가 (±0.63 → ±2.32)**

EXP-008 inflated의 ±0.63 안정성은 *깨끗한 test + uploader 누설*에 기댄 인위적인 결과. EXP-010 ±2.32가 진짜 모델 성능의 분산. **Fold 1만 H-Acc 70.03%**로 다른 fold(73~77%) 대비 4~7%p 낮음 — 해당 fold의 uploader 분포가 가장 어려운 케이스. EXP-019 분석 1순위.

**3. Macro 2nd 더 큰 충격 (−14.17%, vs Top −8.40%)**

fine-class에서 충격이 가장 크다. 클래스 불균형 + uploader leakage가 fine-class에서 증폭됨. → EXP-022 (dual-head + parent-aware smoothing)로 직격 회복 필요.

**4. Top accuracy −8.40%**

5-class 분류는 비교적 견고. 부풀려진 부분이 주로 "외운 fine-class 라벨"에서 옴.

### 실행 환경

- branch: main, commit `5b0c0e4` (코드), `b4988db` (결과)
- timestamp: 2026-04-29
- exp_name: `exp_010_groupkfold` (model_output 디렉토리명)
- 결과 파일: `experiments/exp_010_true_baseline.json`

---

## 5. 누적 적용된 개선 (방향성 참고)

EXP-010에 누적 적용된 변경점들. 절대 효과 크기는 EXP-010 이후 ablation으로 재측정 필요하지만, 방향성은 직관적으로 유효:

| # | 변경 | 근거 |
|---|------|------|
| 1 | confidence ≥ 4 필터 (train만) | 논문 A — 라벨 노이즈 제거 |
| 2 | 계층 손실 `L_CE + 0.3·L_Top + 0.1·L_Contr (τ=0.07)` | 논문 B 수식 |
| 3 | hidden_size 128 → 256 | 분산 안정화 (구 데이터에서 ±1.21 → ±0.29 4배 안정 — 단 EXP-010에서는 ±2.32) |
| 4 | class_weights = balanced (`N/(K·count)`) | 11.2x 불균형 해소 |
| 5 | StratifiedGroupKFold by uploader (Fix #2) | uploader leakage 차단 |

채택 안 한 변경:
- ❌ hidden_size 512: 구 데이터에서 과적합 — EXP-010 환경에선 train ~5,000개로 늘어 재시도 가치 있음
- ❌ TTA (Gaussian noise σ=1e-3, n=5): L2-정규화 임베딩에서 효과 없음

---

## 6. 다음 단계 (EXP-010 baseline 기준 우선순위)

### 즉시 (학습 없이 가능, ROI 최고)

**EXP-019: uploader 단위 분석 + 블랙리스트**
- Fold 1 H-Acc 70.03% 하락 원인 추적
- 저품질 uploader (val acc 낮은) 식별 → 블랙리스트 처리
- conf 필터와 독립적 추가 노이즈 제거 수단
- 예상 +1~2% H-Acc

**Macro hF threshold tuning (post-processing)**
- per-class threshold를 Optuna로 macro hF 최적화
- 학습 결과(`predictions.csv`) 위에서 수초만에 실행
- 예상 +0.5~1.5%

### 단기 (1~2시간 학습)

**EXP-022: dual-head + conditional decoding + parent-aware smoothing**
- top-class head + 2nd-class head 병렬 출력 (`L = L_child + 0.5·L_parent`)
- inference 시 `P(child) = P(parent)·P(child|parent)` 조건부 디코딩
- sibling=0.05, cross-parent=0.0인 parent-aware label smoothing
- Macro 2nd 회복 직격탄, 예상 +1.5~3%

**GCE/SCE noise-robust loss**
- FSDnoisy18k 결론: conf<4 train에서 GCE(q=0.7)가 CE 대체 가능
- Symmetric CE (α=0.1, β=1.0) 비교 ablation
- 예상 +0.5~1.5%

### 중기 (raw audio 전환)

- BEATs/HTSAT/EfficientAT 마지막 2 block fine-tune (RTX 3060 bs=16~24)
- SpecAugment + Mixup + bg noise mix
- BSD35k-CS 통합 (weighted sampling 3:1, GroupKFold 재구성)
- 예상 +3~7% (단일 모델)

### 최종 (마감 2주 전부터)

- 8-model 5-fold ensemble (BEATs×2 + HTSAT + EfficientAT + EXP-010 + dual-head + 2-stage + text-only)
- per-class blending Optuna
- TTA (raw audio 5-crop avg)
- 최종 제출 파일 생성 (2026-06-15 마감)

---

*최종 갱신: 2026-04-29 | 진정한 baseline: EXP-010, H-Acc 74.01% ± 2.32%*
