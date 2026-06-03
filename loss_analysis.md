# Loss Analysis for DCASE 2026 Hierarchical Audio Classification

> Reviewer 관점 노트 — 이 리포트는 일반론이 아니라 **당신의 실제 모델(`BaseClassifier`), 실제 confusion matrix(`v4_35k_602020_pureval/cm_foldavg_baseline_b0.csv`), 실제 평가지표(H-Acc, λ=0.5), 5-top/23-sub 계층, CLAP 멀티모달**에 한정해 세 손실을 심사한다. 모든 결론은 측정된 오류 패턴에 근거를 둔다.

---

## 1. Problem Understanding

### 1.1 모델이 실제로 내보내는 것 (코드 확인 결과)

`BaseClassifier(hidden_size=128)` 기준 ([models.py](dcase2026_task1_baseline/models.py)):

- `latent_projector` 최종 출력 = `hidden//2 = 64`
- `residual_classifier`(2 blocks, BatchNorm+LeakyReLU) 통과 후의 텐서가 곧 반환되는 `z` → **z는 64D, 그리고 정규화되지 않은 post-activation 벡터**
- `class_predictor`: `64 → 32 → 23`. 즉 **z는 분류기 입력 그 자체**다.

> ⚠️ **프롬프트 본문과 코드의 불일치**: 본문은 "z = 128D, Latent Projector 256→128"이라 적었으나 실제는 z=64D다. 다행히 제출한 `HierarchicalProxyLoss`의 `embedding_dim=64` 기본값은 실제와 맞다. 만약 본문 서술을 믿고 `embedding_dim=128`로 설정하면 `z·proxy.T`에서 즉시 shape 에러가 난다. **모든 metric/proxy/contrastive 손실은 64D 기준으로 설계해야 한다.**

이 사실의 함의가 분석 전체를 지배한다: **z를 건드리는 손실(A의 SupCon, B의 proxy)은 분류기의 입력 분포를 직접 재성형한다.** z는 별도의 표현 공간이 아니라 분류기 바로 앞단이므로, 계층 구조를 z에 강제하면 23-way 분리능과 직접 충돌할 수 있다. 반대로 logits만 건드리는 손실(C)은 이 충돌이 없다.

### 1.2 실제 오류 구조 (baseline_b0, 5-fold 평균 행정규화 CM)

측정된 오류는 **두 개의 서로 다른 가족**으로 깔끔하게 갈린다.

**가족 ① — Cross-top 누수 (H-Acc를 가장 크게 깎는 오류, λ=0.5에서 ~2배 페널티):**

| GT | 잘못 간 곳 | 비율 | 성격 |
|----|-----------|------|------|
| fx-a | **ss-n** | **0.4485** | fx→ss (치명적) |
| fx-v | **ss-u** | **0.2561** | fx→ss |
| ss-n | **fx-n** | **0.1923** | ss→fx |
| sp-c | fx-h / ss-u / ss-i | 0.171 / 0.157 / 0.136 | sp→fx, sp→ss |
| fx-n | ss-n | 0.0341 | fx→ss |

→ **fx와 ss top-class가 음향적으로 심하게 엉켜 있다.** fx-a는 절반 가까이 ss-n으로 샌다. sp-c는 전방위로 cross-top 누수. 이것이 H-Acc의 1차 병목이다.

**가족 ② — Within-top sibling 혼동 (Macro Acc를 깎지만 H-Acc는 부분점수로 방어됨):**

| GT | recall | 형제 혼동 |
|----|--------|----------|
| m-m | 56.25% | →m-sp 0.219, →m-si 0.172 |
| ss-i | 33.54% | →ss-u 0.268 |
| is-e | 70.56% | is 내부 분산 |
| sp-c | 41.43% | →sp-s 0.10 (+ ①의 cross-top) |
| fx-ex | 42.50% | →fx-h 0.175 |

→ 같은 부모 아래 형제끼리의 혼동. 이건 top은 맞으므로 H-Acc 손실이 작고 **Macro Acc만 깎는다.**

### 1.3 핵심 통찰 (심사의 축)

> **이 문제는 단일 병목이 아니라 직교하는 두 병목을 가진다.**
> - H-Acc ← 가족① (cross-top fx↔ss, sp-c) 를 줄여야 오른다.
> - Macro Acc ← 가족② (sibling) 를 줄여야 오른다.
>
> **세 손실은 이 두 병목에 서로 다르게 작용한다.** 그리고 결정적으로, **top-level 응집(A의 SupCon)과 sibling 분리(B의 sibling-sep)는 방향이 반대다.** A는 형제를 끌어모으고(top 응집), B는 형제를 떼어놓는다. 이 긴장이 세 손실의 운명을 가른다. 어떤 손실도 "둘 다 공짜로" 잡지 못한다.
>
> 멀티모달은 이미 audio-only/text-only를 크게 능가하므로 모달 붕괴는 병목이 아니다. 따라서 손실 설계는 **표현 붕괴 방지가 아니라 계층 기하 구조 정렬**에 집중해야 한다.

---

## 2. Analysis of Loss A — Hierarchical CE + Top Loss + SupCon

`L_total = L_CE(23) + λ_top·L_top + λ_contr·L_contr`, top-level positives 기준 SupCon. (현재 `hier_loss_LTop_LContr.ipynb`에 구현되어 있음.)

### Assumptions
- 같은 top의 샘플은 z 공간에서 가까워야 한다(= 형제를 한 덩어리로 본다).
- 배치 안에 top당 양성쌍이 충분하다. RTX 3060, batch=64에서 fx(8 sub)·ss(4 sub)는 양성 풍부, 그러나 sp(3)·m(3)·is(5)는 배치에 따라 양성이 1~2개로 희박해져 SupCon gradient가 noisy.
- z를 L2 정규화해야 한다(현재 z는 비정규화 post-BN 벡터 → forward에서 `F.normalize` 필수).

### Strengths
- **가족① 직격**: SupCon이 fx 클러스터와 ss 클러스터를 통째로 떼어놓으면 fx-a→ss-n(0.4485), fx-v→ss-u(0.256), ss-n→fx-n(0.192)가 구조적으로 줄어든다. H-Acc 병목과 정확히 정렬.
- `L_top`(subclass 확률을 top으로 합산한 NLL)은 top 정확도를 직접 끌어올린다 → H-Acc의 분모 항을 직접 개선.
- **이미 구현·검증(논문 B)되어 한계비용이 0에 가깝다.** 가장 먼저 돌릴 수 있다.

### Weaknesses
- **가족②와 정면 충돌.** top-level SupCon은 fx 8형제를 한 점으로 끌어모은다. 이는 fx-ex↔fx-h, ss-i↔ss-u 분리를 *악화*시킬 수 있다. z가 분류기 입력이므로(§1.1) 이 응집이 23-way 분리능을 직접 잠식한다.
- λ_contr가 크면 z가 5개 덩어리로 붕괴 → Macro Acc 하락. λ_contr=0.1은 보수적이지만 형제 희박 클래스에서 위험.
- SupCon은 배치 통계에 민감 → 64D·batch64·소수 top에서 분산 큼. fold 간 변동성 증가 가능.

### Expected Effect on Confusion Matrix
- ↓ fx↔ss 블록(가족①) — 기대 가장 큰 개선 지점.
- = 또는 ↑ within-fx / within-ss 블록(가족②) — ss-i↔ss-u, fx-ex↔fx-h가 그대로거나 악화될 수 있음.

### Expected Effect on Hierarchical Metrics
- **상승 가능성 높음.** cross-top을 직접 공략하고 top NLL을 더하므로 H-Acc/H-F1에 가장 정렬된 "표현 기반" 접근.

### Expected Effect on Macro Accuracy
- **중립~소폭 하락.** sibling 응집 부작용. λ_contr 튜닝으로 완화는 가능하나 본질적 trade-off.

### Risks
- z 비정규화 시 τ=0.07 스케일 깨짐(흔한 버그).
- λ_contr 과대 → top 5덩어리 붕괴 → 23-way 붕괴.
- 소수 top의 noisy SupCon → fold 변동성.

---

## 3. Analysis of Loss B — Hierarchical Proxy Loss (제출 코드)

5개 항: child-cls(proxy), parent-cls(proxy), child→parent 정렬, sibling 분리(margin 0.4), parent 분리(margin 0.0). proxy는 학습 파라미터.

### 제출 코드 자체에 대한 비평 (이게 심사의 절반이다)

1. **이중 분류 경로 충돌 (가장 심각).** 이 손실의 `child_logits = z·child_proxies/τ`는 사실상 **기존 `class_predictor`를 대체하는 두 번째 분류 헤드**다. 그런데 당신의 `evaluate_model`은 모델의 `class_logit`(=`class_predictor` 출력)을 쓴다. 그러면:
   - (a) 학습은 proxy 헤드로, 추론은 `class_predictor`로 → **train/test 불일치**. proxy가 정렬한 z를 `class_predictor`가 동일하게 해석한다는 보장 없음.
   - (b) 또는 추론도 proxy argmax로 → `evaluate_model`·`write_confusion_matrix`·체크포인트 포맷 전면 수정 필요. "baseline 코드 파괴 금지" 원칙과 충돌.
   - → **둘 중 하나를 명시적으로 결정하기 전까지 Loss B는 미정의 상태다.** 가장 안전한 절충은 `L_CE(class_predictor)` 를 **유지**하고 proxy 5항을 보조 정규화로 더하는 것 — 그러나 그러면 항이 6개가 된다.

2. **z 정규화 누락.** docstring은 "Normalized embeddings"라 가정하나 `forward` 안에서 `z`를 정규화하지 않는다(proxy만 정규화). 호출부에서 `F.normalize(z, dim=1)`를 반드시 넣어야 τ 스케일이 의미를 가진다.

3. **기하학적 긴장 (64D에서 위험).** `child→parent 정렬`은 형제 8개(fx)를 같은 parent proxy로 끌어당기고, 동시에 `sibling 분리(cos ≤ 0.4)`는 그 8개를 서로 떼어놓으라 한다. "같은 점 근처에 있되 서로 0.4 이상 떨어져라" — 64D에서 8개는 가능하지만 margin/weight 균형이 조금만 어긋나도 진동·미수렴. parent_margin=0.0은 5 parent를 거의 직교로 미는 강한 제약(5≤64라 가능은 함).

4. **하이퍼파라미터 폭발.** α,β,γ,δ + sibling_margin + parent_margin + τ = **7개**를 소규모 BSD10k(train_pool 8764)에서 동시 튜닝. proxy(28벡터×64=1792 파라미터)는 작아 과적합은 덜하나, **튜닝 비용이 A·C 대비 압도적.** Optuna 없이는 사실상 운에 맡기는 셈.

### Assumptions
- 계층이 proxy 벡터들의 명시적 기하(parent 중심 + child 위성)로 표현 가능하다.
- z가 metric space로 적합하다(정규화·적절 τ 전제).

### Strengths
- **유일하게 가족①과 가족②를 동시에 겨냥한다.** `parent 분리`→cross-top↓(가족①), `sibling 분리`→형제 혼동↓(가족②). 이론상 H-Acc와 Macro를 둘 다 올릴 수 있는 **유일한** 후보.
- 성공 시 latent 기하가 가장 해석가능(t-SNE에서 parent 클러스터 안에 분리된 child 위성) → **연구 스토리·publishability 최고.**

### Weaknesses
- 위 코드 비평 1~4. 특히 **이중 헤드 문제와 7-HP 튜닝**이 실전 도입의 벽.
- 가장 구현·디버깅 비용이 크고 fold 변동성·미수렴 위험 최고.

### Expected Effect on Confusion Matrix
- 성공 시: ↓ cross-top(fx↔ss) **그리고** ↓ sibling(ss-i↔ss-u, m-m↔형제). 이상적이면 두 블록 모두 개선.
- 실패 시: proxy 미정렬로 전반적 잡음 증가, 특정 fold 붕괴.

### Expected Effect on Hierarchical Metrics
- **고분산.** 잘 되면 A보다 높을 수 있으나, 기대값(평균)은 튜닝 리스크 때문에 A·C보다 불확실.

### Expected Effect on Macro Accuracy
- **세 손실 중 Macro 상승 잠재력 1위** (명시적 sibling 분리는 A·C에 없는 고유 항). 가족②를 직접 공략하는 유일한 손실.

### Risks
- 이중 분류 경로 미정의(치명) / z 정규화 누락 / 7-HP / 64D 기하 긴장 / 소규모 데이터에서 proxy 콜드스타트.

---

## 4. Analysis of Loss C — Hierarchical Cost Matrix Loss

cost(within-top)=1, cost(cross-top)=5 형태의 비용 행렬로 예측을 직접 페널티. 보통 expected-cost `L = Σ_j p_j·cost(y,j)` 또는 cost-weighted CE로 구현. **logits만 사용 — z 불변.**

### Assumptions
- 평가지표 H-Acc가 본질적으로 비용 기반(cross-top 2배 페널티, λ=0.5)이라는 사실. → **cost matrix는 H-Acc 지표 자체의 학습용 surrogate다.**
- within-top 형제 간 비용은 모두 1로 동일(=형제 구분을 비용으로는 강제하지 않음).

### Strengths
- **아키텍처 적합성 만점.** 기존 23 logits에 그대로 적용, 새 파라미터·새 헤드·z 조작·정규화 전부 불필요. `CrossEntropyLoss`의 드롭인 교체. baseline 코드 보존 원칙과 완벽 호환.
- **H-Acc와 가장 직접적으로 정렬.** 지표를 미분가능 surrogate로 직접 최소화 → 단위 노력당 H-Acc 상승 확률 최고.
- HP가 사실상 cost 비율 1개 → 튜닝 부담 최소.

### Weaknesses
- **지표 게이밍 위험.** cross-top이 비싸지면 모델은 fx-a를 "맞히려" 하기보다 **같은 top의 쉬운 형제(fx-h 등)로 옮겨** top만 지킨다. 즉 fx-a→ss-n(cost5)을 fx-a→fx-h(cost1)로 바꿔 H-Acc는 오르지만 **fx-a 진짜 recall은 안 오른다.**
- within-top 비용이 평평 → **가족②(sibling)에 무력.** Macro Acc 개선 기여 거의 없음, 오히려 희귀 형제를 다수 형제로 흡수해 **Macro 하락 가능.**
- 비용비가 과격(5:1)하면 어려운 클래스(fx-a, sp-c, ss-i)를 포기하고 top만 지키는 degenerate 해로 수렴.

### Expected Effect on Confusion Matrix
- ↓ cross-top 블록(특히 fx-a→ss-n, sp-c→ss/fx) — 오류가 **top 내부로 재배치**됨.
- ↑ within-top 블록 가능 — cross-top에서 밀려난 질량이 형제로 이동(fx-a→fx-h 증가 등).

### Expected Effect on Hierarchical Metrics
- **상승 확률 가장 높음(가장 직접적).** 단, 진짜 분류력 향상이 아니라 지표 정렬에 의한 상승일 수 있음 — 해석 시 주의.

### Expected Effect on Macro Accuracy
- **중립~하락.** 형제 구분에 무력하고 희귀 클래스를 다수 형제로 흡수할 수 있음.

### Risks
- 지표 게이밍(진짜 recall 정체) / 과격 비용비 시 희귀 클래스 포기 / Macro 하락.

---

## 5. Comparison Table

| 기준 | Loss A (CE+Top+SupCon) | Loss B (Hierarchical Proxy) | Loss C (Cost Matrix) |
|------|----|----|----|
| **Architectural Compatibility** | 중 (z 정규화 필요, z=분류기 입력 충돌) | **낮음** (이중 분류 헤드, eval/ckpt 수정) | **높음** (logits만, 드롭인) |
| **Hierarchical Awareness** | 높음 (top 응집) | **최고** (parent+child+sibling 명시) | 높음 (비용으로 명시) |
| **Expected H-Acc Gain** | 중~상 (안정) | 상 but **고분산** | **상 (가장 직접)** |
| **Expected Macro Gain** | 중립~하락 | **상 (sibling 분리 고유)** | 중립~하락 |
| **Implementation Complexity** | 낮음 (이미 구현) | **높음** | **최저** |
| **# Hyperparameters** | 3 (λ_top,λ_contr,τ) | **7** | 1 (cost 비율) |
| **Risk Level** | 중 | **높음** | 중 (게이밍) |
| **Research Novelty** | 낮음 (논문 B 재현) | **높음** | 최저 (고전 cost-sensitive) |
| **Publishability** | 중 | **높음** | 낮음 |
| **두 병목 커버** | ①만 | **①+②** | ①만 |

---

## 6. Recommended Experiment Order

데이터 조건은 세 실험 모두 `v4_35k_baseline모델` 동일 세팅(80/20 holdout, KFold5 on combined, global v4_all/ge2/ge3/ge4)으로 고정해 **손실 효과만 분리**한다. 비교 기준선은 동일 데이터의 CE 결과(`outputs/v4_35k_baseline_model/`).

1. **[Step 0] 기준 고정** — CE(baseline_b0) H-Acc·Macro·CM을 fold별로 동결. fx-a recall, fx-a→ss-n 누수, ss-i→ss-u, Macro를 *추적 지표*로 등록.
2. **[Step 1] Loss C 먼저 (저비용 고확률).** cost 비율을 보수적으로(예 within=1, cross=2~3, **5는 게이밍 위험으로 1차 회피**). H-Acc 상승 + **fx-a 진짜 recall 동시 추적** — H-Acc만 오르고 recall 정체면 게이밍 확정. 1일 내 신호 확보.
3. **[Step 2] Loss A (이미 구현).** `hier_loss_LTop_LContr.ipynb` 실행. z `F.normalize` 적용 여부 먼저 코드 확인. λ_contr ∈ {0.05, 0.1, 0.2} 소탐색. **Macro가 떨어지면 sibling 응집 부작용 확정** → λ_contr 하향.
4. **[Step 3] C+A 결합 시험.** cost-CE(top 정렬) + SupCon(cross-top 표현 분리)는 **상보적**(둘 다 가족① 공략, 서로 다른 메커니즘). 가장 현실적인 "H-Acc 최대화" 조합 후보.
5. **[Step 4] Loss B는 마지막, 그리고 조건부.** 먼저 **이중 헤드 결정**(권장: `class_predictor`+`L_CE` 유지, proxy 5항을 보조 정규화로 추가)과 **z 정규화**를 코드로 못 박은 뒤, Optuna로 7-HP를 탐색. A·C에서 Macro가 끝내 안 오를 때만 투자. 성공 시 t-SNE 시각화로 publishability 확보.

> 원칙 재확인: **어떤 전략도 동일 데이터 CE baseline의 H-Acc를 넘지 못하면 실패로 간주.** 특히 Loss C/A가 Macro를 깎으면서 H-Acc만 올리는 경우, "지표 게이밍 vs 진짜 개선"을 fx-a recall로 판별할 것.

---

## 7. Final Ranking (당신의 문제 한정)

1. **Loss C (Cost Matrix)** — 단위 노력당 H-Acc(랭킹 지표) 상승 확률 1위, 통합 비용 최저, 가족① 직격. *단, Macro·진짜 recall 게이밍을 반드시 감시.*
2. **Loss A (CE+Top+SupCon)** — 이미 구현, 안정적, cross-top 표현 분리로 H-Acc 정렬. C와 상보적이라 결합 가치 높음.
3. **Loss B (Hierarchical Proxy)** — 잠재 상한(H-Acc+Macro 동시, 최고 novelty)은 최고지만, 이중 헤드 미정의·7-HP·64D 기하 긴장으로 **기대값과 분산이 가장 불리**. 고위험 고보상.

### 가장 실패 확률 높은 손실 — **Loss B**, 그 이유
- (1) 제출 코드가 기존 `class_predictor`와 **이중 분류 경로**를 만들어 train/test 정합성이 미정의. (2) z 정규화 누락 시 τ 무의미. (3) 64D에서 child-parent 정렬 ↔ sibling 분리의 기하 긴장. (4) 8764 샘플에서 7-HP 동시 최적화. — 이 넷이 동시에 해결되지 않으면 평균적으로 CE baseline조차 밑돌 위험.

---

## 8. Final Recommendation

- **가장 먼저 구현할 손실** → **Loss C**. 드롭인 교체, 1-HP, H-Acc 직격. 가장 빠르게 "계층 손실이 이 데이터에서 작동하는가"의 신호를 준다. (단, 비용비는 2~3부터, 5는 게이밍 회피.)
- **H-Acc 상승 확률이 가장 높은 손실** → **Loss C** (지표 자체의 surrogate). 단 *진짜* 분류력 향상까지 원하면 **A**가 더 정직하게 cross-top을 표현 수준에서 줄인다.
- **Macro Acc 상승 확률이 가장 높은 손실** → **Loss B** (명시적 sibling 분리 — A·C엔 없는 고유 항). 가족②(ss-i↔ss-u, m-m↔형제, sp-c↔sp-s)를 직접 공략하는 유일한 후보.
- **Confusion matrix 구조를 가장 잘 개선할 손실** → 성공 시 **Loss B**(두 블록 동시), 안정적으로는 **Loss A**(cross-top 블록을 표현 수준에서 정리).
- **가장 publishable한 방향** → **Loss B** (멀티모달 CLAP + 학습된 계층 proxy 기하, t-SNE로 parent-child 구조 시각화 — DCASE/metric-learning 스토리). A는 논문 B 재현이라 신규성 약함, C는 고전.
- **내가(reviewer로서) 직접 고른다면** → **C로 빠른 H-Acc 신호 확보 → A와 결합(C-CE + top-SupCon) → 그 위에서만 B를 조건부 투자.** 이유: 당신의 랭킹 지표는 H-Acc이고 1차 병목은 cross-top(fx↔ss, sp-c)이며, 그걸 가장 싸고 직접적으로 미는 게 C, 표현 수준에서 보강하는 게 A다. B의 상한은 매력적이나, **이중 헤드 정의와 7-HP를 먼저 못 박지 않으면 평균적으로 baseline을 밑돈다.** 고위험 카드를 마지막에 두는 것이 합리적이다.

> 마지막 경고 (reviewer): 세 손실 모두 "H-Acc는 올랐는데 Macro와 fx-a/sp-c/ss-i 같은 **희귀·어려운 클래스의 진짜 recall이 정체/하락**"하는 함정을 공유한다. 특히 C와 A의 top 응집은 어려운 클래스를 쉬운 형제·동일 top으로 흡수해 지표를 미화할 수 있다. **모든 실험 보고에 H-Acc와 함께 (i) Macro Acc, (ii) fx-a·sp-c·ss-i recall, (iii) fx↔ss cross-top 누수율을 반드시 병기**해 게이밍과 진짜 개선을 분리하라.
