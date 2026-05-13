# evaluate.py - 평가 메트릭 완벽 가이드

## 📋 파일 개요
`evaluate.py`는 모델의 **다양한 평가 메트릭**을 계산합니다.
단순 정확도부터 계층적 정밀도/재현율/F1까지 포괄적인 평가를 제공합니다.

---

## 🔍 전체 메트릭 체계

```
평가 메트릭
├─ 단순 정확도 (Flat Metrics)
│  ├─ 미시 정확도 (Micro Accuracy)
│  └─ 거시 정확도 (Macro Accuracy)
│
└─ 계층적 메트릭 (Hierarchical Metrics)
   ├─ 계층적 정확도 (Hierarchical Accuracy)
   ├─ 계층적 정밀도 (Hierarchical Precision)
   ├─ 계층적 재현율 (Hierarchical Recall)
   ├─ 계층적 F1 (Hierarchical F1)
   ├─ 가중 계층적 정밀도
   ├─ 가중 계층적 재현율
   └─ 가중 계층적 F1
```

---

## 📊 단순 정확도

### 1️⃣ 미시 정확도 (Micro Accuracy)

```python
micro_accuracy = correct_predictions / total_predictions
```

**정의**: 전체 샘플에서 정답의 비율
- 모든 샘플을 동등하게 취급
- 클래스 불균형의 영향을 받음

**예시**:
```
총 1000개 샘플
정답: 850개

미시 정확도 = 850 / 1000 = 85%
```

### 2️⃣ 거시 정확도 (Macro Accuracy)

```python
macro_accuracy = (acc_class_1 + acc_class_2 + ... + acc_class_N) / N
```

**정의**: 각 클래스별 정확도의 평균
- 클래스별로 동등하게 취급
- 클래스 불균형으로부터 보호

**예시**:
```
클래스 A (100개): 90개 정답 → 90% 정확도
클래스 B (900개): 760개 정답 → 84.4% 정확도

거시 정확도 = (90% + 84.4%) / 2 = 87.2%
미시 정확도 = (90 + 760) / 1000 = 85%
```

**차이점**:
- **미시**: 자주 나타나는 클래스에 편향
- **거시**: 모든 클래스를 공평하게 평가

---

## 🌳 계층적 메트릭 상세

### 배경: 계층적 분류

**음향 분류의 계층 구조**:
```
레벨 1 (상위 클래스):
┌─ dog (견)
├─ speech (음성)
├─ car (자동차)
├─ music (음악)
└─ environmental (환경음)

레벨 2 (하위 클래스):
dog:
  - dog-barking (짖음)
  - dog-growling (으르렁)
  - dog-whimpering (윙윙)

speech:
  - speech-male (남성 음성)
  - speech-female (여성 음성)
```

**문제점**: 계층을 무시하는 평가
```
정답: dog-barking
예측: speech-male
미시 정확도: 0 (완전 오류)

하지만:
- 상위 클래스는 일치하지 않음 ✗
- 완전히 다른 카테고리 ✗✗
```

### 1️⃣ 계층적 정확도 (Hierarchical Accuracy)

```python
def hierarchical_accuracy(subcat, predictions_gt, lambda_param=0.5):
    prediction_scores = []
    
    for prediction, gt in predictions_gt:
        if subcat == gt:  # 정답 샘플일 때만
            preditcion_top_level, preditcion_sub_level = extend_subcat(prediction)
            gt_top_level, gt_sub_level = extend_subcat(gt)
            
            if preditcion_top_level == gt_top_level and preditcion_sub_level == gt_sub_level:
                prediction_scores.append(1)  # 완벽한 예측
            elif preditcion_top_level == gt_top_level and preditcion_sub_level != gt_sub_level:
                prediction_scores.append(lambda_param)  # 상위만 맞음 (부분 점수)
            else:
                prediction_scores.append(0.0)  # 완전히 다름
    
    classAcc = sum(prediction_scores) / len(prediction_scores)
    return classAcc
```

**채점 시스템**:
```
정답: dog-barking

경우 1: 예측 = dog-barking
점수: 1.0 (100% 정답)

경우 2: 예측 = dog-growling
점수: 0.5 (50% 정답, λ=0.5)
- 상위 클래스 일치 ✓
- 하위 클래스 불일치 ✗

경우 3: 예측 = speech-male
점수: 0.0 (0% 정답)
- 상위 클래스 불일치 ✗✗
```

**λ (Lambda) 파라미터**:
```
λ = 0.5 (기본값): 부분 점수를 절반만 준다
λ = 0.0: 부분 점수 무시 (표준 정확도)
λ = 1.0: 상위 클래스 일치를 전체 점수로 준다
```

---

### 2️⃣ 계층적 정밀도/재현율/F1

```python
def hierarchical_prf(subcat, predictions_gt):
    # 클래스 경로 비교
    # 경로: 상위-하위 형태
    
    hPP = []  # 정밀도 값들
    hRR = []  # 재현율 값들
    
    for prediction, gt in predictions_gt:
        pi = extend_subcat(prediction)  # ('dog', 'dog-barking')
        ti = extend_subcat(gt)
        pi_intersection_ti = intersection(pi, ti)  # 교집합
        
        if subcat == prediction:  # 예측이 해당 클래스일 때
            hP = len(pi_intersection_ti) / len(pi)  # 정밀도
            hPP.append(hP)
        
        if subcat == gt:  # 정답이 해당 클래스일 때
            hR = len(pi_intersection_ti) / len(ti)  # 재현율
            hRR.append(hR)
    
    classP = sum(hPP) / len(hPP)
    classR = sum(hRR) / len(hRR)
    classF = 2 * classP * classR / (classP + classR) if classP + classR > 0 else 0
    
    return classP, classR, classF
```

**경로 교집합 예시**:

```
정답: dog-barking        경로: ('dog', 'dog-barking')
예측: dog-growling       경로: ('dog', 'dog-growling')
교집합: ('dog',)

정밀도 계산:
- 예측 경로에서 정답과 공통: 1개 ('dog')
- 예측 경로 전체: 2개 ('dog', 'dog-growling')
- 정밀도: 1/2 = 0.5

재현율 계산:
- 정답 경로에서 예측과 공통: 1개 ('dog')
- 정답 경로 전체: 2개 ('dog', 'dog-barking')
- 재현율: 1/2 = 0.5

F1: 2 × (0.5 × 0.5) / (0.5 + 0.5) = 0.5
```

---

### 3️⃣ 가중 계층적 메트릭 (Weighted Hierarchical)

```python
def hierarchical_prf_weighted(subcat, predictions_gt, lambda_param=0.75):
    hPP = []
    hRR = []
    
    for prediction, gt in predictions_gt:
        pi = extend_subcat(prediction)
        ti = extend_subcat(gt)
        pi_intersection_ti = intersection(pi, ti)
        
        # 가중치 결정
        if prediction == gt:
            w = 1.0  # 완벽 일치: 가중치 1
        elif get_top_level(prediction) == get_top_level(gt):
            w = lambda_param  # 상위만 일치: 가중치 λ
        else:
            w = 0.0  # 전체 불일치: 가중치 0
        
        if subcat == prediction:
            hP = (w * len(pi_intersection_ti)) / len(pi)
            hPP.append(hP)
        
        if subcat == gt:
            hR = (w * len(pi_intersection_ti)) / len(ti)
            hRR.append(hR)
    
    classP = sum(hPP) / len(hPP) if hPP else 0
    classR = sum(hRR) / len(hRR) if hRR else 0
    classF = 2 * classP * classR / (classP + classR) if classP + classR > 0 else 0
    
    return classP, classR, classF
```

**가중 vs 일반 비교**:

```
일반 계층적 메트릭:
└─ 교집합 크기만 고려

가중 계층적 메트릭:
├─ 완벽 일치: w=1.0
├─ 상위만 일치: w=0.75
└─ 불일치: w=0.0

예시 (λ=0.75):
정답: dog-barking
예측: dog-growling

일반:
- 정밀도: 0.5
- 재현율: 0.5

가중:
- 정밀도: 0.75 × 0.5 = 0.375
- 재현율: 0.75 × 0.5 = 0.375
```

---

## 🔍 evaluate_model() 함수

```python
def evaluate_model(model_class, model_path, data_loader, device, 
                   class_to_topclass, output_dir, fold_id, class_dict=None):
```

**단계별 처리**:

### 1️⃣ 모델 로드

```python
checkpoint = torch.load(model_path, map_location=device)
config = checkpoint["config"]
model = model_class(**config)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()
```

### 2️⃣ 예측 수집

```python
predictions = {"sound_id": [], "gt": [], "pred": [], "pred_score": []}

with torch.no_grad():
    for data in data_loader:
        labels = data['class_idx'].to(device)
        sound_ids = data['sound_id']
        
        audio_emb = data.get('audio_embedding', None)
        text_emb = data.get('text_embedding', None)
        
        if audio_emb is not None:
            audio_emb = audio_emb.to(device)
        if text_emb is not None:
            text_emb = text_emb.to(device)

        _, class_logits, _ = model(audio_emb, text_emb)
        probs = torch.softmax(class_logits, dim=1)

        # Top-1 예측
        top1 = torch.argmax(probs, dim=1)
        max_probs = probs.gather(1, top1.unsqueeze(1)).squeeze(1)

        # 저장
        for i in range(labels.size(0)):
            sid = sound_ids[i]
            if isinstance(sid, torch.Tensor):
                sid = sid.item()

            predictions["sound_id"].append(sid)
            predictions["gt"].append(labels[i].item())
            predictions["pred"].append(top1[i].item())
            predictions["pred_score"].append(float(max_probs[i]))
```

### 3️⃣ 메트릭 계산

```python
def compute_metrics(predictions, class_to_topclass, class_dict):
    total = len(predictions["gt"])
    preds = predictions["pred"]
    gts = predictions["gt"]
    
    # 미시 정확도
    micro_acc = sum(p == g for p, g in zip(preds, gts)) / total
    
    # 거시 정확도 (클래스별)
    macro_acc_list = []
    for class_idx in range(len(class_dict)):
        class_gts = [g == class_idx for g in gts]
        class_preds = [p == class_idx for p in preds]
        if sum(class_gts) > 0:
            acc = sum(cp and cg for cp, cg in zip(class_preds, class_gts)) / sum(class_gts)
            macro_acc_list.append(acc)
    macro_acc = np.mean(macro_acc_list) if macro_acc_list else 0
    
    # 계층적 메트릭 (모든 클래스에 대해)
    pred_class_names = [id_to_class.get(p) for p in preds]
    gt_class_names = [id_to_class.get(g) for g in gts]
    predictions_gt = list(zip(pred_class_names, gt_class_names))
    
    # ... 계층적 메트릭 계산
    
    return {
        'micro_accuracy': micro_acc,
        'macro_accuracy': macro_acc,
        'hierarchical_accuracy': hier_acc,
        'hierarchical_precision': hier_p,
        'hierarchical_recall': hier_r,
        'hierarchical_f1': hier_f1,
        # ... 더 많은 메트릭
    }
```

---

## 📊 출력 예시

```
Test Results for Fold 0 (mode=both):
────────────────────────────────
Micro Accuracy:    85.23%
Macro Accuracy:    82.47%

Hierarchical Accuracy:   84.56%
Hierarchical Precision:  81.34%
Hierarchical Recall:     83.12%
Hierarchical F1:         82.21%

Weighted (λ=0.75) Metrics:
  Weighted Accuracy:     84.12%
  Weighted Precision:    80.98%
  Weighted Recall:       82.67%
  Weighted F1:           81.81%
```

---

## 🎯 학습 포인트

- ✅ **미시 vs 거시**: 데이터셋 균형에 따라 선택
- ✅ **계층적 평가**: 부분 점수로 더 공정한 평가
- ✅ **λ 파라미터**: 상위 클래스 일치의 가중치 조절
- ✅ **경로 교집합**: 계층 구조의 공통 부분 계산
- ✅ **가중 메트릭**: 예측의 신뢰도를 반영
