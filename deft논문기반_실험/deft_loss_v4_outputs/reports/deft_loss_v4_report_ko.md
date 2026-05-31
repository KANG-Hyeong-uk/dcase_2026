# DEFT Loss v4 OOF Score Stacking Report

## 실험 설정
- base input: 04 feature set
- base model: 04 MLP structure
- loss weight: confidence 1 = 5.0, confidence 2 = 1.5, confidence 3/4/5 = 1.0
- target: confidence 123 vs 45
- evaluation: BSD10k OOF

## 핵심 결과
- best F1 method: OOF ridge stacker threshold=0.3860 f1=0.7832 precision=0.6753 recall=0.9320
- best macro-F1 method: OOF ridge stacker threshold=0.5690 macro_f1=0.6776 recall_123=0.6416 recall_45=0.7216
- best balanced recall method: rank average: binary + expected + P45 threshold=0.4540 recall_123=0.6842 recall_45=0.6847

## 목적별 threshold
```text
                       objective                                method  threshold  precision  recall     f1  macro_f1  recall_123  recall_45  auc_pr
                      F1-optimal                     OOF ridge stacker      0.386     0.6753  0.9320 0.7832    0.5816      0.2609     0.9320  0.8375
                macro-F1-optimal                     OOF ridge stacker      0.569     0.7686  0.7216 0.7443    0.6776      0.6416     0.7216  0.8375
          balanced recall 123/45 rank average: binary + expected + P45      0.454     0.7815  0.6847 0.7299    0.6753      0.6842     0.6847  0.8366
precision-optimal @ recall>=0.70 rank average: binary + expected + P45      0.436     0.7745  0.7000 0.7354    0.6752      0.6638     0.7000  0.8366
recall-optimal @ precision>=0.70 rank average: binary + expected + P45      0.225     0.7002  0.8812 0.7803    0.6301      0.3775     0.8812  0.8366
```

## 저장 파일
- predictions: `deft논문기반_실험/deft_loss_v4_outputs/predictions/BSD10k_oof_deft_loss_v4_scores_with_stackers.csv`
- threshold summary: `deft논문기반_실험/deft_loss_v4_outputs/reports/deft_loss_v4_threshold_summary.csv`
- best thresholds: `deft논문기반_실험/deft_loss_v4_outputs/reports/deft_loss_v4_best_by_objective.csv`
- PR curves: `deft논문기반_실험/deft_loss_v4_outputs/plots/deft_loss_v4_pr_curves.png`