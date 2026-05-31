# Confidence 1/2 Weighted Loss Report

## Choice
- Selected weighted loss over oversampling.
- `up_sampling_confi4` showed `12 vs 345` weighted loss had better/stabler macro F1 than the confidence-1-only setup, while oversampling/XGBoost hurt 5-class MAE strongly.

## Loss weights
 confidence  train_count  loss_weight
          1           74    20.540625
          2          524     2.927840
          3         2296     1.000000
          4         4232     1.000000
          5          543     1.000000

## Metrics
```text
 accuracy  macro_precision  macro_recall  macro_f1  weighted_precision  weighted_recall  weighted_f1    mae                                       task  best_epoch  best_val_loss
   0.5675           0.4010        0.3272    0.3211              0.5570           0.5675       0.5332 0.5182           confidence_5class_weighted12_mlp          13         1.0179
   0.6989           0.6825        0.6877    0.6844              0.7044           0.6989       0.7010 0.3011 confidence_binary_123_vs_45_weighted12_mlp           4         0.5894
```

## 5-class report
```text
              precision    recall  f1-score   support

confidence_1     0.0625    0.1250    0.0833        16
confidence_2     0.2429    0.3036    0.2698       112
confidence_3     0.5000    0.2663    0.3475       492
confidence_4     0.6344    0.8302    0.7192       907
confidence_5     0.5652    0.1111    0.1857       117

    accuracy                         0.5675      1644
   macro avg     0.4010    0.3272    0.3211      1644
weighted avg     0.5570    0.5675    0.5332      1644

```

## Binary report
```text
                precision    recall  f1-score   support

confidence_123     0.5931    0.6419    0.6166       620
 confidence_45     0.7718    0.7334    0.7521      1024

      accuracy                         0.6989      1644
     macro avg     0.6825    0.6877    0.6844      1644
  weighted avg     0.7044    0.6989    0.7010      1644

```