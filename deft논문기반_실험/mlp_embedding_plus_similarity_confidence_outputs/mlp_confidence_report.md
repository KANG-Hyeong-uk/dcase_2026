# Embedding Feature MLP Confidence Classification Report

## Input features
- audio_embedding_512 + text_embedding_512 + class_one_hot + similarity_margin_4
- audio embedding dim: 512
- text embedding dim: 512
- class one-hot dim: 23
- scalar feature dim: 4
  - audio_text_sim
  - audio_class_sim
  - audio_assigned_label_margin
  - text_assigned_label_margin
- total input dim: 1051

## Metrics
```text
                           task  accuracy  macro_precision  macro_recall  macro_f1  weighted_precision  weighted_recall  weighted_f1    mae  best_epoch  best_val_loss
          confidence_5class_mlp    0.5937           0.4366        0.2843    0.2969              0.5684           0.5937       0.5476 0.4629           4         0.9448
confidence_binary_123_vs_45_mlp    0.7032           0.6830        0.6656    0.6702              0.6958           0.7032       0.6958 0.2968           3         0.5721
```

## 5-class confusion matrix
```text
        pred_1  pred_2  pred_3  pred_4  pred_5
true_1       0       1       9       6       0
true_2       0      11      38      63       0
true_3       0       4     183     303       2
true_4       0       4     123     770      10
true_5       0       0       3     102      12
```

## 5-class classification report
```text
              precision    recall  f1-score   support

confidence_1       0.00      0.00      0.00        16
confidence_2       0.55      0.10      0.17       112
confidence_3       0.51      0.37      0.43       492
confidence_4       0.62      0.85      0.72       907
confidence_5       0.50      0.10      0.17       117

    accuracy                           0.59      1644
   macro avg       0.44      0.28      0.30      1644
weighted avg       0.57      0.59      0.55      1644

```

## Binary confusion matrix
```text
          pred_123  pred_45
true_123       318      302
true_45        186      838
```

## Binary classification report
```text
                precision    recall  f1-score   support

confidence_123       0.63      0.51      0.57       620
 confidence_45       0.74      0.82      0.77      1024

      accuracy                           0.70      1644
     macro avg       0.68      0.67      0.67      1644
  weighted avg       0.70      0.70      0.70      1644

```