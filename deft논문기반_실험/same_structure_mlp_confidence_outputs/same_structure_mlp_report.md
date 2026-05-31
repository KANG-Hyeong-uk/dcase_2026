# Same-Structure MLP Confidence Classification Report

## Model
- Dropout(dropout) -> Linear(input_dim, 5)
- This matches the actual SimpleMLP code in `bsd10k_confidence_mlp_colab.ipynb`.

## Input features
- audio_embedding_512 + text_embedding_512 + class_one_hot + similarity_margin_4
- total input dim: 1051

## Dropout sweep metrics
```text
                           experiment    acc    mae    mse  precision_macro  recall_macro  f1_macro  precision_weighted  recall_weighted  f1_weighted  best_epoch  dropout
same_structure_similarity_dropout_0.2 0.5680 0.4872 0.6095           0.5059        0.3369    0.3640              0.5588           0.5680       0.5537          13      0.2
same_structure_similarity_dropout_0.3 0.5643 0.4881 0.6058           0.4733        0.3288    0.3539              0.5538           0.5643       0.5480          13      0.3
same_structure_similarity_dropout_0.4 0.5630 0.4868 0.5953           0.4994        0.3256    0.3518              0.5509           0.5630       0.5458          13      0.4
same_structure_similarity_dropout_0.7 0.5123 0.5652 0.7368           0.3626        0.3568    0.3487              0.5399           0.5123       0.5149           3      0.7
same_structure_similarity_dropout_0.5 0.5589 0.4891 0.5940           0.4806        0.3221    0.3483              0.5496           0.5589       0.5409          13      0.5
same_structure_similarity_dropout_0.6 0.5064 0.5725 0.7495           0.3494        0.3518    0.3413              0.5373           0.5064       0.5111           3      0.6
same_structure_similarity_dropout_0.8 0.5333 0.5233 0.6464           0.3702        0.3441    0.3397              0.5499           0.5333       0.5290           5      0.8
```

## Best classification report
```text
              precision    recall  f1-score   support

confidence_1       0.50      0.05      0.09        21
confidence_2       0.45      0.16      0.24       150
confidence_3       0.45      0.50      0.47       656
confidence_4       0.64      0.70      0.67      1210
confidence_5       0.48      0.28      0.35       155

    accuracy                           0.57      2192
   macro avg       0.51      0.34      0.36      2192
weighted avg       0.56      0.57      0.55      2192

```

## Best confusion matrix
```text
        pred_1  pred_2  pred_3  pred_4  pred_5
true_1       1       1       7      12       0
true_2       1      24      63      61       1
true_3       0      18     325     308       5
true_4       0      10     308     852      40
true_5       0       0      12     100      43
```