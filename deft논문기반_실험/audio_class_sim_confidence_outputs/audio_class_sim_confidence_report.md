# audio_class_sim confidence 평가 결과

## Class embedding mapping
metadata의 class 값을 data/class_embedding/{class}.npy와 매핑했다.

## Confidence별 Mean ± Std
```text
 confidence    n   mean    std  median      q1     q3
          1  106 0.0377 0.1317  0.0374 -0.0492 0.1258
          2  749 0.0661 0.1331  0.0668 -0.0214 0.1609
          3 3280 0.0744 0.1320  0.0741 -0.0192 0.1624
          4 6045 0.0780 0.1306  0.0750 -0.0127 0.1674
          5  776 0.1331 0.1353  0.1142  0.0436 0.2342
```

## Summary metrics
```text
 n_rows  is_mean_monotonic_increasing  confidence_5_mean_minus_confidence_1_mean  confidence_5_mean_minus_confidence_4_mean  similarity_5bin_accuracy  similarity_5bin_macro_precision  similarity_5bin_macro_recall  similarity_5bin_macro_f1  similarity_5bin_weighted_precision  similarity_5bin_weighted_recall  similarity_5bin_weighted_f1
  10956                          True                                     0.0953                                      0.055                    0.2107                           0.2107                         0.246                    0.1677                              0.4078                           0.2107                       0.2536
```

## Similarity 5-bin ranges
```text
  sim_bin  predicted_confidence    n  sim_min  sim_mean  sim_max
Q1_pred_1                     1 2192  -0.3858   -0.1046  -0.0344
Q2_pred_2                     2 2191  -0.0344    0.0075   0.0446
Q3_pred_3                     3 2191   0.0446    0.0778   0.1107
Q4_pred_4                     4 2191   0.1108    0.1494   0.1927
Q5_pred_5                     5 2191   0.1927    0.2681   0.5185
```

## Similarity 5-bin true confidence counts
```text
true_confidence           1    2    3     4    5
predicted_by_similarity                         
1                        31  168  712  1206   75
2                        24  154  640  1250  123
3                        19  138  652  1200  182
4                        20  154  666  1213  138
5                        12  135  610  1176  258
```

## Similarity 5-bin metrics
```text
                       method  accuracy  macro_precision  macro_recall  macro_f1  weighted_precision  weighted_recall  weighted_f1
audio_class_sim_quantile_5bin    0.2107           0.2107         0.246    0.1677              0.4078           0.2107       0.2536
```

## Classification report
```text
              precision    recall  f1-score   support

confidence_1       0.01      0.29      0.03       106
confidence_2       0.07      0.21      0.10       749
confidence_3       0.30      0.20      0.24      3280
confidence_4       0.55      0.20      0.29      6045
confidence_5       0.12      0.33      0.17       776

    accuracy                           0.21     10956
   macro avg       0.21      0.25      0.17     10956
weighted avg       0.41      0.21      0.25     10956

```

## 저장된 그림
- plots/confidence_mean_std_audio_class_sim.png
- plots/confidence_boxplot_audio_class_sim.png