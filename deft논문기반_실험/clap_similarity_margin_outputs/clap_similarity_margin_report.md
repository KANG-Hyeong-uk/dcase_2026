# CLAP similarity margin confidence 평가 결과

## Margin definitions
- top1_top2_margin = top1_sim - top2_sim
- assigned_label_margin = assigned_class_sim - best_other_class_sim

## Top1 assigned-class match summary
```text
feature_source  top1_matches_assigned_rate  assigned_rank_mean  assigned_rank_median
         audio                      0.1836              7.3815                   5.0
          text                      0.1832              7.2912                   5.0
```

## Summary metrics
```text
                    feature  is_mean_monotonic_increasing  confidence_5_mean_minus_confidence_1_mean  confidence_5_mean_minus_confidence_4_mean  similarity_5bin_accuracy  similarity_5bin_macro_precision  similarity_5bin_macro_recall  similarity_5bin_macro_f1  similarity_5bin_weighted_precision  similarity_5bin_weighted_recall  similarity_5bin_weighted_f1
     audio_top1_top2_margin                         False                                    -0.0056                                    -0.0010                    0.1907                           0.1907                        0.1831                    0.1434                              0.3885                           0.1907                       0.2380
audio_assigned_label_margin                          True                                     0.1017                                     0.0387                    0.2090                           0.2090                        0.2518                    0.1646                              0.4118                           0.2090                       0.2535
      text_top1_top2_margin                         False                                     0.0216                                     0.0155                    0.2019                           0.2019                        0.2144                    0.1596                              0.3920                           0.2019                       0.2442
 text_assigned_label_margin                          True                                     0.0616                                     0.0278                    0.2066                           0.2067                        0.2260                    0.1612                              0.4080                           0.2066                       0.2521
```

## Confidence별 Mean ± Std
```text
                    feature  confidence    n    mean    std  median      q1      q3
     audio_top1_top2_margin           1  106  0.0523 0.0444  0.0413  0.0194  0.0724
     audio_top1_top2_margin           2  749  0.0500 0.0471  0.0363  0.0151  0.0687
     audio_top1_top2_margin           3 3280  0.0500 0.0476  0.0356  0.0143  0.0703
     audio_top1_top2_margin           4 6045  0.0478 0.0459  0.0336  0.0140  0.0670
     audio_top1_top2_margin           5  776  0.0468 0.0438  0.0336  0.0136  0.0656
audio_assigned_label_margin           1  106 -0.1606 0.1263 -0.1497 -0.2391 -0.0748
audio_assigned_label_margin           2  749 -0.1354 0.1291 -0.1270 -0.2141 -0.0531
audio_assigned_label_margin           3 3280 -0.1168 0.1230 -0.1172 -0.1955 -0.0347
audio_assigned_label_margin           4 6045 -0.0976 0.1145 -0.0969 -0.1696 -0.0246
audio_assigned_label_margin           5  776 -0.0589 0.1022 -0.0556 -0.1288  0.0081
      text_top1_top2_margin           1  106  0.0464 0.0403  0.0340  0.0177  0.0660
      text_top1_top2_margin           2  749  0.0528 0.0483  0.0388  0.0176  0.0719
      text_top1_top2_margin           3 3280  0.0539 0.0527  0.0384  0.0157  0.0736
      text_top1_top2_margin           4 6045  0.0525 0.0523  0.0363  0.0146  0.0727
      text_top1_top2_margin           5  776  0.0680 0.0600  0.0497  0.0192  0.1010
 text_assigned_label_margin           1  106 -0.1349 0.1116 -0.1421 -0.2030 -0.0632
 text_assigned_label_margin           2  749 -0.1273 0.1318 -0.1209 -0.2146 -0.0500
 text_assigned_label_margin           3 3280 -0.1124 0.1272 -0.1116 -0.1959 -0.0327
 text_assigned_label_margin           4 6045 -0.1011 0.1218 -0.1026 -0.1760 -0.0283
 text_assigned_label_margin           5  776 -0.0733 0.1339 -0.0741 -0.1554  0.0110
```

## Margin 5-bin metrics
```text
                    feature  accuracy  macro_precision  macro_recall  macro_f1  weighted_precision  weighted_recall  weighted_f1
     audio_top1_top2_margin    0.1907           0.1907        0.1831    0.1434              0.3885           0.1907       0.2380
audio_assigned_label_margin    0.2090           0.2090        0.2518    0.1646              0.4118           0.2090       0.2535
      text_top1_top2_margin    0.2019           0.2019        0.2144    0.1596              0.3920           0.2019       0.2442
 text_assigned_label_margin    0.2066           0.2067        0.2260    0.1612              0.4080           0.2066       0.2521
```

## audio_top1_top2_margin
### Bin ranges
```text
  sim_bin  predicted_confidence    n  margin_min  margin_mean  margin_max
Q1_pred_1                     1 2192      0.0000       0.0053      0.0110
Q2_pred_2                     2 2191      0.0110       0.0179      0.0257
Q3_pred_3                     3 2191      0.0257       0.0349      0.0451
Q4_pred_4                     4 2191      0.0451       0.0602      0.0794
Q5_pred_5                     5 2191      0.0794       0.1247      0.3469
```
### True confidence counts
```text
true_confidence       1    2    3     4    5
predicted_by_margin                         
1                    17  144  636  1239  156
2                    14  147  661  1218  151
3                    28  156  613  1233  161
4                    24  143  680  1174  170
5                    23  159  690  1181  138
```
### Classification report
```text
              precision    recall  f1-score   support

confidence_1       0.01      0.16      0.01       106
confidence_2       0.07      0.20      0.10       749
confidence_3       0.28      0.19      0.22      3280
confidence_4       0.54      0.19      0.29      6045
confidence_5       0.06      0.18      0.09       776

    accuracy                           0.19     10956
   macro avg       0.19      0.18      0.14     10956
weighted avg       0.39      0.19      0.24     10956

```

## audio_assigned_label_margin
### Bin ranges
```text
  sim_bin  predicted_confidence    n  margin_min  margin_mean  margin_max
Q1_pred_1                     1 2192     -0.6153      -0.2720     -0.1981
Q2_pred_2                     2 2191     -0.1981      -0.1620     -0.1300
Q3_pred_3                     3 2191     -0.1300      -0.1021     -0.0733
Q4_pred_4                     4 2191     -0.0733      -0.0420     -0.0064
Q5_pred_5                     5 2191     -0.0064       0.0594      0.3375
```
### True confidence counts
```text
true_confidence       1    2    3     4    5
predicted_by_margin                         
1                    38  213  796  1073   72
2                    18  156  697  1204  116
3                    24  144  609  1269  145
4                    19  126  580  1255  211
5                     7  110  598  1244  232
```
### Classification report
```text
              precision    recall  f1-score   support

confidence_1       0.02      0.36      0.03       106
confidence_2       0.07      0.21      0.11       749
confidence_3       0.28      0.19      0.22      3280
confidence_4       0.57      0.21      0.30      6045
confidence_5       0.11      0.30      0.16       776

    accuracy                           0.21     10956
   macro avg       0.21      0.25      0.16     10956
weighted avg       0.41      0.21      0.25     10956

```

## text_top1_top2_margin
### Bin ranges
```text
  sim_bin  predicted_confidence    n  margin_min  margin_mean  margin_max
Q1_pred_1                     1 2192      0.0000       0.0058      0.0119
Q2_pred_2                     2 2191      0.0119       0.0192      0.0274
Q3_pred_3                     3 2191      0.0274       0.0381      0.0499
Q4_pred_4                     4 2191      0.0499       0.0660      0.0865
Q5_pred_5                     5 2191      0.0865       0.1408      0.3516
```
### True confidence counts
```text
true_confidence       1    2    3     4    5
predicted_by_margin                         
1                    18  126  648  1275  125
2                    29  150  653  1222  137
3                    15  167  642  1238  129
4                    33  164  695  1158  141
5                    11  142  642  1152  244
```
### Classification report
```text
              precision    recall  f1-score   support

confidence_1       0.01      0.17      0.02       106
confidence_2       0.07      0.20      0.10       749
confidence_3       0.29      0.20      0.23      3280
confidence_4       0.53      0.19      0.28      6045
confidence_5       0.11      0.31      0.16       776

    accuracy                           0.20     10956
   macro avg       0.20      0.21      0.16     10956
weighted avg       0.39      0.20      0.24     10956

```

## text_assigned_label_margin
### Bin ranges
```text
  sim_bin  predicted_confidence    n  margin_min  margin_mean  margin_max
Q1_pred_1                     1 2192     -0.5652      -0.2787     -0.2052
Q2_pred_2                     2 2191     -0.2051      -0.1669     -0.1345
Q3_pred_3                     3 2192     -0.1344      -0.1046     -0.0767
Q4_pred_4                     4 2190     -0.0767      -0.0446     -0.0083
Q5_pred_5                     5 2191     -0.0082       0.0718      0.3516
```
### True confidence counts
```text
true_confidence       1    2    3     4    5
predicted_by_margin                         
1                    26  200  732  1103  131
2                    33  148  647  1248  115
3                    17  136  642  1258  139
4                    16  139  644  1224  167
5                    14  126  615  1212  224
```
### Classification report
```text
              precision    recall  f1-score   support

confidence_1       0.01      0.25      0.02       106
confidence_2       0.07      0.20      0.10       749
confidence_3       0.29      0.20      0.23      3280
confidence_4       0.56      0.20      0.30      6045
confidence_5       0.10      0.29      0.15       776

    accuracy                           0.21     10956
   macro avg       0.21      0.23      0.16     10956
weighted avg       0.41      0.21      0.25     10956

```