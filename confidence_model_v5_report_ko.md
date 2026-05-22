# BSD10k Confidence v5 Experiment Report

- Device: `cuda`
- Elapsed: 12.7 min
- Rows: 10,956
- Folds: 5
- MLP architecture: Linear(input, 512) -> GELU -> Dropout -> Linear(512, 256) -> GELU -> Dropout -> Linear(256, 4)
- Loss: sample-weighted ordinal BCE using metadata `class` inverse-frequency weights
- Class sample weights: enabled; source=`metadata`
- Tree backend: XGBoost
- Plot backend: matplotlib

## Fold Mean Results

| Model             | Feature Dim | MAE             | Accuracy (%) | Macro F1        | Macro Precision | Macro Recall    | QWK             |
| ----------------- | ----------- | --------------- | ------------ | --------------- | --------------- | --------------- | --------------- |
| E1 (Base MLP)     | 1047        | 0.5189 ± 0.0060 | 57.69 ± 1.04 | 0.3050 ± 0.0058 | 0.4184 ± 0.0267 | 0.2897 ± 0.0064 | 0.3226 ± 0.0178 |
| E2 (+ Agreement)  | 1050        | 0.5183 ± 0.0085 | 57.14 ± 0.55 | 0.3109 ± 0.0135 | 0.3991 ± 0.0095 | 0.2944 ± 0.0111 | 0.3422 ± 0.0278 |
| E3 (+ Proto/Cons) | 1057        | 0.5077 ± 0.0071 | 58.23 ± 1.50 | 0.3166 ± 0.0122 | 0.4112 ± 0.0274 | 0.3001 ± 0.0101 | 0.3728 ± 0.0260 |
| E4 (Tree Scalar)  | 33          | 0.5321 ± 0.0060 | 57.23 ± 0.73 | 0.2578 ± 0.0072 | 0.3405 ± 0.0160 | 0.2701 ± 0.0056 | 0.3411 ± 0.0133 |

## OOF Results

| experiment               | feature_dim | mae    | accuracy | macro_f1 | macro_precision | macro_recall | quadratic_weighted_kappa | spearman |
| ------------------------ | ----------- | ------ | -------- | -------- | --------------- | ------------ | ------------------------ | -------- |
| E3_proto_consistency_mlp | 1057        | 0.5077 | 0.5823   | 0.3174   | 0.4075          | 0.3001       | 0.3728                   | 0.4946   |
| E2_agreement_mlp         | 1050        | 0.5183 | 0.5714   | 0.3121   | 0.4112          | 0.2944       | 0.3426                   | 0.4590   |
| E1_base_mlp              | 1047        | 0.5189 | 0.5769   | 0.3060   | 0.4123          | 0.2897       | 0.3228                   | 0.4586   |
| E4_tree_scalar           | 33          | 0.5321 | 0.5723   | 0.2579   | 0.3386          | 0.2701       | 0.3411                   | 0.4535   |

## Best Model

- Best by MAE: `E3_proto_consistency_mlp`
- MAE: 0.5077
- Accuracy: 58.23%
- Macro F1: 0.3174
- QWK: 0.3728

## Saved Artifacts

- Summary CSV: `outputs/confidence_model_v5/reports/v5_experiment_summary.csv`
- Fold metrics CSV: `outputs/confidence_model_v5/reports/v5_fold_metrics.csv`
- Class distribution mapping CSV: `outputs/confidence_model_v5/reports/v5_class_distribution_mapping.csv`
- Class sample weights CSV: `outputs/confidence_model_v5/reports/v5_class_sample_weights.csv`
- Loss plots: `outputs/confidence_model_v5/plots/*_mean_loss.png`
- Row-normalized OOF confusion plots: `outputs/confidence_model_v5/plots/*_confusion_row_normalized.png`
- OOF predictions: `outputs/confidence_model_v5/predictions/BSD10k_oof_v5_predictions.csv`
