# BSD10k Confidence Ceiling Estimate

| baseline                                        | mae    | accuracy | quadratic_weighted_kappa | note                                             |
| ----------------------------------------------- | ------ | -------- | ------------------------ | ------------------------------------------------ |
| Random ±1 perturbation                          | 0.5001 |          |                          | 50% of labels moved to a valid adjacent class    |
| Majority class = 4                              | 0.5360 | 0.5518   | 0.0000                   | All samples predicted as confidence 4            |
| Class prior random sampling                     | 0.7646 |          |                          | Mean over 500 seeded draws; std=0.0064           |
| Previous best: ensemble_avg_best_ordinal_towers | 0.5088 | 0.5888   | 0.3768                   | From confidence_model_true_5class_experiments.py |

Random ±1 perturbation is a label-noise proxy, not an achievable model target. It estimates the error produced when half of labels are shifted to an adjacent valid ordinal class.
