
# AtmoSync — Final Model Selection

## 1. Objective

Select a multiclass risk prediction architecture for the AtmoSync Micro-Climate Arbitrage Analytics project.

The target classes are:
- High
- Medium
- Low

Model selection considers macro F1, class-level recall, temporal validation, and practical trade-offs.

## 2. Models Evaluated

1. Baseline custom-weighted XGBoost
2. Hierarchical XGBoost
   - Stage 1: Multiclass risk classification
   - Stage 2: Binary Low-versus-Medium classification
3. Threshold-optimized hierarchical XGBoost

All three models were evaluated on the same 13,474 final test observations.

## 3. Final Test Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline XGBoost | 82.83% | 0.4862 |
| Hierarchical XGBoost | 80.50% | 0.5293 |
| Optimized Hierarchical XGBoost | 80.29% | 0.5292 |

## 4. Class-Level Performance

### Baseline XGBoost

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| High | 0.4776 | 0.2817 | 0.3543 |
| Low | 0.8547 | 0.9767 | 0.9117 |
| Medium | 0.4273 | 0.1243 | 0.1926 |

### Hierarchical XGBoost

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| High | 0.4776 | 0.2817 | 0.3543 |
| Low | 0.8813 | 0.9144 | 0.8976 |
| Medium | 0.3553 | 0.3187 | 0.3360 |

### Optimized Hierarchical XGBoost

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| High | 0.4066 | 0.3497 | 0.3760 |
| Low | 0.8813 | 0.9142 | 0.8975 |
| Medium | 0.3477 | 0.2863 | 0.3140 |

## 5. Temporal Validation

Three expanding-window validation folds were used.

| Model | Mean Macro F1 | Standard Deviation |
|---|---:|---:|
| Baseline XGBoost | 0.3841 | 0.0826 |
| Hierarchical XGBoost | 0.4278 | 0.0816 |

The hierarchical model achieved higher Macro F1 in all three temporal folds.

The threshold-optimized hierarchical model was not evaluated across these same three folds.

## 6. Final Selection

**Selected candidate: Hierarchical XGBoost**

Reasons:
- Highest final test Macro F1 (0.5293).
- Higher mean temporal validation Macro F1.
- Improved Medium-risk recall from 12.43% to 31.87%.
- Better balance between the three risk classes than the baseline.

## 7. Limitations

- High-risk recall remains low at 28.17%.
- Medium-risk precision remains limited at 35.53%.
- Temporal performance varies between folds.
- The threshold-optimized hierarchical model has not undergone equivalent multi-fold temporal validation.
- The dataset is synthetic and does not establish real-world predictive performance.
- Further testing on independent real-world data is required before operational use.

## 8. Final Decision

The standard Hierarchical XGBoost model is selected as the leading research candidate for the next stage of the AtmoSync project.

The model will be used for the reusable prediction pipeline and API development, with its limitations clearly disclosed.

