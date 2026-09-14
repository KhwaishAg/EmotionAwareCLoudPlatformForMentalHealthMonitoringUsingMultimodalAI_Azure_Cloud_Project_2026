# Member 1 — Questionnaire Modality Implementation

## 1. Objective

Deliver a reusable questionnaire stress-risk component and a fusion-ready contract.

## 2. Dataset

The component uses 52 real questionnaire responses; no synthetic data was used.

## 3. Data Preparation

Cleaning and missing-value policy are followed by semantic ordinal encoding, Q10/Q11 multi-select handling, and Q1/Q2/Q25 categorical handling. Q10 `Nothing`, Q2 `All of them`, and Q14/Q16 `It varies considerably` preserve their specified behavior.

## 4. Feature Engineering

Derived dimensions are overwhelm, stress impact, emotional strain, stress-aligned coping, stress persistence, recovery difficulty, stress behavior, and academic preparation load.

## 5. Prediction Target

Self-reported Q4 defines the target: 1–2 `low`, 3 `moderate`, and 4–5 `high`. Q4 is excluded from predictors. This is project-defined, non-clinical stress classification.

## 6. Models

Logistic Regression and Random Forest were saved as sklearn pipelines.

## 7. Validation

Evaluation used 4-fold StratifiedKFold.

## 8. Results

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.5769 ± 0.1601 | 0.5278 ± 0.2119 | 0.5073 ± 0.1943 |
| Random Forest | 0.6154 ± 0.1088 | 0.5611 ± 0.1285 | 0.5518 ± 0.1534 |

## 9. Selected Model

Random Forest is selected from current cross-validation evidence; this preliminary selection is not clinical validation.

## 10. Explainability

Top model-associated indicators are `stress_impact_dimension`, `overwhelm_dimension`, `Q9_encoded`, `Q12_encoded`, and `stress_behavior_dimension`. They are not causal claims.

## 11. Confidence

Current OOF mean confidence is 0.3818 and median confidence is 0.3544. This project-level model-confidence indicator is not clinical certainty.

## 12. Fusion

Usable modalities are weighted by confidence × completeness, normalized, and assessed for cross-modal agreement. Missing modalities are ignored rather than treated as zero risk.

## 13. Limitations

- Only 52 respondents; high predictor-to-sample ratio and class imbalance.
- Preliminary cross-validation estimates may be unstable across folds.
- Questionnaire self-report bias.
- No clinical diagnosis or clinical validation.
- Voice and behavioral models are developed separately by other members.

## 14. Reproducibility

Model metadata records `random_state = 42` and `synthetic_data_used = false`. Run `predict.py`, `fusion_engine.py`, and `final_validation.py` from the repository root.
