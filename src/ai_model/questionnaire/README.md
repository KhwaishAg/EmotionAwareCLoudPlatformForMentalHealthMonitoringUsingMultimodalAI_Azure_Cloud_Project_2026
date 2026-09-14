# Questionnaire modality

This component implements a project-defined, non-clinical questionnaire stress classification workflow using 52 real questionnaire responses. No synthetic training data was used.

Raw responses are cleaned with a documented missing-value policy, then transformed through semantic ordinal encoding, Q10/Q11 multi-select encoding, and Q1/Q2/Q25 categorical encoding. Q10 `Nothing`, Q2 `All of them`, and Q14/Q16 `It varies considerably` retain their required semantics. Derived features include overwhelm, stress impact, emotional strain, stress-aligned coping, stress persistence, recovery difficulty, stress behavior, and academic preparation load.

The self-reported Q4 target maps 1–2 to `low`, 3 to `moderate`, and 4–5 to `high`. Q4 and Q4-derived fields are excluded from the 75 model predictors to prevent target leakage. Logistic Regression and Random Forest are evaluated using 4-fold `StratifiedKFold`; Random Forest is currently selected by cross-validation evidence. Metrics include accuracy, balanced accuracy, macro precision/recall, and macro F1.

Explainability values are model-associated indicators, not causes. Confidence uses `0.5 × maximum probability + 0.5 × probability margin`; it is a project-level model-confidence heuristic, not clinical certainty.

## Prediction and fusion

`predict_questionnaire(response)` accepts raw Q1–Q20 values plus optional Timestamp/Q21–Q25 values and returns the standardized fusion-ready contract. The saved sklearn pipeline applies fitted imputation during inference and is never refit. See [prediction contract](../../../results/questionnaire/prediction_contract.md).

The fusion engine combines this output with independently developed `voice` and/or `behavioral` contracts. It uses `confidence × completeness` weighting and ignores missing modalities rather than assigning zero risk. See [fusion README](../fusion/README.md).

## Limitations

There are only 52 respondents, a high predictor-to-sample ratio, class imbalance, potentially unstable cross-validation estimates, and questionnaire self-report bias. This is not a clinical diagnosis or clinical validation. Voice and behavioral models are separate components developed by other members.

## Run

```powershell
python src/ai_model/questionnaire/predict.py
python src/ai_model/fusion/fusion_engine.py
python src/ai_model/questionnaire/final_validation.py
```
