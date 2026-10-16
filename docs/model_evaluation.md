# Model evaluation summary

All numbers come from reports already in the repository. None of them is clinical validation or evidence of real-world
accuracy. The same summary is served by the running app at `GET /evaluation` and shown in the UI under "How these
models were evaluated".

| Modality | Data | Evaluation | Result | How far to trust it |
|---|---|---|---|---|
| Questionnaire (Random Forest) | 52 real responses (low 8 / moderate 20 / high 24) | 4-fold stratified cross-validation, same folds for both candidates; selection by macro-F1 | accuracy 0.615 +/- 0.109, balanced accuracy 0.561 +/- 0.129, macro-F1 0.552 +/- 0.153 (chance for 3 classes is about 0.33; always predicting the largest class gives 0.46 accuracy) | Preliminary: tiny sample, wide fold-to-fold spread, target derived from one self-reported stress item (Q4, excluded from predictors) |
| Behavioral (Logistic Regression) | 900 **synthetic** rows | 80/20 hold-out split | accuracy 0.989 | Not meaningful as performance: synthetic classes are cleanly separated by construction |
| Voice stress risk | none | none | **no model** | Needs a dataset with defensible stress labels; see `docs/voice_model_training.md` |
| Vocal emotion (experimental, separate) | CREMA-D: 7,442 acted clips, 91 actors, 6 emotions | speaker-independent: trained on 73 actors, evaluated once on 18 unseen actors (1,469 clips); 5-fold grouped CV on the training actors | held-out accuracy **0.481**, macro-F1 **0.468** (chance 0.167); CV macro-F1 0.467 +/- 0.007. Per class F1: angry 0.63, sad 0.57, neutral 0.47, disgust 0.39, fearful 0.39, happy 0.36 | Acted speech by actors, not students; labels are the intended acted emotion, **not stress**. Shows the audio pipeline extracts informative features; says nothing about stress. Never fused into the assessment |
| Fusion | model outputs only | unit and integration tests (`tests/test_fusion_engine.py`, `tests/test_assessment_api.py`) | weighting, agreement, missing-modality policy and contract validation behave as specified | Tests verify the logic, not predictive validity |

Source reports: `results/voice_emotion/evaluation_report.json` (CREMA-D, incl. confusion matrix), `results/questionnaire/evaluation_report.txt` (+ confusion matrices, out-of-fold predictions,
feature importance), `results/behavioral/evaluation_report.txt`, `models/*/training_metadata.json`.

## What "tests assess predictions are working" means here

* Code correctness is covered by the automated suite (`python -m pytest tests src/backend/questionnaire/test_api.py`).
* Directional sanity checks exist (for example healthier daily habits score lower than poor habits, voice training does
  not look good on shuffled labels), but they do not replace validation on real labeled data.
* To strengthen the evidence: collect more real questionnaire responses and retrain/evaluate with
  `src.ai_model.questionnaire.train_models` / `evaluate_models` (needs the private `questionnaire_with_target.csv`);
  replace the synthetic behavioral data with real self-reports; collect labeled voice data and run
  `python -m src.ai_model.voice.train_voice` (it produces a held-out evaluation report automatically).
