# Voice model: training, evaluation and inference

## Status

**No voice model has been trained.** The repository contains no audio recordings, no labels and no model artifact
(`data/raw/audio/` and `models/voice/` hold only `.gitkeep`). The training pipeline below is implemented and tested on
in-memory toy data, but it has never been run on a real dataset. Until it is, the assessment API reports voice as
`model_not_trained` and fusion uses the other modalities.

Public speech-emotion corpora (CREMA-D, RAVDESS, SAVEE, TESS, EmoDB) contain **acted** emotions, not student stress.
They are used here only to validate the audio pipeline and to train a **separate emotion-recognition model** (see
"Experimental emotion model" below). Their emotion labels are **never** mapped to `low`/`moderate`/`high`, and a stress
model trained with `--dataset-kind public_acted` (or `synthetic_demo`) is refused at inference. The stress-risk voice
model stays unavailable until a dataset with defensible stress labels exists; the training pipeline and fusion contract
need no changes when it does.

## What the model consumes and produces

* Input: one recording -> preprocessing (mono, 16 kHz, silence trim, normalise; minimum duration 5 s) -> the **41
  combined acoustic features** in the fixed order `COMBINED_FEATURE_NAMES`
  (`src/ai_model/voice/features.py`: MFCC, pitch, energy, spectral).
* Output: the standardized fusion contract (`modality`, `risk_score`, `risk_level`, `confidence`,
  `available_features`, `total_features`, `key_factors`, `data_quality`) that `src/ai_model/fusion/fusion_engine.py`
  validates. The model must predict **exactly** the classes `low`, `moderate`, `high`;
  `risk_score = 0*P(low) + 0.5*P(moderate) + 1*P(high)`; confidence = half the top probability plus half the margin
  (same as the questionnaire and behavioral modalities).

## Dataset required

| Requirement | Detail |
|---|---|
| Recordings | WAV, 16 kHz mono preferred, 5 s minimum (30-60 s target) responding to the standard prompt in `dataset/README.md` |
| Metadata CSV | one row per recording with columns `filename`, `participant_id`, `target` (names configurable via flags) |
| Labels | `low` / `moderate` / `high`, derived from a defined, documented source (for example the questionnaire-based stress level for the same participant). **The team must decide and document this mapping; the code does not invent one.** |
| Size | at least 10 distinct participants per class (hard minimum enforced; far more is needed for a trustworthy estimate) |
| Participants | anonymous ids (`P001`...), several recordings per participant allowed, consent obtained, no PII in filenames/metadata |
| Storage | keep raw audio out of Git (`.gitignore` already excludes `data/raw/audio/*.wav|mp3|flac`) |

## Train and evaluate

```bash
python -m src.ai_model.voice.train_voice \
    --metadata data/raw/audio/metadata.csv --audio-dir data/raw/audio \
    --dataset-kind real_student --model-type random_forest
```

Useful options: `--features-csv` (skip audio extraction; CSV produced by `save_voice_dataset`), `--model-type
logistic_regression`, `--seed`, `--test-size`, `--min-participants-per-class`, `--output-dir`, `--report-dir`,
`--overwrite` (existing artifacts are never clobbered otherwise). `--dataset-kind` is mandatory and honest labeling is
part of the contract: `real_student`, `public_acted` or `synthetic_demo`. The trainer is label-set agnostic internally
(`TrainingSettings.labels/task`), which is how the emotion model reuses it.

What it does, in order: validates the dataset (labels, participant ids, finite features, minimum participants per
class) -> **participant-disjoint**, label-stratified train/held-out split with a fixed seed (no speaker in both sides) ->
grouped cross-validation on the training participants -> fits on the training split only -> evaluates **once** on the
held-out participants -> writes artifacts. It aborts with a clear message (exit code 2, nothing written) when the data
cannot support training.

Artifacts in `models/voice/` (model files are git-ignored by default; report and metadata are small JSON):

| File | Purpose |
|---|---|
| `voice_random_forest.joblib` (or `voice_logistic_regression.joblib`) | serialized model |
| `selected_model.json` | tells the loader which artifact to use |
| `voice_explainability_metadata.json` | top features (model association, not causation) |
| `voice_training_report.json` (+ copy in `results/voice/evaluation_report.json`) | dataset kind, counts, fingerprint of the feature table, seed, library versions, cross-validation and held-out metrics (accuracy, macro-F1, per-class, confusion matrix) |

Held-out metrics describe the supplied dataset only. They are not evidence of clinical validity or real-world accuracy.
Small datasets give wide uncertainty; do not quote the numbers without the participant counts.

## Inference and safety behavior

`predict_voice_from_features(features)` / `predict_voice_contract_from_audio(path)` and `POST /assessment` (`voice`
as `audio_base64` WAV or `features`):

| Situation | Result |
|---|---|
| No model artifact | `VoiceModelNotTrainedError` -> API `model_not_trained`; other modalities still fused |
| Corrupt/unreadable artifact, bad `selected_model.json`, or model trained on a different feature set | `VoiceModelLoadError` -> `model_unavailable` |
| Stress model trained with `--dataset-kind synthetic_demo` or `public_acted`, or any model whose report `task` is not `stress_risk` | refused (`model_unavailable`); the dataset-kind refusal can be overridden with `VOICE_ALLOW_DEMO_MODEL=1` (demos/tests only), the task refusal cannot |
| Missing, extra or non-numeric/non-finite features (not exactly the 41) | `ValueError` -> `invalid_input` (never imputed) |
| Invalid, corrupt, too-short or non-WAV audio | `PredictionAudioError` -> `invalid_input` |
| Model raises during inference | `PredictionModelError` -> `prediction_failed` (details logged server-side only) |

`GET /status` shows whether a voice model exists and, from the training report, its dataset kind, held-out macro-F1 and
held-out participant count, with a caveat note. A model without a training report is flagged as unknown provenance.

## Tests

`tests/test_voice_training.py` (validation, participant-disjoint split, reproducibility, leakage guard with shuffled
labels, artifact writing, CLI, provenance gate, safe loading), `tests/test_voice_contract.py`, `tests/test_assessment_api.py`
and the pre-existing `tests/test_voice*.py`, `tests/test_audio_pipeline.py`. These use toy data generated in memory to test
the code; they do not and cannot validate model quality.

## Experimental emotion model (CREMA-D) - separate from stress risk

* **What:** a six-class *acted emotion* classifier (angry, disgust, fearful, happy, neutral, sad) trained on
  [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (7,442 clips, 91 professional actors, 12 sentences;
  Cao et al. 2014; Open Database License). Code: `src/ai_model/voice/emotion.py` (inference) and `train_emotion.py`.
* **Purpose:** exercise and validate the preprocessing/feature-extraction pipeline on a real labeled corpus. It is not a
  stress or mental-health measure.
* **Separation, enforced in code:** own label set, own artifact directory (`models/voice_emotion/`), output is not the
  fusion contract (no `risk_score`/`risk_level`), it is never passed to `fuse_modalities`, and the stress loader refuses
  it by `task`. The API returns it in a distinct `voice_emotion` field (not in `modality_status`/`modality_details`) and
  the UI shows it in its own labeled box.
* **Evaluation:** speaker-independent: the held-out set contains only actors never seen in training. CREMA-D clips are
  1-5 s, so the 5 s minimum used for student recordings is relaxed to 1 s for this model (`EMOTION_AUDIO_CONFIG`).
* **Train:** place the AudioWAV files in `data/raw/audio/crema-d/` (git-ignored) and run
  `python -m src.ai_model.voice.train_emotion` (features are cached in `data/processed/crema_d_features.csv`).
* **Result (actual run):** 73 training actors / 18 held-out actors, held-out accuracy 0.481 and macro-F1 0.468 against a
  chance level of 0.167; 5-fold grouped CV macro-F1 0.467 +/- 0.007. Angry (F1 0.63) and sad (0.57) are recognised best;
  happy (0.36), fearful and disgust (0.39) are often confused (see the confusion matrix in the report). Full report:
  `models/voice_emotion/voice_training_report.json` (copy in `results/voice_emotion/`).
* **Limits:** acted speech by actors, not spontaneous speech and not students; labels are the intended acted emotion.
  Do not use its output as a stress label or describe it as measuring student stress.
