"""Voice training pipeline, model provenance gating and safe model loading.

All datasets here are generated in memory purely to test the pipeline. They are
labeled ``synthetic_demo``, are never a real predictor, and the metrics they
produce say nothing about real-world performance.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.ai_model.fusion.fusion_engine import validate_modality_result
from src.ai_model.voice import predict as voice_predict
from src.ai_model.voice import train_voice as tv
from src.ai_model.voice.features import COMBINED_FEATURE_NAMES
from src.ai_model.voice.model import VoiceRandomForest
from src.ai_model.voice.predict import (
    PredictionModelError,
    VoiceModelLoadError,
    VoiceModelNotTrainedError,
    predict_voice_from_features,
)
from tests.voice_fixtures import full_feature_dict, write_test_wav

FEATURES = list(COMBINED_FEATURE_NAMES)


def make_frame(participants_per_class=12, recordings=3, seed=0, shift=2.0):
    """Learnable toy frame: class shifts a few features. Pipeline test data only."""
    rng = np.random.RandomState(seed)
    rows = []
    for level, label in enumerate(tv.RISK_LABELS):
        for p in range(participants_per_class):
            for _ in range(recordings):
                values = rng.randn(len(FEATURES))
                values[:5] += shift * level
                rows.append({"participant_id": f"{label[:1]}{p:03d}", **dict(zip(FEATURES, values)), "target": label})
    return pd.DataFrame(rows)


def settings(**overrides):
    base = dict(dataset_kind="synthetic_demo", min_participants_per_class=5)
    base.update(overrides)
    return tv.TrainingSettings(**base)


@pytest.fixture(scope="module")
def trained():
    return tv.train_voice_model(make_frame(), settings())


class TestDatasetValidation:
    def test_valid_frame_passes(self):
        assert len(tv.validate_dataset(make_frame(), settings())) == 3 * 12 * 3

    def test_unsupported_labels_rejected(self):
        frame = make_frame()
        frame.loc[0, "target"] = "angry"
        with pytest.raises(tv.TrainingDataError, match="Unsupported labels"):
            tv.validate_dataset(frame, settings())

    def test_labels_are_normalized(self):
        frame = make_frame()
        frame["target"] = frame["target"].str.upper()
        assert set(tv.validate_dataset(frame, settings())["target"]) == set(tv.RISK_LABELS)

    def test_missing_columns_rejected(self):
        with pytest.raises(tv.TrainingDataError, match="missing required columns"):
            tv.validate_dataset(make_frame().drop(columns=[FEATURES[0]]), settings())

    def test_missing_participant_rejected(self):
        frame = make_frame()
        frame.loc[0, "participant_id"] = None
        with pytest.raises(tv.TrainingDataError, match="participant id"):
            tv.validate_dataset(frame, settings())

    def test_non_finite_features_rejected(self):
        frame = make_frame()
        frame.loc[3, FEATURES[2]] = np.nan
        with pytest.raises(tv.TrainingDataError, match="non-finite"):
            tv.validate_dataset(frame, settings())

    def test_too_few_participants_rejected_not_silently_trained(self):
        with pytest.raises(tv.TrainingDataError, match="Not enough distinct participants"):
            tv.validate_dataset(make_frame(participants_per_class=3), settings())

    def test_missing_class_rejected(self):
        frame = make_frame()
        with pytest.raises(tv.TrainingDataError, match="participants per class"):
            tv.validate_dataset(frame[frame["target"] != "high"], settings())

    def test_bad_settings_rejected(self):
        with pytest.raises(tv.TrainingDataError):
            tv.validate_dataset(make_frame(), settings(dataset_kind="made_up"))
        with pytest.raises(tv.TrainingDataError):
            tv.validate_dataset(make_frame(), settings(model_type="svm"))
        with pytest.raises(tv.TrainingDataError):
            tv.validate_dataset(make_frame(), settings(test_size=0.9))


class TestSplitAndTraining:
    def test_split_is_participant_disjoint_and_stratified(self):
        data = tv.validate_dataset(make_frame(), settings())
        train_idx, test_idx = tv.participant_split(data, settings())
        train_ids = set(data.iloc[train_idx]["participant_id"])
        test_ids = set(data.iloc[test_idx]["participant_id"])
        assert not train_ids & test_ids
        assert set(data.iloc[train_idx]["target"]) == set(tv.RISK_LABELS) == set(data.iloc[test_idx]["target"])

    def test_report_contents(self, trained):
        model, report, explain = trained
        assert report["dataset_kind"] == "synthetic_demo" and report["synthetic_data"] is True
        assert report["classes"] == list(tv.RISK_LABELS)
        assert report["feature_count"] == len(FEATURES)
        assert report["split"]["train"]["participants"] + report["split"]["held_out_test"]["participants"] == 36
        assert report["cross_validation_on_train"]["folds"] >= 2
        assert "not evidence of clinical validity" in report["disclaimer"]
        assert 0.0 <= report["held_out_metrics"]["macro_f1"] <= 1.0
        assert report["held_out_metrics"]["confusion_matrix"]["labels"]
        assert explain["top_features"]
        assert model.feature_names_ == FEATURES
        assert set(model.classes_) == set(tv.RISK_LABELS)

    def test_pipeline_learns_separable_toy_data(self, trained):
        # Only proves the pipeline fits and evaluates correctly on a separable toy problem.
        assert trained[1]["held_out_metrics"]["accuracy"] > 0.6

    def test_training_is_reproducible(self, trained):
        again = tv.train_voice_model(make_frame(), settings())[1]
        assert again["held_out_metrics"]["accuracy"] == trained[1]["held_out_metrics"]["accuracy"]
        assert again["dataset_fingerprint_sha256"] == trained[1]["dataset_fingerprint_sha256"]

    def test_logistic_regression_supported(self):
        _, report, explain = tv.train_voice_model(make_frame(), settings(model_type="logistic_regression"))
        assert report["model_name"] == "voice_logistic_regression"
        assert explain["method"] == "mean_absolute_logistic_coefficient"

    def test_random_labels_do_not_look_good(self):
        """Guards against evaluation leakage: shuffled labels must not score well."""
        frame = make_frame(participants_per_class=15)
        shuffled = frame.copy()
        by_participant = shuffled.groupby("participant_id")["target"].first()
        permuted = dict(zip(by_participant.index, np.random.RandomState(1).permutation(by_participant.values)))
        shuffled["target"] = shuffled["participant_id"].map(permuted)
        report = tv.train_voice_model(shuffled, settings())[1]
        assert report["held_out_metrics"]["accuracy"] < 0.7


class TestArtifactsAndCli:
    def test_save_artifacts_and_no_clobber(self, trained, tmp_path):
        model, report, explain = trained
        paths = tv.save_artifacts(model, report, explain, tmp_path / "m", tmp_path / "r")
        assert all(path.exists() for path in paths.values())
        with pytest.raises(FileExistsError):
            tv.save_artifacts(model, report, explain, tmp_path / "m")
        tv.save_artifacts(model, report, explain, tmp_path / "m", overwrite=True)

    def test_cli_trains_from_feature_csv(self, tmp_path, capsys):
        csv = tmp_path / "features.csv"
        make_frame().to_csv(csv, index=False)
        code = tv.main(["--features-csv", str(csv), "--dataset-kind", "synthetic_demo", "--min-participants-per-class", "5",
                        "--output-dir", str(tmp_path / "models"), "--report-dir", str(tmp_path / "results")])
        assert code == 0
        assert (tmp_path / "models" / "voice_random_forest.joblib").exists()
        assert (tmp_path / "results" / "evaluation_report.json").exists()
        assert "synthetic_demo" in capsys.readouterr().err

    def test_cli_aborts_cleanly_without_data(self, tmp_path, capsys):
        code = tv.main(["--metadata", str(tmp_path / "nope.csv"), "--dataset-kind", "real_student",
                        "--output-dir", str(tmp_path / "models")])
        assert code == 2
        assert "Training aborted" in capsys.readouterr().err
        assert not (tmp_path / "models").exists()

    def test_cli_refuses_undersized_dataset(self, tmp_path, capsys):
        csv = tmp_path / "features.csv"
        make_frame(participants_per_class=3).to_csv(csv, index=False)
        code = tv.main(["--features-csv", str(csv), "--dataset-kind", "real_student", "--output-dir", str(tmp_path / "m")])
        assert code == 2 and not (tmp_path / "m").exists()

    def test_audio_metadata_path_extracts_features(self, tmp_path):
        audio = tmp_path / "audio"
        audio.mkdir()
        rows = []
        for index, label in enumerate(tv.RISK_LABELS):
            write_test_wav(audio / f"r{index}.wav", seconds=5.5)
            rows.append({"filename": f"r{index}.wav", "participant_id": f"P{index}", "target": label})
        pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)
        frame = tv.load_training_frame(tmp_path / "metadata.csv", audio, None, settings())
        assert set(FEATURES) <= set(frame.columns) and len(frame) == 3
        with pytest.raises(tv.TrainingDataError):  # three recordings can never support training
            tv.validate_dataset(frame, settings())


@pytest.fixture
def demo_models_dir(trained, tmp_path, monkeypatch):
    model, report, explain = trained
    tv.save_artifacts(model, report, explain, tmp_path)
    monkeypatch.setattr(voice_predict, "MODELS_DIR", tmp_path)
    monkeypatch.delenv(voice_predict.ALLOW_DEMO_MODEL_ENV, raising=False)
    return tmp_path


class TestProvenanceGate:
    def test_synthetic_demo_model_is_not_served(self, demo_models_dir):
        with pytest.raises(VoiceModelLoadError, match="not defensible stress"):
            predict_voice_from_features(full_feature_dict())

    def test_demo_model_requires_explicit_override(self, demo_models_dir, monkeypatch):
        monkeypatch.setenv(voice_predict.ALLOW_DEMO_MODEL_ENV, "1")
        result = predict_voice_from_features(full_feature_dict())
        validate_modality_result(result)
        assert result["key_factors"], "explainability metadata should be used"

    def test_acted_emotion_data_is_not_served_as_stress(self, demo_models_dir):
        report_path = demo_models_dir / voice_predict.TRAINING_REPORT_FILENAME
        report = json.loads(report_path.read_text())
        report["dataset_kind"] = "public_acted"
        report_path.write_text(json.dumps(report))
        with pytest.raises(VoiceModelLoadError, match="not defensible stress"):
            predict_voice_from_features(full_feature_dict())

    def test_non_stress_task_model_is_never_served(self, demo_models_dir, monkeypatch):
        monkeypatch.setenv(voice_predict.ALLOW_DEMO_MODEL_ENV, "1")  # even with the demo override
        report_path = demo_models_dir / voice_predict.TRAINING_REPORT_FILENAME
        report = json.loads(report_path.read_text())
        report["task"] = "emotion_recognition"
        report_path.write_text(json.dumps(report))
        with pytest.raises(VoiceModelLoadError, match="not trained for stress-risk"):
            predict_voice_from_features(full_feature_dict())

    def test_real_dataset_kind_is_served(self, demo_models_dir):
        report_path = demo_models_dir / voice_predict.TRAINING_REPORT_FILENAME
        report = json.loads(report_path.read_text())
        report["dataset_kind"], report["synthetic_data"] = "real_student", False
        report_path.write_text(json.dumps(report))
        validate_modality_result(predict_voice_from_features(full_feature_dict()))


class TestSafeLoading:
    @pytest.fixture
    def models_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(voice_predict, "MODELS_DIR", tmp_path)
        return tmp_path

    def test_no_artifacts_means_not_trained(self, models_dir):
        with pytest.raises(VoiceModelNotTrainedError):
            voice_predict.load_selected_voice_model()

    def test_corrupt_artifact(self, models_dir):
        (models_dir / "voice_model.joblib").write_bytes(b"garbage")
        with pytest.raises(VoiceModelLoadError):
            voice_predict.load_selected_voice_model()

    def test_wrong_object_in_artifact(self, models_dir):
        joblib.dump({"not": "a model"}, models_dir / "voice_model.joblib")
        with pytest.raises(VoiceModelLoadError):
            voice_predict.load_selected_voice_model()

    def test_model_trained_on_different_features_rejected(self, models_dir):
        rng = np.random.RandomState(0)
        other = pd.DataFrame(rng.randn(30, 4), columns=list("abcd"))
        VoiceRandomForest(random_state=0).fit(other, np.array(list(tv.RISK_LABELS) * 10)).save(models_dir / "voice_model.joblib")
        with pytest.raises(VoiceModelLoadError, match="different feature set"):
            voice_predict.load_selected_voice_model()

    @pytest.mark.parametrize("content", ["{not json", '{"selected_model": "../evil"}', '{"selected_model": ""}', "[]"])
    def test_bad_selector_rejected(self, models_dir, content):
        (models_dir / "selected_model.json").write_text(content)
        with pytest.raises(VoiceModelLoadError):
            voice_predict.load_selected_voice_model()

    def test_inference_failure_is_wrapped(self):
        class Broken:
            def predict_proba_dict(self, _):
                raise RuntimeError("boom")

        with pytest.raises(PredictionModelError, match="inference failed"):
            predict_voice_from_features(full_feature_dict(), model=Broken())

    def test_wrong_feature_counts(self, trained):
        model = trained[0]
        too_few = dict(list(full_feature_dict().items())[:-1])
        too_many = dict(full_feature_dict(), extra=1.0)
        with pytest.raises(ValueError, match="incomplete"):
            predict_voice_from_features(too_few, model=model)
        with pytest.raises(ValueError, match="Unsupported"):
            predict_voice_from_features(too_many, model=model)
