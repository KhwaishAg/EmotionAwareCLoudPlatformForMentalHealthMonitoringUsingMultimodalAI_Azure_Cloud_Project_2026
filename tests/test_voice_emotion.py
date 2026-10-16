"""Emotion recognition (acted speech) stays separate from stress-risk.

Training data here is an in-memory toy table. These tests check wiring and the
separation guarantees, not model quality.
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.ai_model.fusion.fusion_engine import validate_modality_result
from src.ai_model.voice import emotion as emo
from src.ai_model.voice import predict as voice_predict
from src.ai_model.voice import train_emotion as te
from src.ai_model.voice import train_voice as tv
from src.ai_model.voice.features import COMBINED_FEATURE_NAMES
from src.ai_model.voice.predict import VoiceModelLoadError, predict_voice_from_features
from src.backend.assessment import api as api_module
from tests.voice_fixtures import full_feature_dict, write_test_wav

FEATURES = list(COMBINED_FEATURE_NAMES)
client = TestClient(api_module.app)


def emotion_frame(actors_per_class=22, clips=2, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for index, label in enumerate(emo.EMOTION_LABELS):
        for actor in range(actors_per_class):
            for _ in range(clips):
                values = rng.randn(len(FEATURES))
                values[index] += 3.0
                rows.append({"participant_id": f"{1000 + index * 100 + actor}", **dict(zip(FEATURES, values)), "target": label})
    return pd.DataFrame(rows)


def emotion_settings(**overrides):
    base = dict(dataset_kind="public_acted", min_participants_per_class=20, labels=emo.EMOTION_LABELS,
                task=emo.EMOTION_TASK, model_prefix="voice_emotion")
    base.update(overrides)
    return tv.TrainingSettings(**base)


@pytest.fixture(scope="module")
def trained():
    return tv.train_voice_model(emotion_frame(), emotion_settings())


@pytest.fixture
def emotion_dir(trained, tmp_path, monkeypatch):
    model, report, explain = trained
    tv.save_artifacts(model, report, explain, tmp_path)
    monkeypatch.setattr(emo, "EMOTION_MODELS_DIR", tmp_path)
    return tmp_path


class TestCremaParsing:
    def test_filename_parsing(self):
        assert te.parse_crema_filename("1001_DFA_ANG_XX.wav") == {
            "filename": "1001_DFA_ANG_XX.wav", "participant_id": "1001", "sentence": "DFA", "target": "angry"}
        assert te.parse_crema_filename("1091_WSI_NEU_XX.wav")["target"] == "neutral"

    @pytest.mark.parametrize("name", ["notes.wav", "1001_DFA_XXX_XX.wav", "abc_DFA_ANG_XX.wav", "1001.wav"])
    def test_unknown_files_skipped(self, name):
        assert te.parse_crema_filename(name) is None

    def test_all_codes_map_to_emotion_labels_only(self):
        assert set(te.CREMA_EMOTIONS.values()) == set(emo.EMOTION_LABELS)
        assert not set(te.CREMA_EMOTIONS.values()) & set(tv.RISK_LABELS)

    def test_build_metadata_from_directory(self, tmp_path):
        wavs = tmp_path / "AudioWAV"
        wavs.mkdir()
        for name in ("1001_DFA_ANG_XX.wav", "1001_IEO_HAP_HI.wav", "1002_DFA_SAD_XX.wav"):
            write_test_wav(wavs / name, seconds=1.2)
        (wavs / "readme.txt").write_text("x")
        frame = te.build_metadata(tmp_path)
        assert sorted(frame["target"]) == ["angry", "happy", "sad"]
        assert frame["participant_id"].nunique() == 2

    def test_missing_directory_and_empty_directory(self, tmp_path):
        with pytest.raises(tv.TrainingDataError):
            te.build_metadata(tmp_path / "nope")
        with pytest.raises(tv.TrainingDataError, match="No CREMA-D"):
            te.build_metadata(tmp_path)

    def test_short_clips_pass_emotion_audio_config(self, tmp_path):
        wav = write_test_wav(tmp_path / "1001_DFA_ANG_XX.wav", seconds=1.5)
        features = te._extract_one(str(wav))
        assert features is not None and set(features) == set(FEATURES)


class TestSeparationFromStress:
    def test_stress_trainer_rejects_emotion_labels(self):
        with pytest.raises(tv.TrainingDataError, match="Unsupported labels"):
            tv.validate_dataset(emotion_frame(), tv.TrainingSettings(dataset_kind="public_acted"))

    def test_emotion_trainer_rejects_stress_labels(self):
        frame = emotion_frame()
        frame["target"] = "low"
        with pytest.raises(tv.TrainingDataError, match="Unsupported labels|participants per class"):
            tv.validate_dataset(frame, emotion_settings())

    def test_report_records_task_and_dataset_kind(self, trained):
        report = trained[1]
        assert report["task"] == emo.EMOTION_TASK and report["dataset_kind"] == "public_acted"
        assert report["classes"] == list(emo.EMOTION_LABELS)
        assert report["model_name"] == "voice_emotion_random_forest"

    def test_emotion_model_is_refused_by_stress_loader(self, emotion_dir, monkeypatch):
        monkeypatch.setattr(voice_predict, "MODELS_DIR", emotion_dir)
        monkeypatch.setenv(voice_predict.ALLOW_DEMO_MODEL_ENV, "1")  # even with the demo override
        with pytest.raises(VoiceModelLoadError, match="not trained for stress-risk"):
            predict_voice_from_features(full_feature_dict())

    def test_stress_model_is_refused_by_emotion_loader(self, emotion_dir):
        report_path = emotion_dir / voice_predict.TRAINING_REPORT_FILENAME
        report = json.loads(report_path.read_text())
        report["task"] = "stress_risk"
        report_path.write_text(json.dumps(report))
        with pytest.raises(emo.EmotionModelLoadError):
            emo.load_emotion_model()

    def test_emotion_output_is_not_a_fusion_contract(self, emotion_dir):
        result = emo.predict_emotion_from_features(full_feature_dict())
        assert "risk_score" not in result and "risk_level" not in result
        with pytest.raises(ValueError):
            validate_modality_result(result)
        with pytest.raises(ValueError):
            from src.ai_model.fusion.fusion_engine import fuse_modalities
            fuse_modalities(voice=result)


class TestEmotionInference:
    def test_prediction_shape(self, emotion_dir):
        result = emo.predict_emotion_from_features(full_feature_dict())
        assert result["predicted_emotion"] in emo.EMOTION_LABELS
        assert set(result["probabilities"]) == set(emo.EMOTION_LABELS)
        assert sum(result["probabilities"].values()) == pytest.approx(1.0)
        assert result["experimental"] is True and "not a measure of stress" in result["notice"].lower()

    def test_audio_prediction(self, emotion_dir, tmp_path):
        result = emo.predict_emotion_from_audio(write_test_wav(tmp_path / "clip.wav", seconds=2.0))
        assert result["task"] == emo.EMOTION_TASK

    def test_not_trained(self, tmp_path, monkeypatch):
        monkeypatch.setattr(emo, "EMOTION_MODELS_DIR", tmp_path)
        with pytest.raises(emo.EmotionModelNotTrainedError):
            emo.load_emotion_model()

    def test_corrupt_artifact(self, emotion_dir):
        (emotion_dir / "voice_emotion_random_forest.joblib").write_bytes(b"garbage")
        with pytest.raises(emo.EmotionModelLoadError):
            emo.load_emotion_model()

    def test_bad_features(self, emotion_dir):
        with pytest.raises(ValueError, match="incomplete"):
            emo.predict_emotion_from_features({FEATURES[0]: 1.0})
        with pytest.raises(ValueError, match="Unsupported"):
            emo.predict_emotion_from_features(dict(full_feature_dict(), extra=1.0))


class TestCli:
    def test_trains_from_cached_features(self, tmp_path, capsys):
        csv = tmp_path / "features.csv"
        emotion_frame().to_csv(csv, index=False)
        code = te.main(["--features-csv", str(csv), "--output-dir", str(tmp_path / "m"), "--report-dir", str(tmp_path / "r")])
        assert code == 0
        report = json.loads((tmp_path / "m" / "voice_training_report.json").read_text())
        assert report["dataset"]["name"] == "CREMA-D" and report["task"] == emo.EMOTION_TASK
        assert any("never mapped" in item for item in report["dataset"]["limitations"])
        assert "not a stress" in capsys.readouterr().out.lower()

    def test_aborts_without_data(self, tmp_path, capsys):
        code = te.main(["--crema-dir", str(tmp_path / "missing"), "--features-csv", str(tmp_path / "none.csv"),
                        "--output-dir", str(tmp_path / "m")])
        assert code == 2 and not (tmp_path / "m").exists()


class TestApiSeparation:
    def test_emotion_reported_separately_and_not_fused(self, monkeypatch):
        monkeypatch.setattr(api_module, "predict_emotion_from_audio",
                            lambda path: {"task": "emotion_recognition", "experimental": True, "predicted_emotion": "happy",
                                          "confidence": 0.5, "probabilities": {}, "notice": "n"})
        monkeypatch.setattr(voice_predict, "MODELS_DIR", voice_predict.MODELS_DIR / "does_not_exist")
        import base64
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            encoded = base64.b64encode(write_test_wav(Path(folder) / "c.wav").read_bytes()).decode()
        body = client.post("/assessment", json={"voice": {"audio_base64": encoded}}).json()
        assert body["voice_emotion"] == {"status": "available", "result": body["voice_emotion"]["result"]}
        assert body["voice_emotion"]["result"]["predicted_emotion"] == "happy"
        assert body["modality_status"]["voice"] == "model_not_trained"
        assert body["assessment"]["status"] == "fusion_unavailable"  # emotion never feeds the assessment
        assert body["modality_details"] == {}

    def test_absent_without_voice_input(self):
        assert client.post("/assessment", json={}).json()["voice_emotion"] is None

    def test_untrained_emotion_model_status(self, monkeypatch, tmp_path):
        monkeypatch.setattr(emo, "EMOTION_MODELS_DIR", tmp_path)
        body = client.post("/assessment", json={"voice": {"features": full_feature_dict()}}).json()
        assert body["voice_emotion"] == {"status": "model_not_trained", "result": None}

    def test_evaluation_includes_emotion_caveat(self):
        summary = client.get("/evaluation").json()["voice_emotion"]
        assert "not a stress measure" in summary["caveat"].lower()
