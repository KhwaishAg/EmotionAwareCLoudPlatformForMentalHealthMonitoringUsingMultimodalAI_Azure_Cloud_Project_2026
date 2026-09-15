"""Test-only API checks; fixtures are not research or training data."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from src.ai_model.questionnaire import predict as prediction_module
from src.ai_model.questionnaire.predict import _example_response
from src.backend.questionnaire.api import app


class QuestionnaireApiTests(unittest.TestCase):
    client = TestClient(app)

    def test_valid_prediction_contract(self) -> None:
        response = self.client.post("/predict/questionnaire", json={"responses": _example_response()})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"modality", "risk_score", "risk_level", "confidence", "available_features", "total_features", "key_factors", "data_quality"})

    def test_malformed_request(self) -> None:
        self.assertEqual(self.client.post("/predict/questionnaire", json={"unexpected": {}}).status_code, 422)

    def test_missing_required_data(self) -> None:
        self.assertEqual(self.client.post("/predict/questionnaire", json={"responses": {"Q1": "4th Year"}}).status_code, 422)

    def test_unavailable_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(prediction_module, "MODELS_DIR", Path(directory)):
            response = self.client.post("/predict/questionnaire", json={"responses": _example_response()})
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
