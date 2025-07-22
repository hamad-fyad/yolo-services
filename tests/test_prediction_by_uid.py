import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from tests.helper_function import get_auth_headers
from app import app, init_db, DB_PATH, add_user


class TestGetPredictionByUid(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        add_user("test", "password")
        self.auth_headers = get_auth_headers("test", "password")

        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

        # Make a prediction to get a UID
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.prediction_uid = response.json()["prediction_uid"]

    def test_get_prediction_by_uid_success(self):
        print(self.prediction_uid,"self.prediction_uid")
        response = self.client.get(
            f"/prediction/{self.prediction_uid}", headers=self.auth_headers
        )
        print(response.json())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uid"], self.prediction_uid)
        self.assertIn("original_image", data)
        self.assertIn("predicted_image", data)
        self.assertIn("detection_objects", data)

    def test_get_prediction_by_uid_not_found(self):
        response = self.client.get(
            "/prediction/non_existing_uid", headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")
