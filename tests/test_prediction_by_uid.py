import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
from unittest.mock import patch,Mock

from tests.helper_function import get_auth_headers
from app import app, init_db, DB_PATH, add_user

    
class TestGetPredictionByUid(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Reset and initialize DB
        init_db()
        self.auth_headers = get_auth_headers("test", "password")

        # Create test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

        # Generate a prediction to get UID
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        print(response.json())
        self.prediction_uid = response.json()["prediction_uid"]


    @patch("app.query_detection_objects_by_prediction_uid", return_value=[
    Mock(id=1, label="cat", score=0.9, box=[0, 0, 100, 100]),
    Mock(id=2, label="dog", score=0.85, box=[10, 10, 80, 80])
])
    @patch("app.require_auth", return_value=1)  
    def test_get_prediction_by_uid_success(self, mock_auth, mock_query):
        response = self.client.get(
            f"/prediction/{self.prediction_uid}",
            headers=self.auth_headers
        )
        print(response.json())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction_uid"], self.prediction_uid)
        self.assertIn("original_image", data)
        self.assertIn("predicted_image", data)
        self.assertIn("detection_objects", data)

    @patch("app.require_auth", return_value=1)
    def test_get_prediction_by_uid_not_found(self, mock_auth):
        response = self.client.get(
            "/prediction/non_existing_uid",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")
