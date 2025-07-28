import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import os
from tests.helper_function import get_auth_headers
from app import app, init_db, DB_PATH, PREDICTED_DIR
import uuid

class TestGetPredictionImage(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        init_db()
        self.auth_headers = get_auth_headers("test", "password")

        # Setup dummy image file
        os.makedirs(PREDICTED_DIR, exist_ok=True)
        self.uid = str(uuid.uuid4())
        self.image_filename = f"{self.uid}.jpg"
        self.image_path = os.path.join(PREDICTED_DIR, self.image_filename)

        with open(self.image_path, "wb") as f:
            f.write(b"dummy predicted image content")

    @patch("app.query_prediction_image_by_uid")
    @patch("app.require_auth", return_value=1)
    @patch("os.path.exists", return_value=True)
    def test_get_prediction_image_success_jpeg(self, mock_exists, mock_auth, mock_query):
        mock_query.return_value = self.image_path
        response = self.client.get(
            f"/prediction/{self.uid}/image",
            headers={**self.auth_headers, "accept": "image/jpeg"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("image/jpeg"))

    @patch("app.query_prediction_image_by_uid")
    @patch("app.require_auth", return_value=1)
    def test_get_prediction_image_success_png(self, mock_auth, mock_query_prediction_image_by_uid):
        # Create a dummy .png file
        image_path = os.path.join(PREDICTED_DIR, f"{self.uid}.png")
        with open(image_path, "wb") as f:
            f.write(b"dummy png content")

        # Mock the DB call to return the path to that file
        mock_query_prediction_image_by_uid.return_value = image_path

        response = self.client.get(
            f"/prediction/{self.uid}/image",
            headers={**self.auth_headers, "accept": "image/png"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("image/png"))


    @patch("app.query_prediction_image_by_uid", return_value=None)
    @patch("app.require_auth", return_value=1)
    def test_get_prediction_image_not_found_in_db(self, mock_auth, mock_query):
        response = self.client.get(
            "/prediction/nonexistent-uid/image",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("detail"), "Prediction not found")

    @patch("app.query_prediction_image_by_uid")
    @patch("app.require_auth", return_value=1)
    @patch("os.path.exists", return_value=False)
    def test_get_prediction_image_file_not_found(self, mock_exists, mock_auth, mock_query):
        mock_query.return_value = self.image_path
        response = self.client.get(
            f"/prediction/{self.uid}/image",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("detail"), "Predicted image file not found")

    @patch("app.query_prediction_image_by_uid")
    @patch("app.require_auth", return_value=1)
    @patch("os.path.exists", return_value=True)
    def test_get_prediction_image_not_acceptable(self, mock_exists, mock_auth, mock_query):
        mock_query.return_value = self.image_path
        response = self.client.get(
            f"/prediction/{self.uid}/image",
            headers={**self.auth_headers, "accept": "application/json"}
        )
        self.assertEqual(response.status_code, 406)
        self.assertEqual(response.json().get("detail"), "Client does not accept an image format")

    def test_get_prediction_image_unauthorized(self):
        response = self.client.get(
            f"/prediction/{self.uid}/image"
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json().get("detail"), "Unauthorized")
