import unittest
from fastapi.testclient import TestClient
import os
from tests.helper_function import get_auth_headers
from app import app, init_db, DB_PATH, add_user, PREDICTED_DIR
import sqlite3
import uuid


class TestGetPredictionImage(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        add_user("test", "password")
        self.auth_headers = get_auth_headers("test", "password")

        # Create a dummy predicted image file and db entry
        os.makedirs(PREDICTED_DIR, exist_ok=True)

        self.uid = str(uuid.uuid4())
        self.image_filename = f"{self.uid}.jpg"
        self.image_path = os.path.join(PREDICTED_DIR, self.image_filename)

        with open(self.image_path, "wb") as f:
            f.write(b"dummy predicted image content")

        with sqlite3.connect(DB_PATH) as conn:
            # Insert prediction session record pointing to the dummy image
            conn.execute("""
                INSERT INTO prediction_sessions (uid, predicted_image) 
                VALUES (?, ?)
            """, (self.uid, self.image_path))

    def tearDown(self):
        try:
            os.remove(self.image_path)
        except FileNotFoundError:
            pass

    def test_get_prediction_image_success_jpeg(self):
        response = self.client.get(
            f"/prediction/{self.uid}/image",
            headers={**self.auth_headers, "accept": "image/jpeg"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("image/jpeg"))

    def test_get_prediction_image_success_png(self):
        # Accept header requesting png
        response = self.client.get(
            f"/prediction/{self.uid}/image",
            headers={**self.auth_headers, "accept": "image/png"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("image/png"))

    def test_get_prediction_image_not_found_in_db(self):
        response = self.client.get(
            "/prediction/nonexistent-uid/image",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("detail"), "Prediction not found")

    def test_get_prediction_image_file_not_found(self):
        # Remove the image file to simulate missing file
        os.remove(self.image_path)
        response = self.client.get(
            f"/prediction/{self.uid}/image",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("detail"), "Predicted image file not found")

    def test_get_prediction_image_not_acceptable(self):
        # Accept header not including supported image types
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
