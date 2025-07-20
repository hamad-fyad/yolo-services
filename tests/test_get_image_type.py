import unittest
from fastapi.testclient import TestClient
import os
from tests.helper_function import get_auth_headers
from app import app, init_db, DB_PATH, add_user, UPLOAD_DIR, PREDICTED_DIR


class TestGetImage(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        add_user("test", "password")
        self.auth_headers = get_auth_headers("test", "password")

        # Create dummy image files for testing
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(PREDICTED_DIR, exist_ok=True)

        self.original_filename = "test_original.jpg"
        self.predicted_filename = "test_predicted.jpg"

        with open(os.path.join(UPLOAD_DIR, self.original_filename), "wb") as f:
            f.write(b"dummy original image content")

        with open(os.path.join(PREDICTED_DIR, self.predicted_filename), "wb") as f:
            f.write(b"dummy predicted image content")

    def tearDown(self):
        # Clean up created files
        try:
            os.remove(os.path.join(UPLOAD_DIR, self.original_filename))
            os.remove(os.path.join(PREDICTED_DIR, self.predicted_filename))
        except FileNotFoundError:
            return 
            

    def test_get_original_image_success(self):
        response = self.client.get(
            f"/image/original/{self.original_filename}",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("content-type", response.headers)
        self.assertTrue(response.headers["content-type"].startswith("application/octet-stream") or
                        response.headers["content-type"].startswith("image/"))

    def test_get_predicted_image_success(self):
        response = self.client.get(
            f"/image/predicted/{self.predicted_filename}",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("content-type", response.headers)
        self.assertTrue(response.headers["content-type"].startswith("application/octet-stream") or
                        response.headers["content-type"].startswith("image/"))

    def test_get_invalid_image_type(self):
        response = self.client.get(
            "/image/invalid_type/somefile.jpg",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "Invalid image type")

    def test_get_nonexistent_image(self):
        response = self.client.get(
            "/image/original/nonexistent.jpg",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json().get("detail"), "Image not found")

    def test_get_image_unauthorized(self):
        response = self.client.get(
            f"/image/original/{self.original_filename}"
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json().get("detail"), "Unauthorized")
