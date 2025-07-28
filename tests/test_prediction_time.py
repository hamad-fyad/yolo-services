import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
import io
from tests.helper_function import get_auth_headers
from app import add_user, app, init_db

class TestProcessingTime(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        init_db()
        self.auth_headers = get_auth_headers("test", "password")
        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)
    @patch("app.require_auth", return_value="1")
    def test_predict_includes_processing_time(self, mock_auth):
        """Test that the predict endpoint returns processing time"""
        
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify new field exists
        self.assertIn("time_took", data)
        self.assertIsInstance(data["time_took"], (int, float))
        self.assertGreater(data["time_took"], 0)