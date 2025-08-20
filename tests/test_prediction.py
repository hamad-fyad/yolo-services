import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from tests.helper_function import get_auth_headers
from app import app, init_db,DB_PATH


class Test_Stats(unittest.TestCase):

    
    def setUp(self):
        self.client = TestClient(app)

        init_db()
        
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)
        self.client.post("/signup", json={"username": "test", "password": "password"})
        self.auth_headers = get_auth_headers("test", "password")
   
    def test_prediction_unsupported_format(self):
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.pdf", self.image_bytes, "application/pdf")}
        )
        print(response.json())
        self.assertEqual(response.status_code, 400)
    
    """ if results[0].boxes is None or results[0].boxes == []:
        return {
            "prediction_uid": uid, 
            "detection_count": 0,
            "labels": detected_labels,
            "time_took": time.time() - start_time
        }"""
    def test_prediction_unauthorized_and_image_no_lable(self):
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction_uid", data)
        self.assertIn("labels", data)
        self.assertIn("detection_count", data)
        self.assertIn("time_took", data)

    def test_prediction_valid_image(self):
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction_uid", data)
        self.assertIn("labels", data)
        self.assertIn("detection_count", data)
        self.assertIn("time_took", data)
    @patch("app.model")
    def test_prediction_inference_error(self, mock_model):
        mock_model.side_effect = Exception("YOLO is broken")

        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("Inference failed", data["detail"])

       
