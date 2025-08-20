import unittest
from unittest.mock import patch
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from PIL import Image
import io
import os

import requests
from tests.helper_function import get_auth_headers
from app import app, init_db, DB_PATH, add_user

load_dotenv("/Users/hamadfyad/PycharmProjects/pythonProject522/5/pythonProject/yolo-services/secrets.env")  # loads from .env by default
api_key = os.getenv("PIXABAY_API_KEY")
class TestGetPredictionsByScore(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
       

        init_db()
        self.auth_headers = get_auth_headers("test", "password")

        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

        # Trigger a prediction to populate data
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        self.prediction_uid = response.json()["prediction_uid"]
    @patch('app.require_auth', return_value='1')
    def test_get_predictions_by_score_valid(self,mock_auth):
        pixabay_api_url = f"https://pixabay.com/api/?key={api_key}&q=person&image_type=photo"
        api_response = requests.get(pixabay_api_url)
        self.assertEqual(api_response.status_code, 200, "Pixabay API call failed")
        
        hits = api_response.json().get("hits")
        self.assertTrue(hits, "No image results from Pixabay")

        image_url = hits[0]["largeImageURL"]


        image_response = requests.get(image_url)
        self.assertEqual(image_response.status_code, 200, "Failed to download image from Pixabay")

       
        image_bytes = io.BytesIO(image_response.content)
    
        
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        prediction_uid = response.json()["prediction_uid"]
        response2 = self.client.get(
            "/predictions/score/0.0",
            headers=self.auth_headers
        )
        self.assertEqual(response2.status_code, 200)
        data = response2.json()
        self.assertTrue(any(pred["prediction_uid"] == prediction_uid for pred in data))
    @patch('app.require_auth', return_value='1')
    def test_get_predictions_by_high_score(self, mock_auth):
        # Try a high threshold, unlikely to match any detection
        response = self.client.get(
            "/predictions/score/0.99",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)  # Always a list
        self.assertTrue(all(pred["prediction_uid"] != self.prediction_uid for pred in data))
