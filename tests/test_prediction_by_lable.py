import unittest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from PIL import Image
import io
import os

import requests
from tests.helper_function import get_auth_headers
from app import app, init_db, DB_PATH, add_user, labels



load_dotenv("/Users/hamadfyad/PycharmProjects/pythonProject522/5/pythonProject/yolo-services/secrets.env")  # loads from .env by default
api_key = os.getenv("PIXABAY_API_KEY")


class TestGetPredictionsByLabel(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        add_user("test", "password")
        self.auth_headers = get_auth_headers("test", "password")

        pixabay_api_url = f"https://pixabay.com/api/?key={api_key}&q=person&image_type=photo"
        api_response = requests.get(pixabay_api_url)
        self.assertEqual(api_response.status_code, 200, "Pixabay API call failed")

        hits = api_response.json().get("hits")
        self.assertTrue(hits, "No image results from Pixabay")

        
        image_url = hits[0]["largeImageURL"]


        image_response = requests.get(image_url)
        self.assertEqual(image_response.status_code, 200, "Failed to download image from Pixabay")

       
        self.image_bytes = io.BytesIO(image_response.content)


        # Perform a prediction to insert data
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.prediction_uid = response.json()["prediction_uid"]
        self.detected_labels = response.json()["labels"]

    def test_get_predictions_by_valid_label(self):
        valid_label = self.detected_labels[0]
        response = self.client.get(
            f"/predictions/label/{valid_label}",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(any(pred["uid"] == self.prediction_uid for pred in data))

    def test_get_predictions_by_invalid_label(self):
        invalid_label = "nonexistent_label"
       
        response = self.client.get(
            f"/predictions/label/{invalid_label}",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid image type")