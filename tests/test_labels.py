import secrets
import sqlite3
import unittest
import requests 
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from app import app, init_db,DB_PATH
from dotenv import load_dotenv

load_dotenv("/Users/hamadfyad/PycharmProjects/pythonProject522/5/pythonProject/yolo-services/secrets.env")  # loads from .env by default
api_key = os.getenv("PIXABAY_API_KEY")
class Test_labels(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='png')
        self.image_bytes.seek(0)
    # def test_labels_empty(self):
        
    #     response1 = self.client.get("/prediction/labels")
    #     # Check response
    #     self.assertEqual(response1.status_code, 200)
    #     data = response1.json()
    #     self.assertEqual(data.get("labels"), [])

    
    def test_labels_after_prediction(self):
        """Test prediction endpoint using an image from Pixabay."""

        # Step 1: Call Pixabay API
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
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 200, "Prediction failed")

        print("Prediction response:", response.json())

   
        response2 = self.client.get("/labels")
        temp = response2.json()
        self.assertEqual(response2.status_code, 200)
        self.assertIn("person",temp)





            


                
                

            
        