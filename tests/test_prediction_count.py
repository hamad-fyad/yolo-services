import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from tests.helper_function import get_auth_headers
from app import app, init_db,DB_PATH,add_user


class Test_Count(unittest.TestCase):

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

    def test_prediction_count_empty(self):
        response = self.client.get("/prediction/count",headers=self.auth_headers)
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("count"), 0)
        
 

    def test_prediction_count_after_prediction(self):
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        response2= self.client.get("/prediction/count",headers=self.auth_headers)
        # Check response
        self.assertEqual(response2.status_code, 200)
        data = response2.json()
        self.assertEqual(data.get("count"), 1)




        
        

    
   