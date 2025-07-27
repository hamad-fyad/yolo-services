import base64
import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from app import app,DB_PATH,init_db
from tests.helper_function import get_auth_headers



class Test_Delete(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        init_db()
        self.client.post("/signup", json={"username": "test", "password": "password"})
        self.auth_headers = get_auth_headers("test", "password")
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

   
        
    def test_detete_not_found(self):
        
        response2 = self.client.delete("/prediction/1",headers=self.auth_headers)
        self.assertEqual(response2.status_code, 404)
        self.assertEqual(response2.json()["detail"], "Prediction not found")

    
    def test_delete_prediction(self):
        response = self.client.post(
            "/predict",
            headers=self.auth_headers,
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        print(response.json())
        uid = response.json()["prediction_uid"]

        response2 = self.client.delete(f"/prediction/{uid}", headers=self.auth_headers) 
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response2.json()["message"], "Prediction session deleted")





        
        

    
   