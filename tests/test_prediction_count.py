import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
import time
from datetime import datetime, timedelta
from app import app

class Test_Count(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_get_prediction_count_0(self):
        response = self.client.get("/prediction/count")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"count": 0})
    
    
    def test_predict_includes_processing_time(self):
        """Test that the predict endpoint returns count of prediction sessions"""
        
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        
        response1 = self.client.get("/prediction/count")
        self.assertEqual(response1.status_code, 200)
        self.assertEqual(response1.json(), {"count": 1})



    def test_count_older_8days(self):
        """Test that the predict endpoint returns count of prediction sessions"""
        
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        
        response1 = self.client.get("/prediction/count")
        self.assertEqual(response1.status_code, 200)
        self.assertEqual(response1.json(), {"count": 1})

        
        

    
   