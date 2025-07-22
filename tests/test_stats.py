import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from app import app, init_db,DB_PATH


class Test_Stats(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_stats_empty(self):
        response = self.client.get("/stats")
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, {"total_predictions":0,"average_confidence_score":0.0,"most_common_labels":{}})
    
    def test_stats_after_prediction(self):
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        response2= self.client.get("/stats")
        # Check response
        self.assertEqual(response2.status_code, 200)
        data = response2.json()
        self.assertEqual(data, {"total_predictions":1,"average_confidence_score":0.0,"most_common_labels":{}})


    




        
        

    
   