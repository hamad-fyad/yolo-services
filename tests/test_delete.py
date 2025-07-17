import base64
import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from app import app, init_db,DB_PATH,add_user


class Test_Delete(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        add_user("test", "password")
        self.auth_headers = self.get_auth_headers("test", "password")
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def get_auth_headers(self, username, password):
        # Encode username:password in base64 for Basic Auth header
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded_credentials}"}
        
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





        
        

    
   