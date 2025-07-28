import base64
import unittest
from fastapi.testclient import TestClient
from app import app, init_db, add_user, DB_PATH
import os

class TestLogin(unittest.TestCase):
    def setUp(self):
        init_db()
        # Add test user with plaintext password (your current scheme)
        
        self.client = TestClient(app)
        res = self.client.post("/signup", json={"username": "testuser", "password": "testpass"})    
        assert res.status_code == 200
    def test_successful_login(self):
        response = self.client.post("/login",auth=("testuser", "testpass"))
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        print(json_data)
        # self.assertIn("Username and password required", json_data)
        self.assertEqual(json_data["message"], f"User 'testuser' logged in successfully.")
        # Check if Authorization is Basic <base64>
        self.assertTrue(json_data["Authorization"].startswith("Basic "))
        token = json_data["Authorization"].split(" ")[1]
        decoded_token = base64.b64decode(token).decode("utf-8")
        username, password = decoded_token.split(":")
        self.assertEqual(username, "testuser")
        self.assertEqual(password, "testpass")

    def test_missing_username_or_password(self):
        response = self.client.post("/login", json={"username": "testuser"})
        print(response.json())
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

        response = self.client.post("/login", json={"password": "testpass"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_invalid_credentials(self):
        response = self.client.post("/login", json={"username": "wrong", "password": "wrongpass"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")


