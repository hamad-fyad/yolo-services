import base64
import unittest
from fastapi.testclient import TestClient
from app import app, init_db, add_user, DB_PATH
import os

class TestLogin(unittest.TestCase):
    def setUp(self):
        # Remove existing DB and re-init for clean state
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()
        # Add test user with plaintext password (your current scheme)
        add_user("testuser", "testpass")
        self.client = TestClient(app)

    def test_successful_login(self):
        response = self.client.post("/login", json={"username": "testuser", "password": "testpass"})
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("Authorization", json_data)
        self.assertEqual(json_data["message"], "Login successful")
        # Check if Authorization is Basic <base64>
        self.assertTrue(json_data["Authorization"].startswith("Basic "))
        token = json_data["Authorization"].split(" ")[1]
        decoded = base64.b64decode(token).decode()
        self.assertEqual(decoded, "testuser:testpass")

    def test_missing_username_or_password(self):
        response = self.client.post("/login", json={"username": "testuser"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Username and password required")

        response = self.client.post("/login", json={"password": "testpass"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Username and password required")

    def test_invalid_credentials(self):
        response = self.client.post("/login", json={"username": "wrong", "password": "wrongpass"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid username or password")


