import base64 

def get_auth_headers(username, password):
        # Encode username:password in base64 for Basic Auth header
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded_credentials}"}