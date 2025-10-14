import jwt
import os
from dotenv import load_dotenv
import secrets

from datetime import datetime, timedelta

load_dotenv()

# Secret for signing
SECRET_KEY = os.getenv('JWT_SECRET_KEY')

# 1. Create a token
# payload is received from frontend and backend generates access token
payload = {
    "user_id": 123,
    "exp": datetime.utcnow() + timedelta(seconds=10)  # expires in 5 minutes
}
token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

generated_key = secrets.token_hex(128)

# if secret key is not part of decoded token, then it will fail
try:
    decoded = jwt.decode(token, generated_key, algorithms=["HS256"])
    print("Decoded:", decoded)
except jwt.ExpiredSignatureError:
    print("Token expired!")
except jwt.InvalidTokenError:
    print("Invalid token!")
