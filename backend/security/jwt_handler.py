from datetime import datetime, timedelta
from jose import jwt

# IMPORTANT:
# غيري الـ SECRET_KEY دي بعدين في ملف .env
SECRET_KEY = "DEPI_AI_PROJECT_SECRET_KEY_2026"

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60


def create_access_token(data: dict):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update(
        {
            "exp": expire
        }
    )

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return encoded_jwt