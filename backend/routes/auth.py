from fastapi import APIRouter
from passlib.context import CryptContext


from backend.models.user_model import UserRegister, UserLogin
from backend.data.mongodb import users_collection

from backend.security.jwt_handler import create_access_token

router = APIRouter()

# Password Hashing
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


@router.post("/register")
def register(user: UserRegister):

    # Check if email already exists
    existing = users_collection.find_one(
        {"email": user.email}
    )

    if existing:
        return {
            "success": False,
            "message": "Email already exists"
        }

    # Hash password
    hashed_password = pwd_context.hash(user.password)

    # Save user
    users_collection.insert_one(
        {
            "name": user.name,
            "email": user.email,
            "password": hashed_password
        }
    )

    return {
        "success": True,
        "message": "User registered successfully"
    }
    
    

@router.post("/login")
def login(user: UserLogin):

    existing = users_collection.find_one(
        {"email": user.email}
    )

    if not existing:
        return {
            "success": False,
            "message": "Email not found"
        }

    if not pwd_context.verify(user.password, existing["password"]):
        return {
            "success": False,
            "message": "Invalid password"
        }

    token = create_access_token(
    {
        "sub": existing["email"]
    }
)

    return {
    "success": True,
    "message": "Login successful",
    "access_token": token,
    "token_type": "bearer",
    "user": {
        "name": existing["name"],
        "email": existing["email"]
    }
}    