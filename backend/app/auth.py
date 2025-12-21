import os
from datetime import datetime, timedelta
from typing import Optional

from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import HTTPException, status

# ----------------------
# Password hashing
# ----------------------
PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    if len(pwd_bytes) > 72:
        pwd_bytes = pwd_bytes[:72]
    return PWD_CTX.hash(pwd_bytes.decode("utf-8", errors="ignore"))


def verify_password(password: str, hashed: str) -> bool:
    return PWD_CTX.verify(password, hashed)


# ----------------------
# JWT config
# ----------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALGO = os.getenv("JWT_ALGORITHM", "HS256")

ACCESS_EXPIRE = int(os.getenv("ACCESS_TOKEN_EXPIRE_SECONDS", "900"))
REFRESH_EXPIRE = int(os.getenv("REFRESH_TOKEN_EXPIRE_SECONDS", "1209600"))


# ----------------------
# Token helpers
# ----------------------
def create_access_token(subject: str, expires_seconds: Optional[int] = None) -> str:
    exp = datetime.utcnow() + timedelta(seconds=(expires_seconds or ACCESS_EXPIRE))
    payload = {
        "sub": str(subject),
        "exp": exp,
        "typ": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def create_refresh_token(subject: str, expires_seconds: Optional[int] = None) -> str:
    exp = datetime.utcnow() + timedelta(seconds=(expires_seconds or REFRESH_EXPIRE))
    payload = {
        "sub": str(subject),
        "exp": exp,
        "typ": "refresh",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
