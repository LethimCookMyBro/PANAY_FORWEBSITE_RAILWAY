# backend/app/routes_auth.py
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Response, Request, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr

from app.db_helpers import (
    create_user,
    get_user_by_email,
    save_refresh_token,
    find_refresh_token,
    revoke_refresh_token_by_hash,
    get_user_by_id,
)
from app.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ✅ OAuth2 standard bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ============================================================
# Schemas
# ============================================================

class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class LoginIn(BaseModel):
    email: EmailStr
    password: str


# ============================================================
# Auth Endpoints
# ============================================================

@router.post("/register")
def register(payload: RegisterIn):
    if get_user_by_email(payload.email):
        raise HTTPException(status_code=409, detail="Email already registered")

    pwd_hash = hash_password(payload.password)
    user = create_user(
        email=payload.email,
        password_hash=pwd_hash,
        full_name=payload.full_name,
    )
    return {"id": user["id"], "email": user["email"]}


@router.post("/login")
def login(payload: LoginIn, response: Response, request: Request):
    user = get_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access = create_access_token(subject=str(user["id"]))
    refresh = create_refresh_token(subject=str(user["id"]))

    expires = datetime.utcnow() + timedelta(
        seconds=int(os.getenv("REFRESH_TOKEN_EXPIRE_SECONDS", "1209600"))
    )

    save_refresh_token(
        user_id=user["id"],
        token=refresh,
        ip=request.client.host if request.client else None,
        ua=request.headers.get("user-agent"),
        expires_at=expires,
    )

    cookie_secure = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        key="refresh_token",
        value=refresh,
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
        max_age=int(os.getenv("REFRESH_TOKEN_EXPIRE_SECONDS", "1209600")),
        path="/api/auth",
    )

    return {"access_token": access, "token_type": "bearer"}


@router.post("/refresh")
def refresh(request: Request, response: Response):
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="Missing refresh token")

    try:
        data = decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if data.get("typ") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")

    record = find_refresh_token(token)
    if not record or record.get("revoked"):
        raise HTTPException(status_code=401, detail="Refresh token revoked or not found")

    user_id = record["user_id"]

    access = create_access_token(subject=str(user_id))
    new_refresh = create_refresh_token(subject=str(user_id))

    save_refresh_token(
        user_id=user_id,
        token=new_refresh,
        ip=request.client.host if request.client else None,
        ua=request.headers.get("user-agent"),
        expires_at=datetime.utcnow()
        + timedelta(seconds=int(os.getenv("REFRESH_TOKEN_EXPIRE_SECONDS", "1209600"))),
    )

    revoke_refresh_token_by_hash(token)

    cookie_secure = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie(
        key="refresh_token",
        value=new_refresh,
        httponly=True,
        secure=cookie_secure,
        samesite="lax",
        max_age=int(os.getenv("REFRESH_TOKEN_EXPIRE_SECONDS", "1209600")),
        path="/api/auth",
    )

    return {"access_token": access, "token_type": "bearer"}


@router.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get("refresh_token")
    if token:
        revoke_refresh_token_by_hash(token)

    response.delete_cookie(key="refresh_token", path="/api/auth")
    return {"ok": True}


# ============================================================
# Dependency: get current user from access token
# ============================================================

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    if payload.get("typ") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not an access token",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user = get_user_by_id(int(user_id))
    if not user or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return user


@router.get("/me")
def me(current_user = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "full_name": current_user.get("full_name"),
    }
