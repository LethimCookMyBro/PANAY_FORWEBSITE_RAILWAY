# backend/app/errors.py
"""
Unified Error Response System

Provides consistent error responses across all API endpoints.
All errors follow the same structure for easy frontend handling.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from fastapi import HTTPException
from fastapi.responses import JSONResponse


# ============================================================================
# ERROR CODES
# ============================================================================

class ErrorCode:
    """Machine-readable error codes for frontend handling"""
    
    # Authentication errors (401)
    AUTH_INVALID_TOKEN = "AUTH_INVALID_TOKEN"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    AUTH_MISSING = "AUTH_MISSING"
    AUTH_INVALID_CREDENTIALS = "AUTH_INVALID_CREDENTIALS"
    
    # Authorization errors (403)
    FORBIDDEN = "FORBIDDEN"
    
    # Not found errors (404)
    NOT_FOUND = "NOT_FOUND"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    
    # Client errors (400)
    INVALID_REQUEST = "INVALID_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    
    # Service unavailable errors (503)
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    DB_UNAVAILABLE = "DB_UNAVAILABLE"
    EMBEDDER_UNAVAILABLE = "EMBEDDER_UNAVAILABLE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    # Server errors (500)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    LLM_ERROR = "LLM_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"


# ============================================================================
# ERROR RESPONSE MODEL
# ============================================================================

class ErrorResponse(BaseModel):
    """
    Unified error response format.
    
    Example:
        {
            "error": true,
            "code": "LLM_UNAVAILABLE",
            "message": "The AI service is temporarily unavailable",
            "request_id": "1fbf2890",
            "details": {"retry_after": 30}
        }
    """
    error: bool = True
    code: str
    message: str
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class AppException(Exception):
    """Base exception for application errors"""
    
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ServiceUnavailableError(AppException):
    """Raised when a required service is not available"""
    
    def __init__(self, service: str, details: Optional[Dict] = None):
        code_map = {
            "llm": ErrorCode.LLM_UNAVAILABLE,
            "database": ErrorCode.DB_UNAVAILABLE,
            "embedder": ErrorCode.EMBEDDER_UNAVAILABLE,
        }
        super().__init__(
            code=code_map.get(service.lower(), ErrorCode.SERVICE_UNAVAILABLE),
            message=f"The {service} service is temporarily unavailable",
            status_code=503,
            details=details
        )


class AuthenticationError(AppException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed", code: str = None):
        super().__init__(
            code=code or ErrorCode.AUTH_INVALID_TOKEN,
            message=message,
            status_code=401
        )


class NotFoundError(AppException):
    """Raised when a resource is not found"""
    
    def __init__(self, resource: str = "Resource"):
        super().__init__(
            code=ErrorCode.NOT_FOUND,
            message=f"{resource} not found",
            status_code=404
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_error_response(
    code: str,
    message: str,
    status_code: int = 500,
    request_id: Optional[str] = None,
    details: Optional[Dict] = None
) -> JSONResponse:
    """Create a JSONResponse with unified error format"""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            code=code,
            message=message,
            request_id=request_id,
            details=details
        ).model_dump()
    )
