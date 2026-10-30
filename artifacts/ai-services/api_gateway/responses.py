"""
API Response Models

Standardized response format for all endpoints.
"""

from typing import Any, Dict, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper"""
    success: bool = True
    data: Optional[T] = None
    error: Optional[str] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"key": "value"},
                "error": None,
                "message": "Operation completed",
                "timestamp": "2024-01-01T00:00:00"
            }
        }


class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class APIErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[list[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid request",
                "error_code": "VALIDATION_ERROR",
                "details": [
                    {"code": "FIELD_REQUIRED", "message": "name field is required"}
                ],
                "timestamp": "2024-01-01T00:00:00"
            }
        }


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper"""
    success: bool = True
    data: list[T]
    pagination: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [{"id": "1"}],
                "pagination": {
                    "page": 1,
                    "page_size": 20,
                    "total": 100,
                    "total_pages": 5
                },
                "timestamp": "2024-01-01T00:00:00"
            }
        }


# Helper functions to create responses
def success_response(data: Any = None, message: str = None) -> Dict[str, Any]:
    """Create a success response"""
    return {
        "success": True,
        "data": data,
        "error": None,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }


def error_response(error: str, error_code: str = None, details: list = None, status_code: int = None) -> Dict[str, Any]:
    """
    Create an error response.
    
    Note: The status_code parameter is a hint for the caller but is not included
    in the response dict. Callers should raise HTTPException with this status_code.
    """
    response = {
        "success": False,
        "error": error,
        "error_code": error_code,
        "details": details,
        "timestamp": datetime.utcnow().isoformat()
    }
    # Remove None values
    return {k: v for k, v in response.items() if v is not None}


def paginated_response(data: list, page: int, page_size: int, total: int) -> Dict[str, Any]:
    """Create a paginated response"""
    return {
        "success": True,
        "data": data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        },
        "timestamp": datetime.utcnow().isoformat()
    }