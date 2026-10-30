"""
Custom Exceptions

Application-specific exceptions for better error handling.
"""

from typing import Optional, Dict, Any


class TrashTheTestException(Exception):
    """Base exception for TrashTheTest"""
    def __init__(
        self, 
        message: str, 
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)


# ==================== Authentication Exceptions ====================

class AuthenticationError(TrashTheTestException):
    """Authentication failed"""
    def __init__(self, message: str = "Authentication failed", details: Dict = None):
        super().__init__(message, "AUTH_ERROR", details)


class AuthorizationError(TrashTheTestException):
    """Authorization failed - user doesn't have permission"""
    def __init__(self, message: str = "Access denied", details: Dict = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)


class TokenExpiredError(AuthenticationError):
    """JWT token has expired"""
    def __init__(self, message: str = "Token expired"):
        super().__init__(message, "TOKEN_EXPIRED")


# ==================== Resource Exceptions ====================

class NotFoundError(TrashTheTestException):
    """Resource not found"""
    def __init__(self, resource: str, resource_id: str):
        message = f"{resource} not found: {resource_id}"
        super().__init__(message, "NOT_FOUND", {"resource": resource, "id": resource_id})


class DuplicateError(TrashTheTestException):
    """Resource already exists"""
    def __init__(self, resource: str, identifier: str):
        message = f"{resource} already exists: {identifier}"
        super().__init__(message, "DUPLICATE_ERROR", {"resource": resource})


class ValidationError(TrashTheTestException):
    """Request validation failed"""
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else None
        super().__init__(message, "VALIDATION_ERROR", details)


# ==================== Processing Exceptions ====================

class ProcessingError(TrashTheTestException):
    """General processing error"""
    def __init__(self, message: str, stage: Optional[str] = None):
        details = {"stage": stage} if stage else None
        super().__init__(message, "PROCESSING_ERROR", details)


class ExtractionError(ProcessingError):
    """Text extraction failed"""
    def __init__(self, message: str):
        super().__init__(message, "extraction")


class ChunkingError(ProcessingError):
    """Document chunking failed"""
    def __init__(self, message: str):
        super().__init__(message, "chunking")


class EmbeddingError(ProcessingError):
    """Embedding generation failed"""
    def __init__(self, message: str):
        super().__init__(message, "embedding")


class GenerationError(ProcessingError):
    """Content generation failed"""
    def __init__(self, message: str):
        super().__init__(message, "generation")


class TTSError(ProcessingError):
    """Text-to-speech generation failed"""
    def __init__(self, message: str):
        super().__init__(message, "tts")


# ==================== Job Exceptions ====================

class JobError(TrashTheTestException):
    """Job processing failed"""
    def __init__(self, job_id: str, message: str):
        details = {"job_id": job_id}
        super().__init__(message, "JOB_ERROR", details)


class JobNotFoundError(NotFoundError):
    """Job not found"""
    def __init__(self, job_id: str):
        super().__init__("Job", job_id)


# ==================== Storage Exceptions ====================

class StorageError(TrashTheTestException):
    """File storage error"""
    def __init__(self, message: str):
        super().__init__(message, "STORAGE_ERROR")


# ==================== Service Exceptions ====================

class ServiceUnavailableError(TrashTheTestException):
    """External service unavailable"""
    def __init__(self, service: str):
        message = f"Service unavailable: {service}"
        super().__init__(message, "SERVICE_UNAVAILABLE", {"service": service})


class RateLimitError(TrashTheTestException):
    """Rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "RATE_LIMIT_ERROR")


# Exception handler mapping
EXCEPTION_HANDLERS = {
    AuthenticationError: 401,
    AuthorizationError: 403,
    TokenExpiredError: 401,
    NotFoundError: 404,
    DuplicateError: 409,
    ValidationError: 400,
    ProcessingError: 500,
    JobError: 500,
    StorageError: 500,
    ServiceUnavailableError: 503,
    RateLimitError: 429,
}


def get_status_code(exception: TrashTheTestException) -> int:
    """Get HTTP status code for exception"""
    for exc_type, status_code in EXCEPTION_HANDLERS.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500