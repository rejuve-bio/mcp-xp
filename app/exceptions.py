# app/exceptions.py
from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import status


class BadRequestException(HTTPException):
    def __init__(self, detail: str = "Bad request"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

class UnauthorizedException(HTTPException):
    def __init__(self, detail: str = "Unauthorized"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

class ForbiddenException(HTTPException):
    def __init__(self, detail: str = "Forbidden"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

class NotFoundException(HTTPException):
    def __init__(self, detail: str = "Not Found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class NotAcceptableException(HTTPException):
    def __init__(self, detail: str = "Not Acceptable"):
        super().__init__(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail=detail)

class RequestTimeoutException(HTTPException):
    def __init__(self, detail: str = "Request Timeout"):
        super().__init__(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=detail)

class ConflictException(HTTPException):
    def __init__(self, detail: str = "Conflict"):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)

class PayloadTooLargeException(HTTPException):
    def __init__(self, detail: str = "Payload Too Large"):
        super().__init__(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=detail)

class InternalServerErrorException(HTTPException):
    def __init__(self, detail: str = "Internal Server Error"):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

class NotImplementedException(HTTPException):
    def __init__(self, detail: str = "Not Implemented"):
        super().__init__(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=detail)

class BadGatewayException(HTTPException):
    def __init__(self, detail: str = "Bad Gateway"):
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)

class ServiceUnavailableException(HTTPException):
    def __init__(self, detail: str = "Service Unavailable"):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)

class GatewayTimeoutException(HTTPException):
    def __init__(self, detail: str = "Gateway Timeout"):
        super().__init__(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail)


# ------------------------
# Centralized Exception Handlers
# ------------------------

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles all HTTPExceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """Handles all other unhandled exceptions"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "detail": str(exc)
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """/v1 answers an invalid body with 400, keeping 422 for inputs Galaxy rejects."""
    if request.url.path.startswith("/v1/"):
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})
    return await request_validation_exception_handler(request, exc)
