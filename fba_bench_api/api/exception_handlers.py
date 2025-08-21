from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException

from fba_bench.core.logging import get_request_id
from fba_bench_api.api.errors import AppError

logger = logging.getLogger(__name__)


def _problem_detail(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    detail: Any = None,
    extra: Dict[str, Any] | None = None,
) -> JSONResponse:
    """
    RFC 7807-ish error envelope with correlation id.

    Fields:
      - type (string): machine-readable error type or code
      - title (string): human-readable summary
      - status (int): HTTP status code
      - detail (any): additional details or validation errors
      - instance (string): request path
      - request_id (string): correlation id for tracing
      - extra (object): optional additional metadata
    """
    rid = get_request_id() or "-"
    body: Dict[str, Any] = {
        "type": code,
        "title": message,
        "status": status_code,
        "detail": detail,
        "instance": str(request.url.path),
        "request_id": rid,
    }
    if extra:
        body["extra"] = extra
    return JSONResponse(content=body, status_code=status_code)


def _flatten_validation_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for err in exc.errors():
        items.append(
            {
                "loc": err.get("loc"),
                "msg": err.get("msg"),
                "type": err.get("type"),
                "input": err.get("input", None),
                "ctx": err.get("ctx", None),
            }
        )
    return items


def add_exception_handlers(app: FastAPI) -> None:
    """
    Register centralized exception handlers for FastAPI app.

    Handles:
      - RequestValidationError (422)
      - Starlette/FastAPI HTTPException (as-is status)
      - TimeoutError (504)
      - AppError (custom structured application errors)
      - Generic Exception (500)
    """

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:  # type: ignore[override]
        errors = _flatten_validation_errors(exc)
        logger.warning(
            "Request validation failed",
            extra={
                "request_id": get_request_id() or "-",
                "path": str(request.url.path),
                "errors": errors,
            },
        )
        return _problem_detail(
            request,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="validation_error",
            message="Request validation failed",
            detail=errors,
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:  # type: ignore[override]
        # Preserve status code and message
        status_code = exc.status_code
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        logger.info(
            "HTTP exception",
            extra={
                "request_id": get_request_id() or "-",
                "path": str(request.url.path),
                "status_code": status_code,
                "detail": exc.detail,
            },
        )
        return _problem_detail(
            request,
            status_code=status_code,
            code="http_error",
            message=message,
            detail=exc.detail,
        )

    @app.exception_handler(TimeoutError)
    async def on_timeout(request: Request, exc: TimeoutError) -> JSONResponse:  # type: ignore[override]
        logger.warning(
            "Operation timed out",
            extra={
                "request_id": get_request_id() or "-",
                "path": str(request.url.path),
            },
        )
        return _problem_detail(
            request,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            code="timeout",
            message="Operation timed out. Please retry.",
        )

    @app.exception_handler(AppError)
    async def on_app_error(request: Request, exc: AppError) -> JSONResponse:  # type: ignore[override]
        # Centralized handling for domain-specific application errors
        logger.info(
            "AppError",
            extra={
                "request_id": get_request_id() or "-",
                "path": str(request.url.path),
                "code": exc.code,
                "status_code": exc.status_code,
            },
        )
        return _problem_detail(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            detail=exc.detail,
        )

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:  # type: ignore[override]
        # Log with stack trace for unhandled exceptions
        logger.exception(
            "Unhandled error",
            extra={"request_id": get_request_id() or "-", "path": str(request.url.path)},
        )
        return _problem_detail(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_server_error",
            message="An unexpected error occurred",
        )