"""Simplified authentication for the Kannaka Cannon dashboard.

Phase 1 uses a simple JWT-based dev-mode authentication. A dev-login
endpoint auto-authenticates without requiring email or password,
setting an HTTP-only session cookie with a 30-day TTL.

Future phases will implement magic-link email authentication.
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from jose import JWTError, jwt

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Publicly-known dev secret. Signing production tokens with this would let
# anyone forge session cookies, so get_jwt_secret() refuses to use it when
# dev mode is off.
_DEFAULT_DEV_SECRET = "clipcannon-dev-secret-not-for-production"
_JWT_ALGORITHM = "HS256"
_SESSION_COOKIE_NAME = "clipcannon_session"
_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days
_DEV_USER_ID = "dev-user-001"
_DEV_USER_EMAIL = "dev@clipcannon.local"


def is_dev_mode() -> bool:
    """Check whether the dashboard is running in development mode.

    Defaults to OFF (production): set CLIPCANNON_DEV_MODE=1 for local
    development to enable auto-login and the insecure default JWT secret.

    Returns:
        True if CLIPCANNON_DEV_MODE is set to a truthy value.
    """
    return os.environ.get("CLIPCANNON_DEV_MODE", "0").lower() in ("1", "true", "yes")


def get_jwt_secret() -> str:
    """Resolve the JWT signing secret, enforcing a real secret in production.

    In production (dev mode off) CLIPCANNON_JWT_SECRET must be set to a
    non-default value; otherwise session tokens could be forged with the
    publicly-known dev secret. Raises in that case so the server refuses
    to start or sign/verify tokens.

    Returns:
        The JWT signing secret to use.

    Raises:
        RuntimeError: If dev mode is off and the secret is unset or the
            insecure default.
    """
    secret = os.environ.get("CLIPCANNON_JWT_SECRET", _DEFAULT_DEV_SECRET)
    if not is_dev_mode() and secret == _DEFAULT_DEV_SECRET:
        raise RuntimeError(
            "CLIPCANNON_JWT_SECRET is unset or set to the insecure default "
            "while CLIPCANNON_DEV_MODE is off. Set CLIPCANNON_JWT_SECRET to a "
            "strong, unique value for production, or set CLIPCANNON_DEV_MODE=1 "
            "for local development."
        )
    return secret


def create_session_token(user_id: str, email: str) -> str:
    """Create a JWT session token.

    Args:
        user_id: The user identifier.
        email: The user email address.

    Returns:
        Encoded JWT token string.
    """
    now = int(time.time())
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + _SESSION_TTL_SECONDS,
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=_JWT_ALGORITHM)


def verify_session_token(token: str) -> dict[str, str | int] | None:
    """Verify and decode a JWT session token.

    Args:
        token: The JWT token string to verify.

    Returns:
        Decoded payload dictionary if valid, None if invalid or expired.
    """
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[_JWT_ALGORITHM])
        return payload  # type: ignore[return-value]
    except JWTError as exc:
        logger.debug("JWT verification failed: %s", exc)
        return None


def get_current_user(request: Request) -> dict[str, str | int] | None:
    """Extract the current user from the session cookie.

    Args:
        request: The incoming FastAPI request.

    Returns:
        User payload dictionary if authenticated, None otherwise.
    """
    token = request.cookies.get(_SESSION_COOKIE_NAME)
    if token is None:
        # In dev mode, auto-authenticate
        if is_dev_mode():
            return {
                "sub": _DEV_USER_ID,
                "email": _DEV_USER_EMAIL,
                "iat": int(time.time()),
                "exp": int(time.time()) + _SESSION_TTL_SECONDS,
            }
        return None
    return verify_session_token(token)


def require_auth(func: Callable[..., object]) -> Callable[..., object]:
    """Decorator that requires authentication on a route handler.

    In dev mode, requests are auto-authenticated. In production mode,
    a valid session cookie is required.

    Args:
        func: The route handler function to protect.

    Returns:
        Wrapped function that checks authentication.
    """

    @wraps(func)
    async def wrapper(request: Request, *args: object, **kwargs: object) -> object:
        user = get_current_user(request)
        if user is None:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Authentication required. Use /auth/dev-login in dev mode.",
                    },
                },
            )
        request.state.user = user
        return await func(request, *args, **kwargs)

    return wrapper


@router.get("/dev-login")
async def dev_login(response: Response) -> dict[str, str | bool]:
    """Auto-login endpoint for development mode.

    Creates a session token and sets an HTTP-only cookie.
    Only works when CLIPCANNON_DEV_MODE is enabled.

    Args:
        response: The FastAPI response object for setting cookies.

    Returns:
        Dictionary with login status and user info.
    """
    if not is_dev_mode():
        return {
            "success": False,
            "message": "Dev login is only available in development mode.",
        }

    token = create_session_token(_DEV_USER_ID, _DEV_USER_EMAIL)

    response.set_cookie(
        key=_SESSION_COOKIE_NAME,
        value=token,
        max_age=_SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=False,  # Allow HTTP in dev mode
    )

    logger.info("Dev login: user=%s", _DEV_USER_ID)

    return {
        "success": True,
        "user_id": _DEV_USER_ID,
        "email": _DEV_USER_EMAIL,
        "message": "Logged in via dev mode. Session cookie set (30-day TTL).",
    }


@router.get("/me")
async def get_me(request: Request) -> dict[str, str | bool | None]:
    """Get current authenticated user info.

    Args:
        request: The incoming request with session cookie.

    Returns:
        User information if authenticated, error otherwise.
    """
    user = get_current_user(request)
    if user is None:
        return {
            "authenticated": False,
            "user_id": None,
            "email": None,
        }

    return {
        "authenticated": True,
        "user_id": str(user.get("sub", "")),
        "email": str(user.get("email", "")),
        "dev_mode": is_dev_mode(),
    }


@router.get("/logout")
async def logout(response: Response) -> dict[str, str | bool]:
    """Clear the session cookie.

    Args:
        response: The FastAPI response object for deleting cookies.

    Returns:
        Logout confirmation.
    """
    response.delete_cookie(_SESSION_COOKIE_NAME)
    return {
        "success": True,
        "message": "Session cleared.",
    }
