"""Password hashing and JWT helpers.

Pure functions only: no web framework and no database code, so they are easy
to test in isolation.

Passwords
  Hashed with scrypt from Python's standard library (a memory-hard, salted
  hash recommended for password storage). The stored value looks like
  ``scrypt$N$r$p$salt$hash`` so the cost settings can be raised later and old
  hashes still verify. Hashing takes tens of milliseconds on purpose, so
  callers on the event loop should run it with ``asyncio.to_thread``.

Tokens
  Signed JWTs (HS256) using PyJWT. A token carries only the user id (``sub``),
  the issue time (``iat``) and the expiry (``exp``). Decoding insists on
  ``exp`` and ``sub`` and only accepts HS256, which blocks "alg: none" and
  algorithm-confusion tricks.

Configuration (environment variables)
  JWT_SECRET                   signing key; use a long random value
  ACCESS_TOKEN_EXPIRE_MINUTES  token lifetime (default 30)

  If JWT_SECRET is not set, a random secret is generated for the lifetime of
  the process and a warning is logged. That is safe, but every restart (and
  every auto-reload in development) invalidates existing tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_MINUTES = 30.0
MIN_SECRET_LENGTH = 32

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16

_fallback_secret: str | None = None
_dummy_hash: str | None = None


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = os.urandom(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return "$".join(
        [
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(password: str, stored_hash: str) -> bool:
    """True if ``password`` matches ``stored_hash``. Never raises on bad input."""
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored_hash.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(digest_b64, validate=True)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def verify_against_dummy(password: str) -> None:
    """Spend the same time as a real check.

    Called when a login names a user that does not exist, so response time
    does not reveal which usernames are registered.
    """
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password("not-a-real-password")
    verify_password(password, _dummy_hash)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


class InvalidTokenError(Exception):
    """The token is missing, malformed, expired, or not signed by us."""


def get_jwt_secret() -> str:
    global _fallback_secret
    secret = os.environ.get("JWT_SECRET")
    if secret:
        return secret
    if _fallback_secret is None:
        _fallback_secret = secrets.token_urlsafe(48)
        logger.warning(
            "JWT_SECRET is not set; using a random secret for this process. "
            "Logins stop working when the server restarts. Set JWT_SECRET to keep them."
        )
    return _fallback_secret


def check_security_config() -> None:
    """Called once at startup so configuration problems show up in the log."""
    secret = get_jwt_secret()
    if os.environ.get("JWT_SECRET") and len(secret) < MIN_SECRET_LENGTH:
        logger.warning(
            "JWT_SECRET is shorter than %d characters; use a longer random value",
            MIN_SECRET_LENGTH,
        )


def get_access_token_minutes() -> float:
    raw = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")
    if raw is None:
        return DEFAULT_ACCESS_TOKEN_MINUTES
    try:
        minutes = float(raw)
    except ValueError:
        return DEFAULT_ACCESS_TOKEN_MINUTES
    return minutes if minutes > 0 else DEFAULT_ACCESS_TOKEN_MINUTES


def create_access_token(user_id: str, expires_minutes: float | None = None) -> str:
    minutes = get_access_token_minutes() if expires_minutes is None else expires_minutes
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    """Return the user id (``sub``) from a valid token, else raise InvalidTokenError."""
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc

    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise InvalidTokenError("token has no subject")
    return subject