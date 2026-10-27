"""Registration, login, and the dependency that protects the other routes.

Flow:
  1. POST /auth/register   create an account
  2. POST /auth/login      exchange username + password for a JWT
  3. send ``Authorization: Bearer <token>`` on every other request

Login takes an OAuth2 *form* (username/password fields) rather than JSON so
that the green "Authorize" button in /docs works out of the box.

Password hashing is deliberately slow (tens of milliseconds), so it runs in a
thread and never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from . import repository as repo
from . import security
from .schemas import MAX_PASSWORD_LENGTH, TokenResponse, UserCreate, UserResponse

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

router = APIRouter(prefix="/auth", tags=["auth"])


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> dict[str, Any]:
    """Resolve the Bearer token to a user, or raise 401.

    Also rejects a perfectly valid token whose user no longer exists.
    """
    try:
        user_id = security.decode_access_token(token)
    except security.InvalidTokenError:
        raise _unauthorized("Invalid or expired token") from None

    user = await repo.get_user_by_id(user_id)
    if user is None:
        raise _unauthorized("Invalid or expired token")
    return user


CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(body: UserCreate) -> Any:
    """Create an account. Usernames are unique, ignoring upper/lower case."""
    password_hash = await asyncio.to_thread(security.hash_password, body.password)
    try:
        user = await repo.create_user(body.username, password_hash)
    except repo.UsernameTakenError as exc:
        raise HTTPException(status_code=409, detail="Username is already taken") from exc
    logger.info("User registered: id=%s username=%s", user["id"], user["username"])
    return user


@router.post("/login", response_model=TokenResponse)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Any:
    """Log in with username and password and receive a Bearer token.

    The same error is returned for an unknown user and a wrong password, and
    both take the same time, so the response never reveals which usernames exist.
    """
    user = await repo.get_user_by_username(form.username)

    if user is None or len(form.password) > MAX_PASSWORD_LENGTH:
        await asyncio.to_thread(security.verify_against_dummy, form.password)
        valid = False
    else:
        valid = await asyncio.to_thread(
            security.verify_password, form.password, user["password_hash"]
        )

    if not valid or user is None:
        logger.warning("Failed login attempt for username=%r", form.username)
        raise _unauthorized("Incorrect username or password")

    logger.info("User logged in: id=%s username=%s", user["id"], user["username"])
    return {
        "access_token": security.create_access_token(user["id"]),
        "token_type": "bearer",
        "expires_in": int(security.get_access_token_minutes() * 60),
    }


@router.get("/me", response_model=UserResponse)
async def read_current_user(user: CurrentUser) -> Any:
    """Return the logged-in user. Handy for checking that a token works."""
    return user