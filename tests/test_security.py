import time

import jwt

from task_processing_system import security
from task_processing_system.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

SECRET = "a-test-secret-that-is-at-least-32-characters-long"


def expect_invalid(token):
    try:
        decode_access_token(token)
    except InvalidTokenError:
        return
    raise AssertionError("expected InvalidTokenError")


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def test_hash_and_verify_roundtrip():
    stored = hash_password("correct horse battery")
    assert verify_password("correct horse battery", stored) is True
    assert verify_password("wrong password", stored) is False
    assert verify_password("", stored) is False


def test_hash_is_salted_and_never_contains_the_password():
    a = hash_password("same-password")
    b = hash_password("same-password")
    assert a != b  # different random salt each time
    assert a.startswith("scrypt$")
    assert "same-password" not in a
    assert verify_password("same-password", a) and verify_password("same-password", b)


def test_unicode_passwords_work():
    stored = hash_password("pässwörd-密码-🔑")
    assert verify_password("pässwörd-密码-🔑", stored) is True
    assert verify_password("passwordd", stored) is False


def test_verify_never_raises_on_malformed_hashes():
    for bad in ["", "plaintext", "scrypt$1$2", "bcrypt$a$b$c$d$e", "scrypt$x$y$z$!!!$???", "a$b$c$d$e$f"]:
        assert verify_password("anything", bad) is False


def test_verify_against_dummy_does_not_raise():
    security.verify_against_dummy("whatever")
    security.verify_against_dummy("")


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_token_roundtrip_returns_subject(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = create_access_token("user-123")
    assert decode_access_token(token) == "user-123"


def test_token_carries_only_expected_claims(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    payload = jwt.decode(create_access_token("u1"), SECRET, algorithms=["HS256"])
    assert set(payload) == {"sub", "iat", "exp"}


def test_default_lifetime_is_30_minutes(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.delenv("ACCESS_TOKEN_EXPIRE_MINUTES", raising=False)
    payload = jwt.decode(create_access_token("u1"), SECRET, algorithms=["HS256"])
    assert payload["exp"] - payload["iat"] == 30 * 60


def test_lifetime_can_be_configured(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "5")
    payload = jwt.decode(create_access_token("u1"), SECRET, algorithms=["HS256"])
    assert payload["exp"] - payload["iat"] == 5 * 60


def test_invalid_lifetime_setting_falls_back_to_default(monkeypatch):
    for bad in ["abc", "0", "-5", ""]:
        monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", bad)
        assert security.get_access_token_minutes() == security.DEFAULT_ACCESS_TOKEN_MINUTES


def test_expired_token_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    expect_invalid(create_access_token("u1", expires_minutes=-1))


def test_token_expires_after_its_lifetime(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = create_access_token("u1", expires_minutes=0.02)  # about 1.2 seconds
    assert decode_access_token(token) == "u1"
    time.sleep(2.2)
    expect_invalid(token)


def test_token_signed_with_another_secret_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = create_access_token("u1")
    monkeypatch.setenv("JWT_SECRET", "a-completely-different-secret-value-1234567890")
    expect_invalid(token)


def test_tampered_token_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    header, payload, signature = create_access_token("u1").split(".")
    # change the signature
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    expect_invalid(f"{header}.{payload}.{flipped}")
    # swap in a payload for a different user, keeping the old signature
    other = create_access_token("someone-else").split(".")[1]
    expect_invalid(f"{header}.{other}.{signature}")


def test_unsigned_alg_none_token_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    forged = jwt.encode({"sub": "admin", "exp": int(time.time()) + 600}, key=None, algorithm="none")
    expect_invalid(forged)


def test_token_with_other_algorithm_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    forged = jwt.encode({"sub": "u1", "exp": int(time.time()) + 600}, SECRET, algorithm="HS512")
    expect_invalid(forged)


def test_token_without_expiry_or_subject_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    expect_invalid(jwt.encode({"sub": "u1"}, SECRET, algorithm="HS256"))
    expect_invalid(jwt.encode({"exp": int(time.time()) + 600}, SECRET, algorithm="HS256"))


def test_garbage_tokens_are_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    for garbage in ["", "not-a-token", "a.b.c", "Bearer abc", "....", " "]:
        expect_invalid(garbage)


def test_missing_secret_falls_back_to_a_stable_random_one(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setattr(security, "_fallback_secret", None)
    first = security.get_jwt_secret()
    assert len(first) >= security.MIN_SECRET_LENGTH
    assert security.get_jwt_secret() == first  # same within the process
    assert decode_access_token(create_access_token("u1")) == "u1"