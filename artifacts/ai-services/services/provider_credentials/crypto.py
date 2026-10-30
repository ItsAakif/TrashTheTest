"""Encryption at rest for user provider keys.

Fernet (AES-128-CBC + HMAC) with a key derived from JWT_SECRET via
SHA-256 over a domain-separated info string. Decryption happens only
inside the request that needs the key; the raw key is never logged,
never returned, never placed in a URL, and never cached.
"""

from __future__ import annotations

import base64
import hashlib


def _fernet_key() -> bytes:
    try:
        from config.settings import get_auth_config

        secret = get_auth_config().secret
    except Exception as exc:
        raise RuntimeError("Provider credential crypto unavailable: auth secret not configured.") from exc
    digest = hashlib.sha256(f"trashthetest-provider-cred-v1:{secret}".encode()).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_key(raw_key: str) -> str:
    from cryptography.fernet import Fernet

    if not raw_key or not raw_key.strip():
        raise ValueError("API key must not be empty.")
    token = Fernet(_fernet_key()).encrypt(raw_key.strip().encode())
    return token.decode()


def decrypt_key(token: str) -> str:
    from cryptography.fernet import Fernet, InvalidToken

    try:
        return Fernet(_fernet_key()).decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("Stored credential cannot be decrypted.") from exc


def mask_hint(raw_key: str) -> str:
    """Masked display hint: bullet run + last 4 chars (never the key)."""
    cleaned = (raw_key or "").strip()
    tail = cleaned[-4:] if len(cleaned) >= 4 else "****"
    return f"....{tail}"


__all__ = ["encrypt_key", "decrypt_key", "mask_hint"]
