"""Authentication dependencies for the API.

Provides optional API key verification. When API_KEY is set in the
environment, all protected endpoints require a matching key in the
X-API-Key header. When unset, authentication is disabled.
"""

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> None:
    """Validate the API key if authentication is enabled.

    Skips validation when API_KEY is not configured (empty string).

    Raises:
        HTTPException: 401 if key is missing, 403 if key is invalid.
    """
    settings = get_settings()

    # Auth disabled when no key is configured
    if not settings.api_key:
        return

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")

    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key.")
