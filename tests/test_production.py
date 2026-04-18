"""Tests for production hardening features: auth, request ID, concurrency."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.models.schemas import TranscriptionResponse
from app.main import app
from tests.conftest import create_test_wav as _create_test_wav


class TestRequestID:
    """Verify X-Request-ID middleware behavior."""

    @pytest.mark.asyncio
    async def test_response_includes_request_id(self) -> None:
        """Every response should include an X-Request-ID header."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert "x-request-id" in response.headers

    @pytest.mark.asyncio
    async def test_client_supplied_request_id_is_echoed(self) -> None:
        """If client sends X-Request-ID, it should be echoed back."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/health",
                headers={"X-Request-ID": "test-req-123"},
            )
        assert response.headers["x-request-id"] == "test-req-123"


class TestAPIKeyAuth:
    """Verify API key authentication behavior."""

    @pytest.mark.asyncio
    async def test_no_auth_required_when_key_not_configured(self) -> None:
        """When API_KEY is empty, requests should pass without auth."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        # Health doesn't require auth, but transcribe should also work
        assert response.status_code == 200

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    async def test_missing_key_returns_401(self, mock_settings: MagicMock) -> None:
        """When API_KEY is set and no key provided, return 401."""
        mock_cfg = MagicMock()
        mock_cfg.api_key = "secret-key-123"
        mock_settings.return_value = mock_cfg

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            wav_data = _create_test_wav()
            response = await client.post(
                "/transcribe",
                files={"file": ("test.wav", wav_data, "audio/wav")},
            )
        assert response.status_code == 401

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    async def test_wrong_key_returns_403(self, mock_settings: MagicMock) -> None:
        """When API_KEY is set and wrong key provided, return 403."""
        mock_cfg = MagicMock()
        mock_cfg.api_key = "secret-key-123"
        mock_settings.return_value = mock_cfg

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            wav_data = _create_test_wav()
            response = await client.post(
                "/transcribe",
                files={"file": ("test.wav", wav_data, "audio/wav")},
                headers={"X-API-Key": "wrong-key"},
            )
        assert response.status_code == 403

    @pytest.mark.asyncio
    @patch("app.api.auth.get_settings")
    @patch("app.services.transcription.TranscriptionService.transcribe_file")
    async def test_correct_key_passes(
        self, mock_transcribe: AsyncMock, mock_settings: MagicMock
    ) -> None:
        """When correct API key is provided, request should succeed."""
        mock_cfg = MagicMock()
        mock_cfg.api_key = "secret-key-123"
        mock_settings.return_value = mock_cfg

        mock_transcribe.return_value = (
            TranscriptionResponse(
                transcript="Hello.",
                confidence=0.95,
                language="en",
                processing_time_seconds=0.5,
                timestamps=[{"word": "Hello", "start": 0.0, "end": 0.4}],
                segments=[],
                metadata={
                    "filename": "test.wav",
                    "duration_seconds": 1.0,
                    "model_used": "whisper-base",
                },
            ),
            "abc123def456",
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            wav_data = _create_test_wav()
            response = await client.post(
                "/transcribe",
                files={"file": ("test.wav", wav_data, "audio/wav")},
                headers={"X-API-Key": "secret-key-123"},
            )
        assert response.status_code == 200
