"""Tests for the API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.schemas import TranscriptionMetadata, TranscriptionResponse
from tests.conftest import create_test_wav as _create_test_wav


@pytest.fixture
def mock_transcription_response() -> TranscriptionResponse:
    """Create a mock transcription response for service-level mocking."""
    return TranscriptionResponse(
        transcript="Hello world this is a test.",
        confidence=0.95,
        language="en",
        processing_time_seconds=0.5,
        timestamps=[],
        segments=[],
        metadata=TranscriptionMetadata(
            filename="test.wav",
            duration_seconds=2.0,
            model_used="whisper-base",
        ),
    )


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    """Health endpoint should return status healthy."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_invalid_file_type() -> None:
    """Uploading a non-audio file should return 415."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/transcribe",
            files={"file": ("test.txt", b"not audio data", "text/plain")},
        )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_invalid_extension() -> None:
    """Uploading a file with wrong extension should return 415."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/transcribe",
            files={"file": ("test.pdf", b"fake data", "application/pdf")},
        )
    assert response.status_code == 415


@pytest.mark.asyncio
@patch("app.services.transcription.TranscriptionService.transcribe_file")
async def test_valid_wav_transcription(
    mock_service: AsyncMock,
    mock_transcription_response: TranscriptionResponse,
) -> None:
    """Valid WAV upload should return a structured transcription response."""
    mock_service.return_value = (mock_transcription_response, "abc123def456")

    wav_data = _create_test_wav(duration_ms=2000)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "transcript" in data
    assert "confidence" in data
    assert "language" in data
    assert "processing_time_seconds" in data
    assert "timestamps" in data
    assert "segments" in data
    assert "metadata" in data

    # Validate metadata
    assert data["metadata"]["filename"] == "test.wav"
    assert data["metadata"]["model_used"] == "whisper-base"
    assert data["metadata"]["duration_seconds"] > 0

    # Validate types
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["processing_time_seconds"], float)
    assert isinstance(data["timestamps"], list)
    assert isinstance(data["segments"], list)


@pytest.mark.asyncio
@patch("app.services.transcription.TranscriptionService.transcribe_file")
async def test_response_json_schema(
    mock_service: AsyncMock,
    mock_transcription_response: TranscriptionResponse,
) -> None:
    """Response should match the exact JSON schema contract."""
    mock_service.return_value = (mock_transcription_response, "abc123def456")

    wav_data = _create_test_wav()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/transcribe",
            files={"file": ("audio.wav", wav_data, "audio/wav")},
        )

    data = response.json()
    required_keys = {
        "transcript", "confidence", "language",
        "processing_time_seconds", "timestamps", "segments", "metadata",
    }
    assert required_keys.issubset(set(data.keys()))

    metadata_keys = {"filename", "duration_seconds", "model_used"}
    assert metadata_keys.issubset(set(data["metadata"].keys()))

    # If timestamps present, validate structure
    if data["timestamps"]:
        ts = data["timestamps"][0]
        assert "word" in ts
        assert "start" in ts
        assert "end" in ts
