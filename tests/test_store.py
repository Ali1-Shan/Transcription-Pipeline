"""Tests for database-backed transcript storage and retrieval."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.transcriber import TranscriptionResult
from app.database import Base
from app.main import app
from app.models.schemas import (
    TranscriptionMetadata,
    TranscriptionResponse,
)
from app.models.transcript import Transcript
from app.services.transcription import TranscriptionService
from tests.conftest import create_test_wav as _create_test_wav


@pytest.fixture
async def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


class TestTranscriptDBStorage:
    """Tests for the database-backed transcript persistence."""

    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, db_session: AsyncSession) -> None:
        """Service should save to DB and retrieve by ID."""
        response = TranscriptionResponse(
            transcript="Hello world.",
            confidence=0.95,
            language="en",
            processing_time_seconds=1.5,
            timestamps=[],
            segments=[],
            metadata=TranscriptionMetadata(
                filename="test.wav",
                duration_seconds=2.0,
                model_used="whisper-tiny",
            ),
        )

        # Use the service's internal save method
        service = TranscriptionService.__new__(TranscriptionService)
        tid = await service._save_to_db(db_session, response)

        assert len(tid) == 12

        # Retrieve via service
        result = await service.get_transcript(tid, db_session)
        assert result is not None
        assert result["transcript"] == "Hello world."
        assert result["id"] == tid
        assert result["metadata"]["filename"] == "test.wav"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, db_session: AsyncSession) -> None:
        service = TranscriptionService.__new__(TranscriptionService)
        result = await service.get_transcript("nonexistent", db_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_timestamps_and_segments_persisted(self, db_session: AsyncSession) -> None:
        """Timestamps and segments should survive JSON round-trip."""
        from app.models.schemas import WordTimestamp, SpeakerSegment

        response = TranscriptionResponse(
            transcript="Hello world.",
            confidence=0.95,
            language="en",
            processing_time_seconds=1.0,
            timestamps=[WordTimestamp(word="Hello", start=0.0, end=0.4)],
            segments=[SpeakerSegment(speaker="Speaker 1", text="Hello world.", start=0.0, end=1.0)],
            metadata=TranscriptionMetadata(
                filename="test.wav",
                duration_seconds=1.0,
                model_used="whisper-tiny",
            ),
        )

        service = TranscriptionService.__new__(TranscriptionService)
        tid = await service._save_to_db(db_session, response)
        result = await service.get_transcript(tid, db_session)

        assert len(result["timestamps"]) == 1
        assert result["timestamps"][0]["word"] == "Hello"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["speaker"] == "Speaker 1"


@pytest.mark.asyncio
@patch("app.services.transcription.TranscriptionService.get_transcript")
async def test_get_transcript_not_found(mock_get: AsyncMock) -> None:
    """GET /transcript/{id} should return 404 for unknown IDs."""
    mock_get.return_value = None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/transcript/doesnotexist")
    assert response.status_code == 404


@pytest.mark.asyncio
@patch("app.services.transcription.TranscriptionService.transcribe_file")
async def test_transcribe_returns_transcript_id_header(
    mock_service: AsyncMock,
) -> None:
    """POST /transcribe should return X-Transcript-ID header."""
    mock_response = TranscriptionResponse(
        transcript="Hello.",
        confidence=0.95,
        language="en",
        processing_time_seconds=0.5,
        timestamps=[],
        segments=[],
        metadata=TranscriptionMetadata(
            filename="test.wav",
            duration_seconds=1.0,
            model_used="whisper-base",
        ),
    )
    mock_service.return_value = (mock_response, "abc123def456")

    wav_data = _create_test_wav()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/transcribe",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )

    assert response.status_code == 200
    assert "x-transcript-id" in response.headers
    assert response.headers["x-transcript-id"] == "abc123def456"
