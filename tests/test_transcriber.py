"""Tests for the Transcriber module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.core.transcriber import Transcriber, TranscriptionResult


@pytest.fixture
def mock_settings() -> Settings:
    """Create settings for testing."""
    return Settings(whisper_model_size="tiny")


@pytest.fixture
def mock_whisper_result() -> dict:
    """Create a realistic mock Whisper output."""
    return {
        "text": " Hello world this is a test transcription",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": " Hello world this is",
                "no_speech_prob": 0.05,
                "words": [
                    {"word": " Hello", "start": 0.0, "end": 0.4},
                    {"word": " world", "start": 0.4, "end": 0.8},
                    {"word": " this", "start": 0.8, "end": 1.2},
                    {"word": " is", "start": 1.2, "end": 1.5},
                ],
            },
            {
                "id": 1,
                "start": 2.5,
                "end": 5.0,
                "text": " a test transcription",
                "no_speech_prob": 0.03,
                "words": [
                    {"word": " a", "start": 2.5, "end": 2.7},
                    {"word": " test", "start": 2.7, "end": 3.2},
                    {"word": " transcription", "start": 3.2, "end": 4.8},
                ],
            },
        ],
    }


class TestTranscriber:
    """Tests for the Whisper transcriber wrapper."""

    def test_model_name(self, mock_settings: Settings) -> None:
        """Model name should reflect configured size."""
        transcriber = Transcriber(mock_settings)
        assert transcriber.model_name == "whisper-tiny"

    def test_confidence_computation(
        self, mock_settings: Settings, mock_whisper_result: dict
    ) -> None:
        """Confidence should be computed from no_speech_prob."""
        transcriber = Transcriber(mock_settings)
        confidence = transcriber._compute_confidence(mock_whisper_result)
        # avg no_speech_prob = (0.05 + 0.03) / 2 = 0.04
        # confidence = 1 - 0.04 = 0.96
        assert confidence == pytest.approx(0.96, abs=0.01)

    def test_confidence_empty_segments(self, mock_settings: Settings) -> None:
        """Confidence should be 0.0 for empty segments."""
        transcriber = Transcriber(mock_settings)
        confidence = transcriber._compute_confidence({"segments": []})
        assert confidence == 0.0

    def test_word_timestamp_extraction(
        self, mock_settings: Settings, mock_whisper_result: dict
    ) -> None:
        """Word timestamps should be extracted from all segments."""
        transcriber = Transcriber(mock_settings)
        timestamps = transcriber._extract_word_timestamps(mock_whisper_result)
        assert len(timestamps) == 7
        assert timestamps[0]["word"] == "Hello"
        assert timestamps[0]["start"] == 0.0
        assert timestamps[-1]["word"] == "transcription"

    @patch("app.core.transcriber._load_whisper_model")
    @pytest.mark.asyncio
    async def test_transcribe_async(
        self,
        mock_load: MagicMock,
        mock_settings: Settings,
        mock_whisper_result: dict,
    ) -> None:
        """Async transcribe should return a valid TranscriptionResult."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_whisper_result
        mock_load.return_value = mock_model

        transcriber = Transcriber(mock_settings)
        result = await transcriber.transcribe(Path("test.wav"))

        assert isinstance(result, TranscriptionResult)
        assert "Hello world" in result.text
        assert result.language == "en"
        assert result.confidence > 0.9
        assert len(result.word_timestamps) == 7
