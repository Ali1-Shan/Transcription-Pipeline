"""Tests for the PostProcessor module."""

import pytest

from app.config import Settings
from app.core.postprocessor import PostProcessor


@pytest.fixture
def processor() -> PostProcessor:
    """Create a PostProcessor with all features enabled."""
    settings = Settings(
        enable_filler_removal=True,
        enable_punctuation_correction=True,
        enable_speaker_segmentation=True,
        speaker_segment_interval=30,
    )
    return PostProcessor(settings)


@pytest.fixture
def processor_no_fillers() -> PostProcessor:
    """Create a PostProcessor with filler removal disabled."""
    settings = Settings(
        enable_filler_removal=False,
        enable_punctuation_correction=False,
        enable_speaker_segmentation=False,
    )
    return PostProcessor(settings)


class TestFillerRemoval:
    """Test filler word removal functionality."""

    def test_removes_single_fillers(self, processor: PostProcessor) -> None:
        """Filler words like 'um', 'uh' should be stripped."""
        text = "So um I was thinking uh about this"
        cleaned, _ = processor.process(text, [], 10.0)
        assert "um" not in cleaned.lower().split()
        assert "uh" not in cleaned.lower().split()

    def test_removes_multi_word_fillers(self, processor: PostProcessor) -> None:
        """Multi-word fillers like 'you know' should be removed."""
        text = "I was you know thinking about basically everything"
        cleaned, _ = processor.process(text, [], 10.0)
        assert "you know" not in cleaned.lower()
        assert "basically" not in cleaned.lower()

    def test_removes_literally(self, processor: PostProcessor) -> None:
        """The filler 'literally' should be removed."""
        text = "It was literally the best day"
        cleaned, _ = processor.process(text, [], 10.0)
        assert "literally" not in cleaned.lower()

    def test_preserves_meaningful_text(self, processor: PostProcessor) -> None:
        """Non-filler content should remain intact."""
        text = "The weather today is wonderful"
        cleaned, _ = processor.process(text, [], 10.0)
        assert "weather" in cleaned.lower()
        assert "wonderful" in cleaned.lower()

    def test_handles_empty_text(self, processor: PostProcessor) -> None:
        """Empty input should return empty output."""
        cleaned, _ = processor.process("", [], 0.0)
        assert cleaned == ""

    def test_no_removal_when_disabled(self, processor_no_fillers: PostProcessor) -> None:
        """Fillers should remain when removal is disabled."""
        text = "um uh you know"
        cleaned, _ = processor_no_fillers.process(text, [], 10.0)
        assert "um" in cleaned

    def test_multiple_consecutive_fillers(self, processor: PostProcessor) -> None:
        """Multiple consecutive fillers should all be removed."""
        text = "um uh like basically the answer"
        cleaned, _ = processor.process(text, [], 10.0)
        assert "answer" in cleaned.lower()
        # Should not have double spaces
        assert "  " not in cleaned


class TestPunctuationCorrection:
    """Test punctuation correction functionality."""

    def test_capitalizes_first_letter(self, processor: PostProcessor) -> None:
        """First letter of transcript should be capitalized."""
        text = "hello world"
        cleaned, _ = processor.process(text, [], 10.0)
        assert cleaned[0] == "H"

    def test_adds_period_at_end(self, processor: PostProcessor) -> None:
        """Text should end with a period if no punctuation present."""
        text = "This is a sentence"
        cleaned, _ = processor.process(text, [], 10.0)
        assert cleaned.endswith(".")

    def test_preserves_existing_ending_punctuation(self, processor: PostProcessor) -> None:
        """Existing sentence-ending punctuation should be preserved."""
        text = "Is this a question?"
        cleaned, _ = processor.process(text, [], 10.0)
        assert cleaned.endswith("?")
        assert not cleaned.endswith("?.")

    def test_capitalizes_after_period(self, processor: PostProcessor) -> None:
        """First letter after a period should be capitalized."""
        text = "first sentence. second sentence"
        cleaned, _ = processor.process(text, [], 10.0)
        assert "Second" in cleaned


class TestSpeakerSegmentation:
    """Test mock speaker segmentation."""

    def test_creates_segments(self, processor: PostProcessor) -> None:
        """Should create speaker segments from timestamps."""
        timestamps = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
            {"word": "goodbye", "start": 31.0, "end": 31.5},
        ]
        _, segments = processor.process("hello world goodbye", timestamps, 32.0)
        assert len(segments) >= 2
        assert segments[0]["speaker"] == "Speaker 1"
        assert segments[1]["speaker"] == "Speaker 2"

    def test_fallback_without_timestamps(self, processor: PostProcessor) -> None:
        """Should fall back to time-based splitting without timestamps."""
        text = "This is a test sentence with enough words to split"
        _, segments = processor.process(text, [], 60.0)
        assert len(segments) >= 1
        assert all("speaker" in s for s in segments)

    def test_no_segments_when_disabled(self, processor_no_fillers: PostProcessor) -> None:
        """No segments should be returned when segmentation is disabled."""
        _, segments = processor_no_fillers.process("test", [], 10.0)
        assert segments == []
