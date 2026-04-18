"""Post-processing module for transcript cleanup.

Applies configurable text transformations:
- Filler word removal
- Rule-based punctuation correction
- Mock speaker segmentation

Each step is independently toggleable via configuration.
"""

import re

from loguru import logger

from app.config import Settings

# Filler words/phrases to remove (case-insensitive, whole-word matching)
FILLER_WORDS: list[str] = [
    "um", "uh", "hmm", "hm", "ah",
    "you know", "basically", "literally",
    "i mean", "sort of", "kind of",
    "like",  # Only removed when standalone filler, not in phrases like "I like"
]


class PostProcessor:
    """Applies text cleanup and segmentation to raw transcription output."""

    def __init__(self, settings: Settings) -> None:
        """Initialize with processing configuration.

        Args:
            settings: Application configuration with toggle flags.
        """
        self._enable_filler_removal = settings.enable_filler_removal
        self._enable_punctuation = settings.enable_punctuation_correction
        self._enable_segmentation = settings.enable_speaker_segmentation
        self._segment_interval = settings.speaker_segment_interval

        # Pre-compile filler patterns for performance
        # Sort by length descending so multi-word fillers match first
        sorted_fillers = sorted(FILLER_WORDS, key=len, reverse=True)
        patterns = [rf"\b{re.escape(f)}\b" for f in sorted_fillers]
        self._filler_pattern = re.compile(
            "|".join(patterns), re.IGNORECASE
        )

    def process(
        self,
        text: str,
        word_timestamps: list[dict],
        duration: float,
        language: str = "en",
    ) -> tuple[str, list[dict]]:
        """Apply all enabled post-processing steps.

        Args:
            text: Raw transcript text.
            word_timestamps: Word-level timestamp dicts.
            duration: Total audio duration in seconds.
            language: Detected language code for language-aware processing.

        Returns:
            Tuple of (cleaned text, speaker segments list).
        """
        if self._enable_filler_removal:
            text = self._remove_fillers(text)
            logger.debug("Filler words removed")

        # Punctuation rules are English-only; skip for other languages
        if self._enable_punctuation and language == "en":
            text = self._correct_punctuation(text)
            logger.debug("Punctuation corrected")
        elif self._enable_punctuation and language != "en":
            logger.debug("Skipping punctuation correction for language: {}", language)

        segments: list[dict] = []
        if self._enable_segmentation:
            segments = self._segment_speakers(text, word_timestamps, duration)
            logger.debug("Speaker segmentation applied: {} segments", len(segments))

        return text, segments

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words from the transcript.

        Args:
            text: Input transcript text.

        Returns:
            Text with fillers removed and whitespace normalized.
        """
        cleaned = self._filler_pattern.sub("", text)
        # Collapse multiple spaces into one
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        # Clean up orphaned punctuation (e.g., ", ," → ",")
        cleaned = re.sub(r"(\s*,\s*)+", ", ", cleaned)
        return cleaned

    def _correct_punctuation(self, text: str) -> str:
        """Apply rule-based punctuation corrections.

        Handles:
        - Capitalizing the first letter after sentence-ending punctuation
        - Ensuring sentences end with proper punctuation
        - Fixing spacing around punctuation marks

        Args:
            text: Input text.

        Returns:
            Text with corrected punctuation.
        """
        if not text:
            return text

        # Fix spacing before punctuation marks
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # Ensure space after punctuation (if followed by a letter)
        text = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", text)

        # Capitalize first letter of text
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Capitalize first letter after sentence-ending punctuation
        text = re.sub(
            r"([.!?])\s+([a-z])",
            lambda m: f"{m.group(1)} {m.group(2).upper()}",
            text,
        )

        # Ensure text ends with a period if it doesn't end with punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text

    def _segment_speakers(
        self,
        text: str,
        word_timestamps: list[dict],
        duration: float,
    ) -> list[dict]:
        """Create mock speaker segments by alternating speakers.

        Splits the transcript into segments based on a fixed time interval,
        alternating between Speaker 1 and Speaker 2. This is a mock
        implementation — production would use a diarization model (e.g., Pyannote).

        Args:
            text: Full transcript text.
            word_timestamps: Word-level timestamps for segmentation.
            duration: Total audio duration in seconds.

        Returns:
            List of speaker segment dicts.
        """
        if not word_timestamps:
            # Fall back to simple time-based splitting if no timestamps
            return self._segment_by_time_only(text, duration)

        segments: list[dict] = []
        interval = self._segment_interval
        current_segment_words: list[str] = []
        segment_start = word_timestamps[0]["start"] if word_timestamps else 0.0
        speaker_index = 0

        for word_info in word_timestamps:
            word_start = word_info["start"]
            # Check if this word crosses a segment boundary
            expected_segment = int(word_start // interval)

            if expected_segment > speaker_index and current_segment_words:
                # Flush current segment
                segments.append({
                    "speaker": f"Speaker {(speaker_index % 2) + 1}",
                    "text": " ".join(current_segment_words),
                    "start": round(segment_start, 2),
                    "end": round(word_start, 2),
                })
                current_segment_words = []
                segment_start = word_start
                speaker_index = expected_segment

            current_segment_words.append(word_info["word"])

        # Flush remaining words
        if current_segment_words:
            end_time = word_timestamps[-1]["end"] if word_timestamps else duration
            segments.append({
                "speaker": f"Speaker {(speaker_index % 2) + 1}",
                "text": " ".join(current_segment_words),
                "start": round(segment_start, 2),
                "end": round(end_time, 2),
            })

        return segments

    def _segment_by_time_only(self, text: str, duration: float) -> list[dict]:
        """Fallback segmentation when no word timestamps are available.

        Args:
            text: Full transcript text.
            duration: Audio duration in seconds.

        Returns:
            Simple time-based speaker segments.
        """
        segments: list[dict] = []
        interval = self._segment_interval
        words = text.split()
        num_segments = max(1, int(duration / interval) + (1 if duration % interval else 0))
        words_per_segment = max(1, len(words) // num_segments)

        for i in range(num_segments):
            start_idx = i * words_per_segment
            end_idx = start_idx + words_per_segment if i < num_segments - 1 else len(words)
            segment_words = words[start_idx:end_idx]

            if segment_words:
                segments.append({
                    "speaker": f"Speaker {(i % 2) + 1}",
                    "text": " ".join(segment_words),
                    "start": round(i * interval, 2),
                    "end": round(min((i + 1) * interval, duration), 2),
                })

        return segments
