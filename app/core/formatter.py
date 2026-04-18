"""Output formatter that assembles the final structured JSON response.

Takes raw results from the transcriber and post-processor and builds
a validated Pydantic response model.
"""

from app.models.schemas import (
    SpeakerSegment,
    TranscriptionMetadata,
    TranscriptionResponse,
    WordTimestamp,
)


class OutputFormatter:
    """Builds the final structured transcription response."""

    def format(
        self,
        transcript: str,
        confidence: float,
        language: str,
        processing_time: float,
        word_timestamps: list[dict],
        segments: list[dict],
        filename: str,
        duration: float,
        model_used: str,
    ) -> TranscriptionResponse:
        """Assemble all pipeline outputs into a validated response.

        Args:
            transcript: Cleaned transcript text.
            confidence: Overall confidence score.
            language: Detected language code.
            processing_time: Total processing time in seconds.
            word_timestamps: Word-level timing data.
            segments: Speaker segments.
            filename: Original uploaded filename.
            duration: Audio duration in seconds.
            model_used: Whisper model identifier.

        Returns:
            Validated TranscriptionResponse model.
        """
        return TranscriptionResponse(
            transcript=transcript,
            confidence=round(confidence, 4),
            language=language,
            processing_time_seconds=round(processing_time, 3),
            timestamps=[
                WordTimestamp(
                    word=w["word"],
                    start=w["start"],
                    end=w["end"],
                )
                for w in word_timestamps
            ],
            segments=[
                SpeakerSegment(
                    speaker=s["speaker"],
                    text=s["text"],
                    start=s["start"],
                    end=s["end"],
                )
                for s in segments
            ],
            metadata=TranscriptionMetadata(
                filename=filename,
                duration_seconds=round(duration, 2),
                model_used=model_used,
            ),
        )
