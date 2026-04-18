"""Pydantic v2 models for request validation and response serialization.

These schemas enforce strict typing at the API boundary and serve as
the contract between the pipeline and its consumers.
"""

from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """A single word with its time boundaries in the audio."""

    word: str = Field(description="Transcribed word")
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")

    model_config = {
        "json_schema_extra": {
            "examples": [{"word": "hello", "start": 0.0, "end": 0.4}]
        }
    }


class SpeakerSegment(BaseModel):
    """A segment of speech attributed to a speaker."""

    speaker: str = Field(description="Speaker label (e.g., Speaker 1)")
    text: str = Field(description="Text spoken in this segment")
    start: float = Field(ge=0.0, description="Segment start time in seconds")
    end: float = Field(ge=0.0, description="Segment end time in seconds")


class TranscriptionMetadata(BaseModel):
    """Metadata about the transcription job."""

    model_config = {"protected_namespaces": ()}

    filename: str = Field(description="Original uploaded filename")
    duration_seconds: float = Field(description="Audio duration in seconds")
    model_used: str = Field(description="Whisper model identifier used")


class TranscriptionResponse(BaseModel):
    """Full transcription result returned to the client."""

    transcript: str = Field(description="Cleaned full transcript text")
    confidence: float = Field(
        description="Overall confidence score (0.0 - 1.0)",
        ge=0.0,
        le=1.0,
    )
    language: str = Field(description="Detected language code (e.g., 'en')")
    processing_time_seconds: float = Field(
        description="Total processing time in seconds"
    )
    timestamps: list[WordTimestamp] = Field(
        default_factory=list,
        description="Word-level timestamps",
    )
    segments: list[SpeakerSegment] = Field(
        default_factory=list,
        description="Speaker-segmented transcript sections",
    )
    metadata: TranscriptionMetadata = Field(
        description="Job metadata"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str = Field(description="Error type")
    detail: str = Field(description="Human-readable error description")
    status_code: int = Field(description="HTTP status code")
