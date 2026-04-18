"""SQLAlchemy ORM model for stored transcripts."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Transcript(Base):
    """Persisted transcription result."""

    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(
        String(12), primary_key=True, default=lambda: uuid.uuid4().hex[:12]
    )
    filename: Mapped[str] = mapped_column(String(255))
    transcript: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    language: Mapped[str] = mapped_column(String(10))
    processing_time_seconds: Mapped[float] = mapped_column(Float)
    duration_seconds: Mapped[float] = mapped_column(Float)
    model_used: Mapped[str] = mapped_column(String(50))
    timestamps_json: Mapped[str] = mapped_column(Text, default="[]")
    segments_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
