"""Transcription service — orchestrates the full pipeline.

Encapsulates business logic: input handling, transcription,
post-processing, formatting, and persistence. Routes delegate
to this service instead of wiring pipeline components directly.
"""

import json
import time

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.formatter import OutputFormatter
from app.core.input_handler import InputHandler
from app.core.postprocessor import PostProcessor
from app.core.transcriber import Transcriber
from app.models.schemas import TranscriptionResponse
from app.models.transcript import Transcript


class TranscriptionService:
    """Coordinates transcription pipeline and persistence."""

    def __init__(
        self,
        input_handler: InputHandler,
        transcriber: Transcriber,
        post_processor: PostProcessor,
        formatter: OutputFormatter,
    ) -> None:
        self._input_handler = input_handler
        self._transcriber = transcriber
        self._post_processor = post_processor
        self._formatter = formatter

    async def transcribe_file(
        self,
        file,
        db: AsyncSession,
        request_id: str = "unknown",
    ) -> tuple[TranscriptionResponse, str]:
        """Run the full transcription pipeline and persist the result.

        Args:
            file: FastAPI UploadFile.
            db: Async database session.
            request_id: Request tracing ID for logs.

        Returns:
            Tuple of (TranscriptionResponse, transcript_id).
        """
        start_time = time.monotonic()
        filename = file.filename or "unknown"

        logger.info(
            "Pipeline started | file={} | request_id={}",
            filename,
            request_id,
        )

        # Stage 1: Validate and convert audio
        wav_path, duration = await self._input_handler.process_upload(file)

        try:
            # Stage 2: Whisper transcription (with retry + concurrency limit)
            result = await self._transcriber.transcribe(wav_path)

            # Stage 3: Post-processing
            cleaned_text, segments = self._post_processor.process(
                text=result.text,
                word_timestamps=result.word_timestamps,
                duration=duration,
                language=result.language,
            )

            # Stage 4: Format response
            processing_time = time.monotonic() - start_time
            response = self._formatter.format(
                transcript=cleaned_text,
                confidence=result.confidence,
                language=result.language,
                processing_time=processing_time,
                word_timestamps=result.word_timestamps,
                segments=segments,
                filename=filename,
                duration=duration,
                model_used=self._transcriber.model_name,
            )

            # Stage 5: Persist to database
            transcript_id = await self._save_to_db(db, response)

            logger.info(
                "Pipeline complete | file={} | id={} | duration={:.2f}s | processing={:.2f}s | request_id={}",
                filename,
                transcript_id,
                duration,
                processing_time,
                request_id,
            )

            return response, transcript_id

        finally:
            # Always clean up temp file
            wav_path.unlink(missing_ok=True)

    async def get_transcript(
        self, transcript_id: str, db: AsyncSession
    ) -> dict | None:
        """Retrieve a transcript by ID from the database.

        Args:
            transcript_id: Unique transcript identifier.
            db: Async database session.

        Returns:
            Transcript data dict, or None if not found.
        """
        stmt = select(Transcript).where(Transcript.id == transcript_id)
        result = await db.execute(stmt)
        row = result.scalar_one_or_none()

        if row is None:
            return None

        return {
            "id": row.id,
            "transcript": row.transcript,
            "confidence": row.confidence,
            "language": row.language,
            "processing_time_seconds": row.processing_time_seconds,
            "timestamps": json.loads(row.timestamps_json),
            "segments": json.loads(row.segments_json),
            "metadata": {
                "filename": row.filename,
                "duration_seconds": row.duration_seconds,
                "model_used": row.model_used,
            },
            "created_at": row.created_at.isoformat(),
        }

    async def _save_to_db(
        self, db: AsyncSession, response: TranscriptionResponse
    ) -> str:
        """Persist a transcription response to the database.

        Args:
            db: Async database session.
            response: Formatted pipeline output.

        Returns:
            The generated transcript ID.
        """
        row = Transcript(
            filename=response.metadata.filename,
            transcript=response.transcript,
            confidence=response.confidence,
            language=response.language,
            processing_time_seconds=response.processing_time_seconds,
            duration_seconds=response.metadata.duration_seconds,
            model_used=response.metadata.model_used,
            timestamps_json=json.dumps(
                [t.model_dump() for t in response.timestamps]
            ),
            segments_json=json.dumps(
                [s.model_dump() for s in response.segments]
            ),
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)

        logger.info("Transcript persisted to DB | id={}", row.id)
        return row.id
