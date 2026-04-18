"""Whisper STT wrapper with async support and retry logic.

Wraps OpenAI Whisper in an async-compatible interface using
asyncio.run_in_executor to prevent blocking the FastAPI event loop
during CPU-intensive transcription.
"""

import asyncio
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import whisper
from loguru import logger

from app.config import Settings

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0


@dataclass
class TranscriptionResult:
    """Raw output from the Whisper transcription engine."""

    text: str
    language: str
    confidence: float
    word_timestamps: list[dict] = field(default_factory=list)


class Transcriber:
    """Async-compatible wrapper around OpenAI Whisper."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the transcriber with the configured model size.

        Args:
            settings: Application configuration.
        """
        self._model_size = settings.whisper_model_size
        self._model: whisper.Whisper | None = None
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_transcriptions)
        logger.info("Transcriber initialized with model size: {}", self._model_size)

    def _load_model(self) -> whisper.Whisper:
        """Load the Whisper model (lazy, cached).

        Returns:
            Loaded Whisper model instance.
        """
        if self._model is None:
            logger.info("Loading Whisper model '{}'...", self._model_size)
            self._model = _load_whisper_model(self._model_size)
            logger.info("Whisper model '{}' loaded successfully", self._model_size)
        return self._model

    async def transcribe(self, audio_path: Path, max_retries: int = MAX_RETRIES) -> TranscriptionResult:
        """Transcribe an audio file asynchronously with retry on transient failures.

        Uses a semaphore to limit concurrent Whisper jobs and prevent OOM.
        Offloads the blocking Whisper inference to a thread pool executor
        so the FastAPI event loop remains responsive.

        Args:
            audio_path: Path to the WAV audio file.
            max_retries: Number of retry attempts on failure.

        Returns:
            TranscriptionResult with text, language, confidence, and timestamps.
        """
        async with self._semaphore:
            return await self._transcribe_with_retry(audio_path, max_retries)

    async def _transcribe_with_retry(self, audio_path: Path, max_retries: int) -> TranscriptionResult:
        """Internal retry loop for transcription attempts."""
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 2):
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._transcribe_sync,
                    audio_path,
                )
                return result
            except RuntimeError as exc:
                last_exc = exc
                if attempt <= max_retries:
                    logger.warning(
                        "Transcription attempt {}/{} failed: {} — retrying in {:.0f}s",
                        attempt,
                        max_retries + 1,
                        str(exc),
                        RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.error(
                        "Transcription failed after {} attempts: {}",
                        attempt,
                        str(exc),
                    )

        raise last_exc  # type: ignore[misc]

    def _transcribe_sync(self, audio_path: Path) -> TranscriptionResult:
        """Perform synchronous Whisper transcription.

        Args:
            audio_path: Path to the WAV audio file.

        Returns:
            Parsed transcription result.

        Raises:
            RuntimeError: If Whisper inference fails.
        """
        model = self._load_model()
        logger.info("Starting transcription for: {}", audio_path.name)

        try:
            result = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Whisper transcription failed: {}", str(exc))
            raise RuntimeError(f"Transcription failed: {str(exc)}") from exc

        # Extract word-level timestamps from segments
        word_timestamps = self._extract_word_timestamps(result)

        # Whisper doesn't provide a global confidence score;
        # we compute an average from segment-level "no_speech_prob" as a proxy.
        confidence = self._compute_confidence(result)

        if confidence < 0.5:
            logger.warning(
                "Low confidence ({:.2f}) for file: {} — result may be unreliable",
                confidence,
                audio_path.name,
            )

        language = result.get("language", "en")
        text = result.get("text", "").strip()

        logger.info(
            "Transcription complete: {} words, language={}, confidence={:.2f}",
            len(text.split()),
            language,
            confidence,
        )

        return TranscriptionResult(
            text=text,
            language=language,
            confidence=confidence,
            word_timestamps=word_timestamps,
        )

    def _extract_word_timestamps(self, result: dict) -> list[dict]:
        """Extract word-level timestamps from Whisper output.

        Args:
            result: Raw Whisper transcription result dict.

        Returns:
            List of dicts with 'word', 'start', 'end' keys.
        """
        timestamps: list[dict] = []
        segments = result.get("segments", [])

        for segment in segments:
            words = segment.get("words", [])
            for w in words:
                start = round(w.get("start", 0.0), 3)
                end = round(w.get("end", 0.0), 3)
                if end < start:
                    end = start
                timestamps.append({
                    "word": w.get("word", "").strip(),
                    "start": start,
                    "end": end,
                })

        return timestamps

    def _compute_confidence(self, result: dict) -> float:
        """Compute an approximate confidence score from segment probabilities.

        Uses (1 - avg_no_speech_prob) as a proxy for overall confidence,
        since Whisper does not provide a direct confidence metric.

        Args:
            result: Raw Whisper transcription result dict.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        segments = result.get("segments", [])
        if not segments:
            return 0.0

        no_speech_probs = [
            seg.get("no_speech_prob", 0.0) for seg in segments
        ]
        avg_no_speech = sum(no_speech_probs) / len(no_speech_probs)

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, 1.0 - avg_no_speech))
        return round(confidence, 4)

    @property
    def model_name(self) -> str:
        """Return the model identifier string."""
        return f"whisper-{self._model_size}"


@lru_cache(maxsize=1)
def _load_whisper_model(model_size: str) -> whisper.Whisper:
    """Load and cache a Whisper model by size.

    Args:
        model_size: Model variant to load.

    Returns:
        Loaded Whisper model.
    """
    return whisper.load_model(model_size)
