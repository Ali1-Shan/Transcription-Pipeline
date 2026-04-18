"""Input handler for audio file validation and format conversion.

Validates uploaded files against size/type constraints and normalizes
audio to WAV format for consistent downstream processing.
"""

import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile
from loguru import logger
from pydub import AudioSegment

from app.config import Settings

ALLOWED_CONTENT_TYPES: set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
}

ALLOWED_EXTENSIONS: set[str] = {".wav", ".mp3"}


class InputHandler:
    """Handles audio file validation, size checks, and format conversion."""

    def __init__(self, settings: Settings) -> None:
        """Initialize with application settings.

        Args:
            settings: Application configuration.
        """
        self._max_file_size = settings.max_file_size_bytes

    async def process_upload(self, file: UploadFile) -> tuple[Path, float]:
        """Validate and convert an uploaded audio file.

        Args:
            file: The uploaded file from the request.

        Returns:
            Tuple of (path to WAV file, duration in seconds).

        Raises:
            HTTPException: On invalid file type, size, or processing error.
        """
        self._validate_file_type(file)
        raw_bytes = await self._read_and_validate_size(file)
        wav_path, duration = self._convert_to_wav(raw_bytes, file.filename or "audio")
        logger.info(
            "Audio processed",
            filename=file.filename,
            duration_seconds=round(duration, 2),
        )
        return wav_path, duration

    def _validate_file_type(self, file: UploadFile) -> None:
        """Check that the file has an allowed MIME type and extension.

        Args:
            file: Uploaded file.

        Raises:
            HTTPException: If file type is not supported.
        """
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            logger.warning("Rejected file with extension: {}", ext)
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        content_type = file.content_type or ""
        # Allow application/octet-stream (curl default) when extension is valid
        if (
            content_type
            and content_type != "application/octet-stream"
            and content_type not in ALLOWED_CONTENT_TYPES
        ):
            logger.warning("Rejected file with content type: {}", content_type)
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported content type '{content_type}'.",
            )

    async def _read_and_validate_size(self, file: UploadFile) -> bytes:
        """Read file bytes in chunks and enforce size limit.

        Reads in 64 KB chunks to avoid loading oversized files fully
        into memory before rejecting them.

        Args:
            file: Uploaded file.

        Returns:
            Raw file bytes.

        Raises:
            HTTPException: If file exceeds size limit.
        """
        chunks: list[bytes] = []
        total_size = 0
        chunk_size = 64 * 1024  # 64 KB

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > self._max_file_size:
                max_mb = self._max_file_size / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds maximum allowed ({max_mb:.0f} MB).",
                )
            chunks.append(chunk)

        return b"".join(chunks)

    def _convert_to_wav(self, raw_bytes: bytes, filename: str) -> tuple[Path, float]:
        """Convert audio bytes to WAV format if needed.

        Args:
            raw_bytes: Raw audio file content.
            filename: Original filename for format detection.

        Returns:
            Tuple of (path to temporary WAV file, duration in seconds).

        Raises:
            HTTPException: If audio cannot be decoded.
        """
        ext = Path(filename).suffix.lower()
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_input.write(raw_bytes)
        tmp_input.close()
        input_path = Path(tmp_input.name)

        try:
            if ext == ".mp3":
                audio = AudioSegment.from_mp3(str(input_path))
            else:
                audio = AudioSegment.from_wav(str(input_path))

            duration_seconds = len(audio) / 1000.0

            # Export as 16kHz mono WAV — optimal for Whisper
            wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav_tmp.close()
            wav_path = Path(wav_tmp.name)

            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(str(wav_path), format="wav")

            # Clean up original temp file if it was converted
            if ext != ".wav":
                input_path.unlink(missing_ok=True)

            return wav_path, duration_seconds

        except Exception as exc:
            input_path.unlink(missing_ok=True)
            logger.error("Failed to process audio file: {}", str(exc))
            raise HTTPException(
                status_code=422,
                detail="Could not process audio file. Ensure it is a valid WAV or MP3.",
            ) from exc
