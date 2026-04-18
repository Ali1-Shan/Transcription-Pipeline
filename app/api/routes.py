"""API route definitions.

Thin controller layer — delegates all business logic to the
TranscriptionService. Routes handle HTTP concerns only:
request parsing, auth, rate limiting, response formatting.
"""

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import verify_api_key
from app.config import get_settings
from app.core.formatter import OutputFormatter
from app.core.input_handler import InputHandler
from app.core.postprocessor import PostProcessor
from app.core.transcriber import Transcriber
from app.database import get_db
from app.models.schemas import HealthResponse, TranscriptionResponse
from app.services.transcription import TranscriptionService

router = APIRouter()

# Initialize pipeline components once at module scope
_settings = get_settings()
_service = TranscriptionService(
    input_handler=InputHandler(_settings),
    transcriber=Transcriber(_settings),
    post_processor=PostProcessor(_settings),
    formatter=OutputFormatter(),
)

limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
    description="Upload a WAV or MP3 file and receive a structured transcription.",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(f"{_settings.rate_limit_per_minute}/minute")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV or MP3, max 25 MB)"),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Accept an audio upload and return the transcription result."""
    request_id = getattr(request.state, "request_id", "unknown")

    response, transcript_id = await _service.transcribe_file(
        file=file,
        db=db,
        request_id=request_id,
    )

    return JSONResponse(
        content=response.model_dump(),
        headers={"X-Transcript-ID": transcript_id},
    )


@router.get(
    "/transcript/{transcript_id}",
    summary="Retrieve a stored transcript",
    description="Fetch a previously completed transcription result by its ID.",
    dependencies=[Depends(verify_api_key)],
)
async def get_transcript(
    transcript_id: str,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Retrieve a transcript from the database by ID."""
    data = await _service.get_transcript(transcript_id, db)
    if data is None:
        raise HTTPException(status_code=404, detail="Transcript not found.")
    return JSONResponse(content=data)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the service health status.",
)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="healthy", version="1.0.0")
