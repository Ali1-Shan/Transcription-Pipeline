"""FastAPI application entry point.

Configures the application with:
- Database initialization (SQLAlchemy async)
- Request ID tracing middleware
- API key authentication
- Rate limiting (SlowAPI)
- Global exception handling
- Structured logging
- CORS (configurable via environment)
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from loguru import logger

from app.api.routes import router, limiter
from app.config import get_settings
from app.database import init_db, close_db
from app.middleware import RequestIDMiddleware
from app.models.schemas import ErrorResponse
from app.utils.logger import setup_logger


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    setup_logger(settings.log_level, settings.log_dir)
    await init_db()
    logger.info(
        "Transcription Pipeline starting | model={} | rate_limit={}/min | db={}",
        settings.whisper_model_size,
        settings.rate_limit_per_minute,
        settings.database_url.split("///")[0] + "///...",
    )
    yield
    # Shutdown
    await close_db()
    logger.info("Transcription Pipeline shutting down")


app = FastAPI(
    title="Audio Transcription Pipeline",
    description=(
        "Production-ready audio transcription service powered by OpenAI Whisper. "
        "Accepts WAV/MP3 files and returns structured transcription with timestamps, "
        "speaker segmentation, and metadata."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- Middleware (order matters: last added = first executed) ---

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Parse CORS origins from comma-separated config value
cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID tracing — must be outermost middleware
app.add_middleware(RequestIDMiddleware)


# --- Global Exception Handler ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that prevents raw stack traces from reaching clients.

    Args:
        request: The incoming request.
        exc: The unhandled exception.

    Returns:
        Clean JSON error response.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception | path={} | error={} | request_id={}",
        request.url.path,
        str(exc),
        request_id,
    )
    error = ErrorResponse(
        error="internal_server_error",
        detail="An unexpected error occurred. Please try again later.",
        status_code=500,
    )
    return JSONResponse(status_code=500, content=error.model_dump())


# --- Register Routes ---

app.include_router(router, tags=["Transcription"])
