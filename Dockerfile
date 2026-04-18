# =============================================================================
# Multi-stage Dockerfile for Audio Transcription Pipeline
# =============================================================================
# Stage 1: Builder — installs dependencies in a virtual environment
# Stage 2: Runtime — minimal image with only what's needed to run
# =============================================================================

# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Create venv and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    pip install --no-cache-dir openai-whisper==20231117 --no-build-isolation && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper model during build so it's cached in the image
ARG WHISPER_MODEL=tiny
RUN python -c "import whisper; whisper.load_model('${WHISPER_MODEL}')"

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim AS runtime

# Install runtime system dependencies (ffmpeg for audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pre-downloaded Whisper model from builder
COPY --from=builder /root/.cache/whisper /home/appuser/.cache/whisper

# Set working directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY .env.example .env

# Create logs and data directories with correct permissions
RUN mkdir -p /app/logs /app/data && chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
