# Audio Transcription Pipeline

Production-ready audio transcription service built with **FastAPI** and **OpenAI Whisper**. Accepts WAV/MP3 uploads and returns structured JSON with transcripts, word-level timestamps, speaker segmentation, and metadata. Persists results in a database for later retrieval.

---

## Features

- **Multi-format support** — WAV and MP3 input with automatic normalization to 16kHz mono
- **Word-level timestamps** — Precise start/end time for every transcribed word
- **Post-processing** — Filler word removal, punctuation correction, mock speaker segmentation
- **Structured JSON output** — Consistent response schema with confidence scores and metadata
- **Database persistence** — Transcripts stored in SQLAlchemy async DB (SQLite/PostgreSQL), retrievable by ID
- **Production-hardened** — API key auth, rate limiting, request ID tracing, structured logging, global exception handling, CORS
- **Async architecture** — Non-blocking transcription via thread pool executor with concurrency semaphore
- **Retry logic** — Automatic retry with backoff on transient transcription failures
- **CLI support** — Batch processing without running the API server
- **Docker-ready** — Multi-stage build with pre-cached model and non-root user

---

## Architecture

```
Audio File (WAV/MP3)
        │
        ▼
┌──────────────────┐
│   API Routes      │  Auth, rate limiting, request ID — thin controller
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Transcription     │  Orchestrates full pipeline + DB persistence
│    Service        │
└────────┬─────────┘
         │
    ┌────┼────────────────┐
    ▼    ▼                ▼
┌──────┐ ┌──────────┐ ┌──────────┐
│Input │ │Transcriber│ │Post-     │
│Handler│ │(Whisper)  │ │Processor │
└──────┘ └──────────┘ └──────────┘
                          │
                          ▼
                   ┌──────────┐
                   │ Formatter │
                   └──────┬───┘
                          │
                          ▼
                   ┌──────────┐
                   │ Database  │  SQLAlchemy async (SQLite / PostgreSQL)
                   └──────────┘
```

Each component is a standalone class with a single responsibility — independently testable, replaceable, and configurable.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI | Async HTTP server with auto-generated OpenAPI docs |
| **STT Engine** | OpenAI Whisper | State-of-the-art speech recognition (local, offline) |
| **Data Validation** | Pydantic v2 | Request/response schema enforcement |
| **Database** | SQLAlchemy 2.0 (async) + aiosqlite | Transcript persistence (swappable to PostgreSQL) |
| **Configuration** | pydantic-settings + python-dotenv | Environment-based config management |
| **Authentication** | API key header (optional) | Protect endpoints when `API_KEY` is set |
| **Logging** | Loguru | Structured JSON logging with rotation and retention |
| **Rate Limiting** | SlowAPI | Per-IP request throttling |
| **Middleware** | Request ID tracing | Distributed tracing via `X-Request-ID` header |
| **Audio Processing** | pydub + ffmpeg | Format conversion and normalization |
| **Testing** | pytest + pytest-asyncio + httpx | Async-compatible test suite (35 tests) |

---

## Project Structure

```
transcription_pipeline/
├── app/
│   ├── main.py               # FastAPI app, lifespan, middleware, global error handler
│   ├── config.py             # Pydantic BaseSettings with env-based config
│   ├── database.py           # Async SQLAlchemy engine, session factory, lifecycle
│   ├── middleware.py         # Request ID tracing middleware
│   ├── cli.py                # CLI for batch transcription
│   ├── api/
│   │   ├── routes.py         # POST /transcribe, GET /transcript/{id}, GET /health
│   │   └── auth.py           # Optional API key authentication dependency
│   ├── core/
│   │   ├── input_handler.py  # File validation, size check, MP3 → WAV conversion
│   │   ├── transcriber.py    # Async Whisper wrapper with retry + semaphore
│   │   ├── postprocessor.py  # Filler removal, punctuation, speaker segmentation
│   │   └── formatter.py      # Structured JSON output builder
│   ├── models/
│   │   ├── schemas.py        # Pydantic v2 request/response models
│   │   └── transcript.py     # SQLAlchemy ORM model for transcript storage
│   ├── services/
│   │   └── transcription.py  # Service layer — orchestrates pipeline + persistence
│   └── utils/
│       └── logger.py         # Loguru structured logger setup
├── tests/
│   ├── conftest.py           # Shared test fixtures
│   ├── test_api.py           # API endpoint tests (5 cases)
│   ├── test_store.py         # Database storage tests (5 cases)
│   ├── test_production.py    # Auth, request ID, middleware tests (6 cases)
│   ├── test_postprocessor.py # Post-processor unit tests (12 cases)
│   └── test_transcriber.py   # Transcriber unit tests (5 cases)
├── Dockerfile                # Multi-stage build, non-root, pre-cached model
├── .env.example              # Configuration template
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

### Prerequisites

- **Python 3.11+**
- **ffmpeg** (required for audio processing)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (chocolatey)
choco install ffmpeg
```

### Local Installation

```bash
cd transcription_pipeline

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1       # Windows PowerShell
# source venv/bin/activate        # Linux/macOS

# Install dependencies
pip install setuptools wheel
pip install openai-whisper --no-build-isolation
pip install -r requirements.txt

# Configure
copy .env.example .env
# Edit .env as needed (model size, rate limits, etc.)
```

### Run the Server

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## Usage

### API — Transcribe a File

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -H "X-API-Key: your-key-here" \
  -F "file=@audio.wav;type=audio/wav"
```

> Omit `X-API-Key` if `API_KEY` is not set in your `.env`.

**Response:**
```json
{
  "transcript": "Hi there, this is a sample voice recording created for speech synthesis testing.",
  "confidence": 0.9995,
  "language": "en",
  "processing_time_seconds": 12.34,
  "timestamps": [
    {"word": "Hi", "start": 0.0, "end": 0.3},
    {"word": "there", "start": 0.3, "end": 0.72}
  ],
  "segments": [
    {
      "speaker": "Speaker 1",
      "text": "Hi there, this is a sample voice recording...",
      "start": 0.0,
      "end": 30.0
    }
  ],
  "metadata": {
    "filename": "audio.wav",
    "duration_seconds": 26.3,
    "model_used": "whisper-tiny"
  }
}
```

The response includes an `X-Transcript-ID` header for later retrieval.

### API — Retrieve a Transcript

```bash
curl http://127.0.0.1:8000/transcript/{transcript_id}
```

### API — Health Check

```bash
curl http://127.0.0.1:8000/health
# {"status": "healthy", "version": "1.0.0"}
```

### CLI — Batch Processing

```bash
# Basic transcription
python -m app.cli transcribe audio.wav

# Custom model + output file
python -m app.cli transcribe audio.mp3 --model small --output result.json

# Disable post-processing steps
python -m app.cli transcribe audio.wav --no-fillers --no-segmentation
```

---

## Docker

### Build

```bash
docker build -t transcription-pipeline .
```

### Run

```bash
docker run -p 8000:8000 transcription-pipeline
```

### Override Configuration

```bash
docker run -p 8000:8000 \
  -e WHISPER_MODEL_SIZE=base \
  -e LOG_LEVEL=DEBUG \
  -e RATE_LIMIT_PER_MINUTE=20 \
  -e CORS_ORIGINS=https://app.example.com \
  transcription-pipeline
```

---

## Testing

```bash
pytest -v
```

**35 tests** covering:
- API endpoints: health, file validation (415), transcription with mocked service, JSON schema
- Database storage: save/retrieve, missing ID, timestamps round-trip, transcript ID header
- Production features: request ID middleware, API key auth (401/403/200)
- Post-processing: filler removal (7 cases), punctuation correction (4 cases), speaker segmentation (3 cases)
- Transcriber: model name, confidence computation, timestamp extraction, async transcribe

---

## Configuration

All settings via environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | Model variant: `tiny`, `base`, `small`, `medium`, `large` |
| `MAX_FILE_SIZE_BYTES` | `26214400` | Max upload size (25 MB) |
| `RATE_LIMIT_PER_MINUTE` | `10` | API rate limit per IP |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `2` | Concurrent Whisper inference limit (prevents OOM) |
| `API_KEY` | *(empty)* | API key for auth (empty = auth disabled) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/transcripts.db` | Async DB URL (swap to `postgresql+asyncpg://...` for production) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOG_DIR` | `logs` | Log file output directory |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `ENABLE_FILLER_REMOVAL` | `true` | Toggle filler word removal |
| `ENABLE_PUNCTUATION_CORRECTION` | `true` | Toggle punctuation correction |
| `ENABLE_SPEAKER_SEGMENTATION` | `true` | Toggle speaker segmentation |
| `SPEAKER_SEGMENT_INTERVAL` | `30` | Segment interval in seconds |

---

## Engineering Decisions

### Why Whisper (local) over Alternatives?

| Criteria | Whisper (local) | Vosk | Google STT API |
|---|---|---|---|
| Accuracy | State-of-the-art | Good | Excellent |
| Offline | Yes | Yes | No |
| Cost | Free (compute only) | Free | Pay-per-use |
| Languages | 99+ | Limited | 120+ |
| Privacy | Full control | Full control | Data leaves premises |

**Decision:** Best accuracy-to-cost ratio. Runs fully offline — no API costs, no data privacy concerns, configurable model sizes for latency/accuracy tradeoff.

### Async Design

Whisper inference is CPU-bound. We use `asyncio.run_in_executor()` to offload transcription to a thread pool, keeping FastAPI responsive for health checks and concurrent requests. A semaphore (`max_concurrent_transcriptions`) prevents OOM under concurrent load.

### Service Layer

`TranscriptionService` orchestrates the full pipeline (validate → transcribe → postprocess → format → persist). Routes are thin controllers that handle only HTTP concerns. This separation makes the business logic testable without spinning up a web server.

### Database-Backed Storage

Transcripts are persisted to an async SQLAlchemy database (SQLite for development, swappable to PostgreSQL via `DATABASE_URL`). No data is written to disk as flat files — all state lives in the database. Timestamps and segments are stored as JSON columns for flexibility.

### Post-Processing as Toggleable Steps

Each step (filler removal, punctuation, segmentation) is independently configurable via environment variables — enables A/B testing, per-use-case tuning, and faster processing when only raw transcription is needed.

### Language-Aware Processing

Punctuation correction uses English-specific rules and is automatically skipped for non-English transcriptions to avoid garbling output.

---

## Future Improvements

1. **Real-time streaming** — WebSocket endpoint with chunked audio processing
2. **Job queue** — Redis/Kafka for async batch processing with webhook callbacks
3. **Speaker diarization** — Replace mock segmentation with Pyannote for real multi-speaker attribution
4. **GPU acceleration** — CUDA support for 10-50x faster inference
5. **faster-whisper** — CTranslate2 backend for 4x speed improvement at same accuracy
6. **Caching** — Content-hash-based deduplication via Redis
7. **Observability** — Prometheus metrics for latency histograms and error rates

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` | Run `pip install "setuptools<81"` before installing whisper |
| `Port 8000 already in use` | Kill the process: `netstat -ano \| findstr :8000` then `Stop-Process -Id <PID>` |
| `ffmpeg not found` | Install ffmpeg and ensure it's in your PATH |
| Slow transcription on CPU | Switch to `WHISPER_MODEL_SIZE=tiny` in `.env` for faster processing |
| `SHA256 checksum mismatch` | Delete `~/.cache/whisper/` and retry — the model download was corrupted |
