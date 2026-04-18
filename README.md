# Audio Transcription Pipeline

A backend service that accepts audio files, transcribes them using OpenAI Whisper, and returns structured results with word-level timestamps, speaker segments, and confidence scores. Built with FastAPI, runs fully offline, and persists transcripts in a database for retrieval.

---

## Features

- Accepts WAV and MP3 — normalizes everything to 16kHz mono WAV before processing
- Word-level timestamps from Whisper with `word_timestamps=True`
- Async request handling — Whisper runs in a thread pool so the API stays responsive
- Concurrency control via semaphore (default: 2 parallel jobs) to prevent OOM
- Retry with backoff on transient transcription failures (2 retries, 1s delay)
- Post-processing: filler word removal, punctuation correction, speaker segmentation (each toggleable)
- Transcripts persisted to database, retrievable by ID
- API key auth, per-IP rate limiting, request ID tracing on every response
- CLI mode for batch processing without the server

---

## Architecture

```
Request → Routes (auth, rate limit) → TranscriptionService → DB
                                            │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                        InputHandler    Transcriber    PostProcessor
                        (validate,      (Whisper,      (fillers,
                         convert)        retry)        punctuation)
                                                           │
                                                      Formatter → Response
```

**Routes** are thin controllers — they parse HTTP, check auth, and delegate to the service.
**TranscriptionService** orchestrates the pipeline: validate → transcribe → postprocess → format → persist.
Each component is injected at startup and independently testable.

---

## Tech Stack

| What | Why |
|---|---|
| **FastAPI** | Async HTTP, auto-generated OpenAPI docs, dependency injection |
| **OpenAI Whisper** | Best accuracy-to-cost ratio, runs offline, supports 99+ languages |
| **SQLAlchemy 2.0 (async)** | Transcript persistence — SQLite for dev, PostgreSQL for prod |
| **pydub + ffmpeg** | Audio format conversion and normalization |
| **Loguru** | Structured JSON logging with rotation |
| **SlowAPI** | Per-IP rate limiting |
| **Pydantic v2** | Request/response validation and serialization |

---

## API

### `POST /transcribe`

Upload an audio file, get a transcript.

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "X-API-Key: your-key" \
  -F "file=@recording.wav"
```

```json
{
  "transcript": "Hello, this is a test recording.",
  "confidence": 0.98,
  "language": "en",
  "processing_time_seconds": 3.42,
  "timestamps": [
    {"word": "Hello", "start": 0.0, "end": 0.4}
  ],
  "segments": [
    {"speaker": "Speaker 1", "text": "Hello, this is...", "start": 0.0, "end": 30.0}
  ],
  "metadata": {
    "filename": "recording.wav",
    "duration_seconds": 12.5,
    "model_used": "whisper-tiny"
  }
}
```

Response includes an `X-Transcript-ID` header for later retrieval.

### `GET /transcript/{id}`

Retrieve a stored transcript by ID.

### `GET /health`

Returns `{"status": "healthy", "version": "1.0.0"}`. No auth required.

---

## Setup

```bash
cd transcription_pipeline
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows
# source venv/bin/activate           # Linux/macOS

pip install "setuptools<81" wheel
pip install openai-whisper --no-build-isolation
pip install -r requirements.txt

copy .env.example .env               # then edit as needed
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Requires **ffmpeg** on your PATH (`choco install ffmpeg` / `brew install ffmpeg` / `apt install ffmpeg`).

### Docker

```bash
docker build -t transcription-pipeline .
docker run -p 8000:8000 -e WHISPER_MODEL_SIZE=tiny transcription-pipeline
```

### Tests

```bash
pytest -v   # 35 tests
```

---

## Engineering Decisions

**Why Whisper (local)?**
No API costs, no data leaving the server, configurable model sizes for the speed/accuracy tradeoff. Runs fully offline.

**How is concurrency handled?**
Whisper is CPU-bound. It runs in `asyncio.run_in_executor()` so the event loop stays free. An `asyncio.Semaphore` caps concurrent jobs (default 2) to prevent memory exhaustion. SlowAPI rate-limits at the HTTP level.

**Why is audio not stored?**
Uploaded files are written to temp files, processed, and deleted in a `finally` block. Raw audio is large and often sensitive — storing it creates liability without clear benefit for this use case.

**Why are transcripts stored?**
So clients can retrieve results later via `GET /transcript/{id}`. Stored in SQLAlchemy async — SQLite for development, one config change to switch to PostgreSQL.

**How are failures handled?**
Transient errors (model glitches, corrupted segments) trigger up to 2 retries with 1s delay. Validation failures (wrong format, oversized) fail immediately with proper HTTP codes. A global exception handler ensures no raw stack traces reach clients.

---

## Limitations and Future Work

- **No job queue** — transcription is synchronous per request. For production scale, add Celery/Redis so clients get a job ID and poll for results.
- **Mock speaker segmentation** — splits by time intervals, not real diarization. Would need Pyannote or similar for actual multi-speaker attribution.
- **No cloud storage** — audio normalization uses local temp files. For horizontal scaling, swap to S3 for intermediate storage.
- **CPU only** — no GPU support. Adding CUDA or switching to `faster-whisper` (CTranslate2) would give 4-50x speedup.
- **No caching** — duplicate audio gets transcribed again. Content-hash deduplication via Redis would fix this.

---

## Project Structure

```
app/
├── main.py                 # App entry, lifespan, middleware, error handler
├── config.py               # All settings from env vars
├── database.py             # Async engine + session factory
├── middleware.py            # X-Request-ID tracing
├── api/
│   ├── routes.py           # 3 endpoints (transcribe, transcript, health)
│   └── auth.py             # Optional API key check
├── core/
│   ├── input_handler.py    # Validate + convert audio
│   ├── transcriber.py      # Whisper wrapper (async, retry, semaphore)
│   ├── postprocessor.py    # Text cleanup + segmentation
│   └── formatter.py        # Build response model
├── models/
│   ├── schemas.py          # Pydantic request/response types
│   └── transcript.py       # ORM model
├── services/
│   └── transcription.py    # Pipeline orchestration + DB persistence
└── utils/
    └── logger.py           # Loguru setup
tests/                      # 35 tests (API, DB, auth, processing, transcriber)
```
