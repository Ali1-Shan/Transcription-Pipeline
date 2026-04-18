# Audio Transcription Pipeline

Backend service that takes an audio file (WAV or MP3), transcribes it using OpenAI Whisper running locally, and returns structured JSON — full transcript, word-level timestamps, speaker segments, confidence score, and metadata. Results are stored in a database and retrievable by ID.

---

## What It Does

1. You upload an audio file via `POST /transcribe`
2. The service validates the file, converts it to 16kHz mono WAV, and runs Whisper
3. Raw output goes through post-processing (filler removal, punctuation, speaker segmentation)
4. You get back a structured JSON response with an `X-Transcript-ID` header
5. You can fetch it again later via `GET /transcript/{id}`

---

## Features

- **WAV and MP3 input** — auto-normalized to 16kHz mono for Whisper
- **Word-level timestamps** — start/end time for every transcribed word
- **Speaker segments** — time-interval based segmentation (toggleable)
- **Post-processing** — filler word removal, punctuation correction, each independently toggleable
- **Async processing** — Whisper runs in a thread pool, event loop stays free
- **Concurrency limit** — semaphore caps parallel transcription jobs to prevent OOM
- **Retry on failure** — 2 automatic retries with 1s delay for transient errors
- **Database storage** — transcripts persist in SQLAlchemy async (SQLite / PostgreSQL)
- **API key auth** — optional, enabled by setting `API_KEY` in env
- **Rate limiting** — per-IP throttling via SlowAPI
- **Request tracing** — every response carries an `X-Request-ID` header
- **CLI mode** — transcribe files directly without running the server

---

## Architecture

```
POST /transcribe
       │
       ▼
   API Routes ─── auth, rate limit, request ID
       │
       ▼
 TranscriptionService ─── orchestrates the full pipeline
       │
       ├── InputHandler ──── validate file type/size, convert to WAV
       ├── Transcriber ───── Whisper inference (thread pool, retry, semaphore)
       ├── PostProcessor ─── filler removal, punctuation, segmentation
       ├── Formatter ─────── assemble Pydantic response model
       └── Database ──────── persist transcript (SQLAlchemy async)
       │
       ▼
  JSON Response + X-Transcript-ID header
```

Routes are thin controllers. All business logic lives in `TranscriptionService`. Each component is a standalone class, injected at startup.

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| API | FastAPI | Async, dependency injection, auto OpenAPI docs |
| Transcription | OpenAI Whisper | Offline, free, 99+ languages, configurable model sizes |
| Database | SQLAlchemy 2.0 async + aiosqlite | SQLite for dev, swap to PostgreSQL with one env var |
| Audio | pydub + ffmpeg | Format detection, conversion, normalization |
| Validation | Pydantic v2 | Strict typing at API boundaries |
| Logging | Loguru | Structured JSON logs, file rotation, retention |
| Rate Limiting | SlowAPI | Per-IP request throttling |

---

## API Endpoints

### `POST /transcribe`

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

Returns `X-Transcript-ID` header for later retrieval.

### `GET /transcript/{id}`

Fetch a previously stored transcript.

### `GET /health`

Returns `{"status": "healthy", "version": "1.0.0"}`. No auth required.

---

## Setup

**Requires:** Python 3.11+ and ffmpeg on PATH.

```bash
cd transcription_pipeline
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows
# source venv/bin/activate           # Linux/macOS

pip install "setuptools<81" wheel
pip install openai-whisper --no-build-isolation
pip install -r requirements.txt

copy .env.example .env               # edit as needed
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

API docs at **http://127.0.0.1:8000/docs**

### Docker

```bash
docker build -t transcription-pipeline .
docker run -p 8000:8000 -e WHISPER_MODEL_SIZE=tiny transcription-pipeline
```

### CLI

```bash
python -m app.cli transcribe audio.wav
python -m app.cli transcribe audio.mp3 --model small --output result.json
```

---

## Configuration

All via environment variables (`.env` file):

| Variable | Default | What it does |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `MAX_FILE_SIZE_BYTES` | `26214400` | Upload limit (25 MB) |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `2` | Parallel Whisper jobs allowed |
| `RATE_LIMIT_PER_MINUTE` | `10` | Requests per IP per minute |
| `API_KEY` | *(empty)* | Set to enable auth, leave empty to disable |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/transcripts.db` | Swap to `postgresql+asyncpg://...` for prod |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ENABLE_FILLER_REMOVAL` | `true` | Remove "um", "uh", "you know", etc. |
| `ENABLE_PUNCTUATION_CORRECTION` | `true` | Capitalize, add periods |
| `ENABLE_SPEAKER_SEGMENTATION` | `true` | Split transcript by time intervals |

---

## Engineering Decisions

**Why Whisper locally instead of a cloud API?**
Zero cost per request, no data leaves the server, works offline. Model size is configurable — `tiny` for speed, `large` for accuracy.

**How does it handle concurrent requests?**
Whisper is CPU-heavy. It runs in `asyncio.run_in_executor()` so the event loop stays responsive. An `asyncio.Semaphore` limits parallel jobs (default 2) to prevent the process from running out of memory. SlowAPI adds HTTP-level rate limiting on top.

**Why not store the audio files?**
Audio is large and often contains sensitive content. Files are written to temp storage during processing and deleted in a `finally` block — even if transcription fails. The transcript is the deliverable, not the recording.

**Why persist transcripts to a database?**
Clients need to retrieve results later. SQLAlchemy async with SQLite keeps development simple. Switching to PostgreSQL is a single env var change — no code modifications.

**How does it recover from failures?**
Transient errors (model issues, corrupted audio segments) trigger up to 2 retries with a 1-second delay. Validation errors (wrong file type, too large) return immediately with the correct HTTP status code. A global exception handler catches anything unexpected and returns a clean JSON error — no stack traces leak to clients.

---

## Limitations

These are known trade-offs, not oversights:

- **Synchronous per request** — no job queue. Adding Celery/Redis would let clients submit and poll, which matters at scale.
- **Speaker segmentation is time-based** — not real diarization. Pyannote would solve this but adds significant complexity.
- **CPU only** — no GPU acceleration. `faster-whisper` (CTranslate2 backend) would give 4-50x speedup.
- **No deduplication** — same audio uploaded twice gets transcribed twice. Content hashing with Redis would fix this.
- **Local temp files** — works fine on a single server. For horizontal scaling, intermediate storage should move to S3.

---

## Project Structure

```
app/
├── main.py                 # FastAPI app, lifespan, middleware, error handler
├── config.py               # Environment-based settings (Pydantic BaseSettings)
├── database.py             # Async SQLAlchemy engine + session factory
├── middleware.py            # X-Request-ID tracing
├── cli.py                  # Command-line transcription
├── api/
│   ├── routes.py           # POST /transcribe, GET /transcript/{id}, GET /health
│   └── auth.py             # Optional API key authentication
├── core/
│   ├── input_handler.py    # File validation, size check, format conversion
│   ├── transcriber.py      # Whisper wrapper — async, retry, concurrency limit
│   ├── postprocessor.py    # Filler removal, punctuation, speaker segmentation
│   └── formatter.py        # Assembles final response model
├── models/
│   ├── schemas.py          # Pydantic request/response types
│   └── transcript.py       # SQLAlchemy ORM model
├── services/
│   └── transcription.py    # Pipeline orchestration + DB persistence
└── utils/
    └── logger.py           # Loguru structured logging setup
```
