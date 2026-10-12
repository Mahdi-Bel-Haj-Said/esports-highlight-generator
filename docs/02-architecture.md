# 02 · System Architecture

## Components

```mermaid
flowchart TB
    subgraph Client
        DASH[Editor dashboard<br/>React, internal]
    end

    subgraph API_Layer["API layer (CPU container)"]
        API[FastAPI app<br/>Uvicorn + Gunicorn]
    end

    subgraph Queue
        REDIS[(Redis<br/>Celery broker + result backend)]
    end

    subgraph Workers["Worker layer"]
        CPUW[CPU worker<br/>download, FFmpeg, render]
        GPUW[GPU worker<br/>Whisper, BART, embeddings]
    end

    subgraph Storage
        PG[(PostgreSQL)]
        S3[(S3-compatible storage)]
        FAISS[(FAISS indexes<br/>one per VOD + global)]
    end

    TWITCH[Twitch Helix API]

    DASH -->|REST / JSON| API
    API --> PG
    API -->|enqueue| REDIS
    REDIS --> CPUW
    REDIS --> GPUW
    CPUW --> S3
    GPUW --> S3
    CPUW --> PG
    GPUW --> PG
    GPUW --> FAISS
    API --> FAISS
    CPUW --> TWITCH
    API --> TWITCH
```

### Why this layout

- **The API never runs heavy work.** Requests return in milliseconds with a `job_id`, and everything long runs on Celery workers.
- **Separate CPU and GPU queues.** Downloading and FFmpeg rendering are I/O- and CPU-bound. Whisper and the transformers are GPU-bound. With two queues (`cpu`, `gpu`), a render job never blocks transcription, and the GPU worker runs with `concurrency=1` to avoid out-of-memory errors.
- **Models load once per worker process** (in a `worker_process_init` hook) and stay in GPU memory. Loading Whisper `large-v3` cold takes ~20 s, so reloading it per task would have been wasteful.

## Job lifecycle

```mermaid
stateDiagram-v2
    [*] --> QUEUED
    QUEUED --> DOWNLOADING
    DOWNLOADING --> TRANSCRIBING
    TRANSCRIBING --> ANALYZING : ASR + audio features done
    ANALYZING --> INDEXING : events scored
    INDEXING --> READY_FOR_REVIEW : candidates + search index ready
    READY_FOR_REVIEW --> RENDERING : editor approves clips
    RENDERING --> DONE
    DOWNLOADING --> FAILED
    TRANSCRIBING --> FAILED
    ANALYZING --> FAILED
    RENDERING --> FAILED
    FAILED --> QUEUED : retry (max 3, exponential backoff)
```

Every step is **idempotent** and writes its output to object storage under a deterministic key (`vods/{vod_id}/audio.wav`, `vods/{vod_id}/transcript.json`, …). A retried job skips steps whose output already exists. This saved a lot of GPU time when a late step such as rendering failed.

The pipeline is a Celery **chain**: `download → (transcribe ∥ audio_features) → detect_events → score → (index ∥ summarize)`. The `∥` parts run as a `group` and join in a `chord`.

## Data model (simplified)

```mermaid
erDiagram
    VOD ||--o{ JOB : has
    VOD ||--o{ TRANSCRIPT_SEGMENT : has
    VOD ||--o{ TWITCH_CLIP : has
    VOD ||--o{ CANDIDATE : has
    CANDIDATE ||--o{ EVENT : contains
    JOB ||--o| REEL : produces
    REEL ||--o{ REEL_CLIP : contains
    CANDIDATE ||--o{ REEL_CLIP : used_in

    VOD {
        uuid id
        string twitch_video_id
        string game
        string title
        int duration_s
        string storage_key
    }
    TRANSCRIPT_SEGMENT {
        uuid id
        float start_s
        float end_s
        text text
        float avg_logprob
        int faiss_row
    }
    TWITCH_CLIP {
        string clip_id
        int vod_offset_s
        int view_count
        string title
    }
    CANDIDATE {
        uuid id
        float start_s
        float end_s
        float score
        json signal_breakdown
        string status "pending|approved|rejected"
        string title
        text description
    }
    EVENT {
        uuid id
        string label "multi_kill|clutch|objective|..."
        float confidence
        float t_s
    }
    REEL {
        uuid id
        string format "16:9|9:16"
        string storage_key
        int duration_s
    }
```

`signal_breakdown` stores each signal's contribution to the score (audio, events, clips, keywords). The dashboard shows it, so editors can see **why** a moment was suggested. That transparency helped them trust the system.

Editor feedback (`approved` / `rejected`) is kept. I used it to re-tune the fusion weights (see [ML pipeline §5](03-ml-pipeline.md#5-highlight-scoring-signal-fusion)).

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/v1/jobs` | Create a job from a Twitch video ID, a Twitch URL or an uploaded file |
| `GET` | `/v1/jobs/{job_id}` | Status, progress %, current step, errors |
| `GET` | `/v1/vods/{vod_id}/candidates?min_score=&limit=` | Ranked list of highlight candidates with their signal breakdown |
| `PATCH` | `/v1/candidates/{id}` | Approve or reject a candidate, or adjust its start and end |
| `POST` | `/v1/search` | Semantic search over one VOD or all VODs |
| `POST` | `/v1/vods/{vod_id}/render` | Render a reel from the approved candidates (`16:9` or `9:16`) |
| `GET` | `/v1/reels/{reel_id}` | Reel metadata and a pre-signed download URL |
| `POST` | `/v1/twitch/sync` | Pull the latest VODs and clips for the configured channels |
| `GET` | `/health`, `/metrics` | Liveness, and Prometheus metrics (queue depth, step durations) |

Example: creating a job

```http
POST /v1/jobs
Content-Type: application/json

{
  "source": "twitch",
  "twitch_video_id": "2054xxxxxx",
  "game": "valorant",
  "options": { "language": "en", "target_reel_minutes": 5, "vertical_shorts": true }
}
```

```json
{ "job_id": "8f1c…", "vod_id": "c2a7…", "status": "QUEUED" }
```

Example: semantic search

```http
POST /v1/search
{ "query": "ace on B site with the operator", "vod_id": "c2a7…", "top_k": 5 }
```

```json
[
  { "start_s": 6412.0, "end_s": 6431.5, "score": 0.71,
    "text": "…he's got the op, one, two, THREE, FOUR, THAT'S THE ACE ON B…",
    "candidate_id": "a91e…" }
]
```

## Infrastructure

- **Docker Compose** with five services: `api`, `worker-cpu`, `worker-gpu`, `redis`, `postgres`. Object storage was an S3-compatible bucket.
- The GPU worker runs on the **NVIDIA Container Toolkit** on a single 24 GB VRAM machine.
- Configuration with `pydantic-settings` (`.env`), and secrets (Twitch client secret, DB password) only in environment variables.
- **Observability:** structured JSON logs with `job_id` on every line, Prometheus counters and histograms for step duration, and Flower for watching Celery queues.
- **Testing:** unit tests for scoring, NMS and window merging (pure functions), and integration tests on a 10-minute fixture VOD that runs the full pipeline with small models (`whisper-tiny`, a distilled MNLI model) in CI.
