# 04 · Twitch API Integration

## Authentication

The service uses the **OAuth2 client-credentials flow** (app access token). Everything it reads (videos, clips, users) is public data, so no user login was needed.

```http
POST https://id.twitch.tv/oauth2/token
  ?client_id=...&client_secret=...&grant_type=client_credentials
```

- The token is cached in Redis with its `expires_in` TTL minus a 5-minute safety margin.
- On a `401` response the client refreshes the token once and retries automatically.
- `client_secret` lives only in environment variables, never in the code or the database.

## Endpoints used

| Endpoint | Used for |
|---|---|
| `GET /helix/users?login=` | Resolve channel names to `broadcaster_id` |
| `GET /helix/videos?user_id=&type=archive` | Discover new past broadcasts (VODs) for each tracked channel |
| `GET /helix/videos?id=` | Metadata for one VOD: duration, title, created_at |
| `GET /helix/clips?broadcaster_id=&started_at=&ended_at=` | All viewer clips created during a broadcast |

## Viewer clips as weak labels

Twitch clips are created by viewers **when something exciting happens**. Each clip returns a `vod_offset` (seconds into the VOD where the clip starts) and a `view_count`. That makes them a free, crowd-sourced signal of where the highlights are:

```python
# Illustrative snippet, written for this case study
import numpy as np

def clip_density(clips, duration_s, sigma=10.0):
    """Gaussian kernel density of clip positions, weighted by popularity."""
    t = np.arange(duration_s)
    density = np.zeros(duration_s, dtype=np.float32)
    for c in clips:
        if c.vod_offset is None:
            continue
        center = c.vod_offset + c.duration / 2
        density += np.log1p(c.view_count) * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    return density / density.max() if density.max() > 0 else density
```

Limits I designed around:

- **Small channels have few clips.** The clip signal is only 20% of the fused score, and the model still works with zero clips (it falls back to audio, events and lexicon).
- **`vod_offset` can be `null`** for older clips or deleted VODs. Those clips are skipped.
- Clips are **paginated** (cursor-based, 100 per page), and a popular tournament can have 1,000+ clips. The client follows `pagination.cursor` until it is empty.

## Scheduled sync

A Celery Beat task runs every 30 minutes:

1. For each tracked channel, list `archive` videos newer than the last synced one.
2. Create a VOD record and enqueue a processing job (auto-processing is enabled per channel).
3. Re-fetch clips for VODs from the last 72 hours, because viewers keep clipping after the stream ends.

## Rate limits & resilience

- Helix allows about **800 points per minute** per app token. The client reads `Ratelimit-Remaining` / `Ratelimit-Reset` headers and sleeps when the remaining budget drops below 10%.
- `429` and `5xx` responses get **exponential backoff with jitter** (via `tenacity`, max 5 attempts).
- All Twitch calls go through one `httpx.AsyncClient` with connection pooling and a 10 s timeout.
- Responses for VOD metadata are cached in Redis for 1 hour.
