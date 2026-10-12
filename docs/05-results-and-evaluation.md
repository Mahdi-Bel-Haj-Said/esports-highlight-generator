# 05 · Results & Evaluation

## Evaluation dataset

There was no public benchmark for "esports highlights from caster audio", so I built one with the two editors:

- **20 VODs** (≈58 hours in total): Valorant (9), League of Legends (7), EA FC (4).
- For each VOD, an editor marked every moment they **would** put in a highlight reel, with a start and end time. That gave **612 ground-truth highlights** (≈30 per VOD).
- 6 VODs were used for tuning (weights, thresholds, window sizes) and **14 VODs were held out** for the final numbers below.
- A predicted candidate counts as a match for a ground-truth highlight if their time intervals overlap by at least 50% of the shorter one.

## Detection quality (held-out set, 14 VODs)

| Metric | Value |
|---|---|
| **Recall** (ground-truth highlights found in the candidate list, top 30) | **0.84** |
| **Precision@20** (top-20 candidates the editor would keep) | **0.78** |
| Precision@30 | 0.71 |
| Mean boundary error (start/end vs. editor cut) | ≈2.6 s |

### Ablation: what each signal contributes

| Signals used | Recall@30 | P@20 |
|---|---|---|
| Audio excitement only | 0.61 | 0.55 |
| Events (BART-MNLI) only | 0.66 | 0.60 |
| Audio + events | 0.77 | 0.70 |
| Audio + events + lexicon | 0.80 | 0.73 |
| **Audio + events + lexicon + Twitch clips (final)** | **0.84** | **0.78** |

**Takeaway:** no single signal was enough. Audio catches the hype but also triggers on loud ads and intros. Text catches the meaning but misses moments the casters only scream about. Fusing them gave **+23 recall points** over audio alone.

## Speech-to-text quality

Word Error Rate measured on 45 minutes of manually transcribed caster audio:

| Configuration | WER |
|---|---|
| `whisper-medium`, default | 22.4% |
| `whisper-large-v3`, default | 18.9% |
| `whisper-large-v3` + VAD | 17.1% |
| **`whisper-large-v3` + VAD + domain prompt (final)** | **13.2%** |

Most of the remaining errors were player handles and overlapping speech, which hurt event detection very little.

## Semantic search

100 natural-language queries written by editors and coaches (e.g. *"the pentakill in game 3"*, *"talk about the economy after round 12"*):

| Setup | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| BM25 only | 0.47 | 0.69 | 0.56 |
| MiniLM embeddings only | 0.58 | 0.83 | 0.68 |
| **Hybrid (0.8 embeddings + 0.2 BM25)** | **0.64** | **0.88** | **0.74** |

## Performance (3-hour VOD, single 24 GB GPU)

| Step | Time |
|---|---|
| Download + audio extraction | ~4 min (network-bound) |
| VAD + Whisper large-v3 | ~11 min |
| Audio features | ~40 s |
| BART-MNLI event detection | ~4 min |
| Scoring + NMS | < 5 s |
| Embeddings + FAISS index | ~25 s |
| BART-CNN titles (30 candidates) | ~40 s |
| **Total, until ready for review** | **≈22 min** |
| Reel render (5 min, 1080p) | ~2–3 min |

## Business impact

Measured over the first **6 weeks in production** (≈70 VODs processed), with editors logging their time:

| | Before | After |
|---|---|---|
| Review time per 3h VOD | ~150 min | **~45 min (−70%)** |
| Total time from VOD to published reel | ~3h30 | **~1h15** |
| Highlight reels published per week | 4–5 | **9–10** |
| Short-form clips published per week | ~6 | **~20** (thanks to the 9:16 export) |

The editors' time moved from **searching** footage to **editing and storytelling**, and the team could cover more matches without hiring.
