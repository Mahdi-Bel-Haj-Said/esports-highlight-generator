# 06 · Challenges, Solutions & Lessons

## Challenges

### 1. Whisper hallucinating over music and crowd noise
**Problem:** during intros, breaks and celebrations, Whisper produced invented sentences or repeated one phrase dozens of times. Those fake words triggered false events.
**Solution:** Silero VAD before ASR, filters on `compression_ratio` and `avg_logprob`, and n-gram de-duplication. False event triggers in non-speech parts went close to zero.

### 2. Casters react *after* the action
**Problem:** the loudest moment is the reaction, which comes 2–6 s after the play. Clips centered on the peak missed the play.
**Solution:** asymmetric windows (`−8 s / +4 s`) tuned on the evaluation set, then snapped to sentence boundaries. Mean boundary error dropped from ~6 s to ~2.6 s.

### 3. No labelled data
**Problem:** there was no time or budget for a large annotation campaign.
**Solution:** zero-shot models (BART-MNLI) instead of training a classifier, Twitch viewer clips as weak labels, and a small but carefully labelled evaluation set (20 VODs). After that, editor approve/reject feedback became a growing labelled dataset, which I used to re-fit the fusion weights.

### 4. Loud ≠ interesting
**Problem:** sponsor ads, intro videos and replays are loud but are not highlights.
**Solution:** the `filler` label in zero-shot classification penalizes non-gameplay talk. On top of that, I added a "known jingle" filter: audio fingerprints of the channel's recurring intro and ad stingers are matched with cross-correlation and masked out.

### 5. GPU memory with three transformer models
**Problem:** Whisper large-v3, BART-large-MNLI and BART-large-CNN together approach the 24 GB VRAM limit when batch sizes are high.
**Solution:** fp16 everywhere, one GPU worker process with sequential steps, `torch.cuda.empty_cache()` between stages, and tuned batch sizes. No out-of-memory errors in production after that.

## What I'd do next

- **Vision signals:** kill-feed and scoreboard OCR or detection for each game, to catch moments where the casters are quiet.
- **A small supervised ranker** (e.g. LightGBM on the fused features) trained on the editor feedback dataset, which had passed 2,000 labelled candidates by the end of my contract.
- **Faster ASR** with `faster-whisper` (CTranslate2) or distil-whisper, which would roughly halve transcription time.
- **Live mode:** process the stream in 5-minute rolling chunks so clips are ready minutes after a play, not after the broadcast.
- **Multilingual casting:** Whisper already handles French and Arabic commentary, but the lexicons and zero-shot hypotheses would need translating and testing.

## Lessons learned

1. **Fuse weak signals instead of chasing one perfect model.** Four imperfect signals beat any single one by a wide margin.
2. **Build the evaluation set first.** Without the 20 labelled VODs, every tuning choice would have been a guess.
3. **Show *why* a clip was suggested.** Displaying the signal breakdown made editors trust and adopt the tool much faster.
4. **Design for retries.** Idempotent, cached pipeline steps meant a failure in step 7 never cost 15 minutes of GPU time again.
