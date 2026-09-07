# YouTube Transcript Q&A

## North Star

> Given a single YouTube video and a question about it, return an answer
> grounded in the transcript, citing the approximate timestamp it came from
> — and explicitly say "Not covered in this video" when the transcript
> doesn't contain the answer.

Written before any of the code below. This is the rubric this project grades
itself against in `eval.py`, not a description written after the fact to
match what got built.

## Pipeline

```
YouTube URL  ->  fetch_transcript.py  ->  chunk.py  ->  retrieve.py  ->  answer.py
                 (captions API)          (30s windows)   (TF-IDF)      (LLM, grounded)
```

| Stage | File | In | Out |
|---|---|---|---|
| Fetch | `fetch_transcript.py` | video URL/ID | raw timestamped caption lines |
| Chunk | `chunk.py` | caption lines | ~30s merged, timestamped chunks |
| Retrieve | `retrieve.py` | chunks + question | top-k chunks by TF-IDF similarity |
| Answer | `answer.py` | question + top-k chunks | grounded answer or decline |
| Score | `eval.py` | frozen questions | per-question grade + summary table |

**Why TF-IDF instead of embeddings** — for a single video's transcript
(a few thousand words), TF-IDF is a fair, zero-dependency baseline. It also
gives a retrieval score cheap enough to gate on *before* calling the LLM
(see below). Swap in an embedding model only if the eval numbers show TF-IDF
is the bottleneck — don't upgrade it on assumption.

**Why the refusal check happens at retrieval, not generation** — a related
project (a WhatsApp-paper second brain) found that distance/similarity
scores for questions with no real answer looked statistically identical to
scores for real hits, once handed to a generator. The fix used here:
threshold the retrieval score (`MIN_RETRIEVAL_SCORE` in `answer.py`) and
decline *before* the LLM ever sees the question, rather than trusting the
model to self-censor on weak context.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

This fetches the transcript, chunks it, caches chunks to
`chunks_<video_id>.json`, then drops into an interactive Q&A loop.

## Running the eval

1. Run `main.py` once against your target video to produce
   `chunks_<video_id>.json`.
2. Open `questions.json` and write 10 questions **by watching/skimming the
   video yourself**, before running retrieval on any of them:
   - 5 single-fact questions (answer is stated directly)
   - 3 synthesis questions (answer needs 2+ separate parts of the transcript)
   - 2 negative questions (plausible-sounding, but the video doesn't cover them)
3. Run:
   ```bash
   python eval.py chunks_<video_id>.json questions.json
   ```
   It asks each question through the real pipeline, shows you the system's
   answer, and asks you to grade it y/n against your own notes.
4. Report the honest score per category in this README, e.g.:

   | Kind | Score |
   |---|---|
   | Single | _/5 |
   | Synthesis | _/3 |
   | Negative (should decline) | _/2 |

## Known limitations (stated up front)

- Single video only — no cross-video merging or playlist handling.
- TF-IDF retrieval will miss questions phrased very differently from the
  transcript's wording (no synonym understanding) — an embedding model would
  fix this, at the cost of needing an API key or local model server.
- `MIN_RETRIEVAL_SCORE` was picked by eyeballing synthetic test cases, not
  tuned against a real video's eval results — re-tune it once you have real
  eval numbers, and report what you changed it to and why.
- Auto-generated captions have no punctuation and inconsistent casing;
  chunking merges by time window, not by sentence, so a chunk can start or
  end mid-sentence.
