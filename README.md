# YouTube Transcript Q\&A

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
YouTube URL  ->  fetch\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_transcript.py  ->  chunk.py  ->  retrieve.py  ->  answer.py
                 (captions API)          (30s windows)   (TF-IDF)      (LLM, grounded)
```

|Stage|File|In|Out|
|-|-|-|-|
|Fetch|`fetch\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_transcript.py`|video URL/ID|raw timestamped caption lines|
|Chunk|`chunk.py`|caption lines|\~30s merged, timestamped chunks|
|Retrieve|`retrieve.py`|chunks + question|top-k chunks by TF-IDF similarity|
|Answer|`answer.py`|question + top-k chunks|grounded answer or decline|
|Score|`eval.py`|frozen questions|per-question grade + summary table|

**Why TF-IDF instead of embeddings** — for a single video's transcript
(a few thousand words), TF-IDF is a fair, zero-dependency baseline. It also
gives a retrieval score cheap enough to gate on *before* calling the LLM
(see below). Swap in an embedding model only if the eval numbers show TF-IDF
is the bottleneck — don't upgrade it on assumption.

**Why the refusal check happens at retrieval, not generation** — a related
project (a WhatsApp-paper second brain) found that distance/similarity
scores for questions with no real answer looked statistically identical to
scores for real hits, once handed to a generator. The fix used here:
threshold the retrieval score (`MIN\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_RETRIEVAL\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_SCORE` in `answer.py`) and
decline *before* the LLM ever sees the question, rather than trusting the
model to self-censor on weak context.

## Running it

```bash
pip install -r requirements.txt
export GEMINI\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_API\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_KEY=...   # free tier, no card required: https://aistudio.google.com

python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

This fetches the transcript, chunks it, caches chunks to
`chunks\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_<video\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_id>.json`, then drops into an interactive Q\&A loop.

## Running the eval

1. Run `main.py` once against your target video to produce
`chunks\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_<video\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_id>.json`.
2. Open `questions.json` and write 10 questions **by watching/skimming the
video yourself**, before running retrieval on any of them:

   * 5 single-fact questions (answer is stated directly)
   * 3 synthesis questions (answer needs 2+ separate parts of the transcript)
   * 2 negative questions (plausible-sounding, but the video doesn't cover them)
3. Run:

```bash
   python eval.py chunks\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_<video\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_id>.json questions.json
   ```

It asks each question through the real pipeline, shows you the system's
answer, and asks you to grade it y/n against your own notes.

4. Report the honest score per category in this README, e.g.:

|Kind|Score|
|-|-|
|Single|4/5|
|Synthesis|2/3|
|Negative (should decline)|2/2|

## Known limitations (stated up front)

* Eval results:\* 4/5 single, 1/3 synthesis, 2/2 negative (7/10 overall).
* Weakest category: synthesis.\* Questions that needed two separate parts of the transcript stitched together (e.g. a stat mentioned early in the video, combined with a detail mentioned much later) mostly failed. This points to a real limitation of TF-IDF retrieval: it scores each chunk independently against the question, so it has no way to recognize that two distant, differently-worded chunks both relate to the same answer. It's good at "find the one chunk that talks about X," and weak at "connect chunk A and chunk C into one answer."
* What worked well: negative questions (2/2).\* The retrieval-score threshold (MIN\_RETRIEVAL\_SCORE in answer.py) correctly caught both questions the video didn't actually cover and declined instead of guessing -- this was the single biggest risk in the design and it held up.
* What would likely help:\* increasing top\_k in retrieve.py (more chunks handed to the LLM per question) is the cheapest thing to try next, since the LLM might successfully combine two chunks itself if it's given more of them. This hasn't been tested yet -- it's a next step, not a claimed fix.

