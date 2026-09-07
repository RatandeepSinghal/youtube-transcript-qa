"""
Stage 4 — Answer
In:  a question, plus the top-k retrieved chunks (with scores)
Out: an answer grounded in those chunks, citing a timestamp, or an explicit
     "not covered in this video" when retrieval confidence is low.

Uses Google's Gemini API (free tier: gemini-2.5-flash, no credit card
required -- get a key at https://aistudio.google.com). Swap the client in
call_llm() if you'd rather use a different provider later.

The refusal check happens BEFORE calling the LLM, using the retrieval score.
This matters: your teammate's second-brain project found that once
low-relevance chunks got handed to a generator, distance alone couldn't
distinguish a real answer from a hallucinated one. Catching it at the
retrieval gate -- instead of hoping the LLM declines on its own -- is more
reliable and gets tested directly in eval.py's negative questions.
"""

import os
import sys
import json
from pathlib import Path

from retrieve import Retriever
from chunk import format_timestamp

SYSTEM_PROMPT = """You answer questions about a YouTube video using ONLY the transcript excerpts provided. Rules:
- Base your answer strictly on the excerpts. Do not use outside knowledge.
- Every claim must be traceable to one of the excerpts. Reference the timestamp it came from, e.g. (around 2:15).
- If the excerpts do not contain enough information to answer, say exactly: "Not covered in this video." Do not guess or fill gaps with general knowledge.
- Keep answers concise: 2-4 sentences."""

# Below this TF-IDF similarity score, retrieval is too weak to trust -- decline
# before even calling the LLM. Tune this against your eval set's negative
# questions rather than assuming it's right.
MIN_RETRIEVAL_SCORE = 0.08


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    excerpt_blocks = []
    for c in chunks:
        ts = format_timestamp(c["start"])
        excerpt_blocks.append(f"[{ts}] {c['text']}")
    excerpts = "\n\n".join(excerpt_blocks)
    return f"Transcript excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Calls Gemini's free-tier API. Reads GEMINI_API_KEY from the environment."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_prompt,
        config={"system_instruction": system_prompt, "max_output_tokens": 400},
    )
    return response.text


def answer_question(question: str, retriever: Retriever, top_k: int = 4) -> dict:
    chunks = retriever.query(question, top_k=top_k)

    if not chunks or chunks[0]["score"] < MIN_RETRIEVAL_SCORE:
        return {
            "question": question,
            "answer": "Not covered in this video.",
            "reason": "retrieval_below_threshold",
            "top_score": chunks[0]["score"] if chunks else 0.0,
            "chunks_used": [],
        }

    user_prompt = build_user_prompt(question, chunks)
    answer_text = call_llm(SYSTEM_PROMPT, user_prompt)

    return {
        "question": question,
        "answer": answer_text.strip(),
        "reason": "answered",
        "top_score": chunks[0]["score"],
        "chunks_used": [{"start": c["start"], "score": c["score"]} for c in chunks],
    }


def main():
    if len(sys.argv) < 3:
        print('Usage: python answer.py <chunks.json> "<question>" [top_k]')
        sys.exit(1)

    chunks_path = Path(sys.argv[1])
    question = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY in your environment before running this stage.")
        sys.exit(1)

    data = json.loads(chunks_path.read_text())
    retriever = Retriever(data["chunks"])
    result = answer_question(question, retriever, top_k=top_k)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
