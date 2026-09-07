"""
Stage 3 — Retrieve
In:  chunks from chunk.py, plus a question
Out: the top-k chunks most relevant to the question, each with a score

Uses TF-IDF + cosine similarity — deliberately not an embedding API. This
keeps the whole pipeline runnable with zero API keys and zero network calls
once the transcript is fetched, and it's a fair baseline: if TF-IDF already
answers most questions, you don't need embeddings for a single video's
worth of text. Swap in a real embedding model later if retrieval quality
turns out to be the bottleneck (check this against your eval numbers first).
"""

import sys
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        texts = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def query(self, question: str, top_k: int = 4) -> list[dict]:
        if self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


def main():
    if len(sys.argv) < 3:
        print('Usage: python retrieve.py <chunks.json> "<question>" [top_k]')
        sys.exit(1)

    chunks_path = Path(sys.argv[1])
    question = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    data = json.loads(chunks_path.read_text())
    retriever = Retriever(data["chunks"])
    results = retriever.query(question, top_k=top_k)

    for r in results:
        print(f"[{r['start']:.0f}s] score={r['score']:.3f}  {r['text'][:120]}")


if __name__ == "__main__":
    main()
