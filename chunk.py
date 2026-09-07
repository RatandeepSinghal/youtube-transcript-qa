"""
Stage 2 — Chunk
In:  raw caption records from fetch_transcript.py
Out: a list of chunks, each covering a ~window_seconds span of the video:
     [{"chunk_id": 0, "start": 0.0, "end": 32.1, "text": "..."}, ...]

Auto-generated captions arrive as short, choppy fragments (2-5 seconds each).
Answering questions against a single fragment is usually pointless — the
sentence carrying the answer is often split across three of them. This stage
merges fragments into fixed-size time windows so each chunk is long enough to
carry a real thought, but short enough that retrieval stays precise.
"""

import sys
import json
from pathlib import Path


def chunk_transcript(captions: list[dict], window_seconds: float = 30.0) -> list[dict]:
    if not captions:
        return []

    chunks = []
    current_texts = []
    window_start = captions[0]["start"]
    chunk_id = 0

    for cap in captions:
        cap_end = cap["start"] + cap["duration"]
        if cap["start"] - window_start >= window_seconds and current_texts:
            chunks.append({
                "chunk_id": chunk_id,
                "start": window_start,
                "end": current_texts[-1][1],
                "text": " ".join(t for t, _ in current_texts),
            })
            chunk_id += 1
            current_texts = []
            window_start = cap["start"]

        current_texts.append((cap["text"].strip(), cap_end))

    if current_texts:
        chunks.append({
            "chunk_id": chunk_id,
            "start": window_start,
            "end": current_texts[-1][1],
            "text": " ".join(t for t, _ in current_texts),
        })

    return chunks


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python chunk.py <transcript.json> [output.json] [window_seconds]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("chunks.json")
    window = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

    data = json.loads(in_path.read_text())
    chunks = chunk_transcript(data["captions"], window_seconds=window)

    out_path.write_text(json.dumps({"video_id": data["video_id"], "chunks": chunks}, indent=2))
    print(f"Produced {len(chunks)} chunks (~{window}s each) -> {out_path}")


if __name__ == "__main__":
    main()
