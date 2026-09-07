"""
Orchestrator — runs the full pipeline for one video:
  fetch -> chunk -> (interactive) retrieve -> answer

Usage:
  export ANTHROPIC_API_KEY=sk-...
  python main.py "https://www.youtube.com/watch?v=zUzd-HgaFkw&list=PLpx4I-WPI8W81xyTc0jaIx4jWjFS98TN0"
"""

import sys
import json
from pathlib import Path

from fetch_transcript import extract_video_id, fetch_transcript
from chunk import chunk_transcript
from retrieve import Retriever
from answer import answer_question


def build_pipeline(video_url_or_id: str, window_seconds: float = 30.0):
    video_id = extract_video_id(video_url_or_id)

    print(f"Fetching transcript for {video_id} ...")
    captions = fetch_transcript(video_id)
    print(f"  {len(captions)} caption lines")

    chunks = chunk_transcript(captions, window_seconds=window_seconds)
    print(f"  merged into {len(chunks)} chunks (~{window_seconds:.0f}s each)")

    # cache to disk so eval.py can reuse without re-fetching
    cache_path = Path(f"chunks_{video_id}.json")
    cache_path.write_text(json.dumps({"video_id": video_id, "chunks": chunks}, indent=2))
    print(f"  cached chunks -> {cache_path}\n")

    return Retriever(chunks)


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "<youtube_url_or_id>"')
        sys.exit(1)

    retriever = build_pipeline(sys.argv[1])

    print("Ask questions about the video (Ctrl+C or empty line to quit).\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        result = answer_question(question, retriever)
        print(f"A: {result['answer']}")
        if result["chunks_used"]:
            cites = ", ".join(f"{c['start']:.0f}s" for c in result["chunks_used"])
            print(f"   (grounded in chunks at: {cites})")
        print()


if __name__ == "__main__":
    main()
