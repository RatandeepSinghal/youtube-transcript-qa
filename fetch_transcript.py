"""
Stage 1 — Fetch
In:  a YouTube video ID or URL
Out: a list of caption records: [{"text": ..., "start": ..., "duration": ...}, ...]

Uses the youtube-transcript-api library (no API key, no auth). This only
works for videos that have captions available (auto-generated or manual).
"""

import re
import sys
import json
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


def extract_video_id(url_or_id: str) -> str:
    """Accept a bare video ID or a full YouTube URL and return the video ID."""
    # Already looks like a bare ID (11 chars, no slashes/dots)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=|/)([A-Za-z0-9_-]{11})(?:[&?/]|$)",  # watch?v=..., /embed/..., youtu.be/...
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a video ID from: {url_or_id}")


def fetch_transcript(video_id: str, languages=("en",)) -> list[dict]:
    """Fetch the transcript for one video. Raises with a clear message on failure."""
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=list(languages))
    except TranscriptsDisabled:
        raise RuntimeError(f"Captions are disabled for video {video_id}.")
    except NoTranscriptFound:
        raise RuntimeError(
            f"No transcript found for video {video_id} in languages {languages}."
        )
    except VideoUnavailable:
        raise RuntimeError(f"Video {video_id} is unavailable (private, deleted, or region-locked).")

    records = [
        {"text": snippet.text, "start": snippet.start, "duration": snippet.duration}
        for snippet in fetched
    ]
    return records


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_transcript.py <video_url_or_id> [output.json]")
        sys.exit(1)

    video_id = extract_video_id(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"transcript_{video_id}.json")

    records = fetch_transcript(video_id)
    out_path.write_text(json.dumps({"video_id": video_id, "captions": records}, indent=2))

    print(f"Fetched {len(records)} caption lines for {video_id} -> {out_path}")


if __name__ == "__main__":
    main()
