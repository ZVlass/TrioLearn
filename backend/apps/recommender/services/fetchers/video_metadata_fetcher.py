import os
import time
import math
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

def load_config():
    load_dotenv()  # loads .env from project root if present

    project_root = os.getenv("PROJECT_ROOT", os.getcwd())

    # Keys: prefer YOUTUBE_API_KEY; fall back to GOOGLE_YOUTUBE_API_KEY if provided
    yt_api_key = (
        os.getenv("YOUTUBE_API_KEY")
        or os.getenv("GOOGLE_YOUTUBE_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not yt_api_key:
        raise RuntimeError(
            "YouTube API key not found. Please set YOUTUBE_API_KEY in your .env"
        )

    # Paths & settings (mirror the naming style used for books)
    fetch_out = os.getenv(
        "VIDEO_FETCH_OUT",
        os.path.join(project_root, "backend", "data", "interim_large"),
    )
    os.makedirs(fetch_out, exist_ok=True)
    meta_csv = os.getenv(
        "VIDEO_META_CSV",
        os.path.join(fetch_out, "videos_metadata.csv"),
    )

    per_query = int(os.getenv("VIDEO_FETCH_PER_KEYWORD", "200"))
    delay = float(os.getenv("VIDEO_FETCH_DELAY", "0.5"))
    max_retries = int(os.getenv("VIDEO_FETCH_MAX_RETRIES", "3"))

    return {
        "PROJECT_ROOT": project_root,
        "YOUTUBE_API_KEY": yt_api_key,
        "VIDEO_FETCH_OUT": fetch_out,
        "VIDEO_META_CSV": meta_csv,
        "VIDEO_FETCH_PER_KEYWORD": per_query,
        "VIDEO_FETCH_DELAY": delay,
        "VIDEO_FETCH_MAX_RETRIES": max_retries,
    }


def youtube_client(api_key: str):
    # build() will handle discovery; no need for developerKey elsewhere
    return build("youtube", "v3", developerKey=api_key)


def parse_iso8601_duration(dur: str) -> Optional[int]:
    """
    Convert ISO-8601 duration (e.g., 'PT1H2M3S') to seconds.
    Works without external deps like 'isodate'.
    """
    if not dur or not dur.startswith("P"):
        return None
    # strip leading 'P'
    t = dur[1:]
    days = hours = minutes = seconds = 0

    # split date/time parts
    if "T" in t:
        date_part, time_part = t.split("T", 1)
    else:
        date_part, time_part = t, ""

    # date part: Y M W D (we only expect D for YouTube)
    num = ""
    for ch in date_part:
        if ch.isdigit():
            num += ch
        else:
            if ch == "D":
                days = int(num or 0)
            # ignore Y/M/W if ever present for simplicity
            num = ""
    # time part: H M S
    num = ""
    for ch in time_part:
        if ch.isdigit():
            num += ch
        else:
            if ch == "H":
                hours = int(num or 0)
            elif ch == "M":
                minutes = int(num or 0)
            elif ch == "S":
                seconds = int(num or 0)
            num = ""

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# Fetchers

def search_video_ids(
    yt,
    query: str,
    target_count: int,
    delay: float,
    max_retries: int,
    safe_search: str = "none",
    published_after: Optional[str] = None,
    region_code: Optional[str] = None,
) -> List[str]:
    """
    Use search.list to collect video IDs up to target_count.
    """
    ids = []
    next_page = None

    while len(ids) < target_count:
        remaining = min(50, target_count - len(ids))
        for attempt in range(max_retries):
            try:
                req = yt.search().list(
                    q=query,
                    part="id",
                    type="video",
                    maxResults=remaining,
                    pageToken=next_page,
                    safeSearch=safe_search,  # none | moderate | strict
                    publishedAfter=published_after,  # RFC 3339 format if used
                    regionCode=region_code,
                )
                resp = req.execute()
                break
            except HttpError as e:
                if attempt + 1 == max_retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
        for item in resp.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                ids.append(vid)
        next_page = resp.get("nextPageToken")
        if not next_page:
            break
        time.sleep(delay)

    return ids[:target_count]


def fetch_video_metadata(yt, video_ids: List[str], delay: float, max_retries: int) -> List[Dict[str, Any]]:
    """
    Enrich IDs via videos.list to get snippet, statistics, contentDetails, etc.
    """
    records = []
    for batch in chunked(video_ids, 50):
        for attempt in range(max_retries):
            try:
                req = yt.videos().list(
                    id=",".join(batch),
                    part="snippet,contentDetails,statistics,topicDetails,status",
                    maxResults=50,
                )
                resp = req.execute()
                break
            except HttpError:
                if attempt + 1 == max_retries:
                    raise
                time.sleep(1.5 * (attempt + 1))

        for item in resp.get("items", []):
            vid = item.get("id")
            snip = item.get("snippet", {}) or {}
            stats = item.get("statistics", {}) or {}
            cdet = item.get("contentDetails", {}) or {}
            stat = item.get("status", {}) or {}
            topics = item.get("topicDetails", {}) or {}

            duration = cdet.get("duration")
            duration_seconds = parse_iso8601_duration(duration) if duration else None

            tags = snip.get("tags") or []
            # Flatten topicCategories if present
            topic_categories = topics.get("topicCategories") or []

            rec = {
                # identity
                "video_id": vid,
                "channel_id": snip.get("channelId"),
                "channel_title": snip.get("channelTitle"),
                "category_id": snip.get("categoryId"),

                # descriptive
                "title": snip.get("title"),
                "description": snip.get("description"),
                "tags": "|".join(tags) if tags else "",
                "default_language": snip.get("defaultLanguage"),
                "default_audio_language": snip.get("defaultAudioLanguage"),
                "published_at": snip.get("publishedAt"),
                "live_broadcast_content": snip.get("liveBroadcastContent"),

                # content details
                "duration": duration,
                "duration_seconds": duration_seconds,
                "dimension": cdet.get("dimension"),
                "definition": cdet.get("definition"),
                "caption": cdet.get("caption"),
                "licensed_content": cdet.get("licensedContent"),
                "projection": cdet.get("projection"),

                # statistics
                "view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
                "like_count": int(stats["likeCount"]) if "likeCount" in stats else None,
                "comment_count": int(stats["commentCount"]) if "commentCount" in stats else None,
                # favoriteCount exists but is always 0 historically; omit by default

                # topics/status
                "topic_categories": "|".join(topic_categories) if topic_categories else "",
                "privacy_status": stat.get("privacyStatus"),
                "made_for_kids": stat.get("madeForKids"),
            }
            records.append(rec)

        time.sleep(delay)

    return records


def fetch_youtube_videos_for_queries(
    queries: List[str],
    per_query: int,
    api_key: str,
    delay: float,
    max_retries: int,
    safe_search: str = "none",
    published_after: Optional[str] = None,
    region_code: Optional[str] = None,
) -> pd.DataFrame:
    yt = youtube_client(api_key)
    all_rows: List[Dict[str, Any]] = []

    for q in queries:
        ids = search_video_ids(
            yt,
            q,
            target_count=per_query,
            delay=delay,
            max_retries=max_retries,
            safe_search=safe_search,
            published_after=published_after,
            region_code=region_code,
        )
        rows = fetch_video_metadata(yt, ids, delay=delay, max_retries=max_retries)
        # Attach query for traceability/debugging
        for r in rows:
            r["source_query"] = q
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    # Stable column order
    cols = [
        "video_id", "title", "description", "channel_id", "channel_title",
        "category_id", "published_at", "duration", "duration_seconds",
        "dimension", "definition", "caption", "licensed_content", "projection",
        "view_count", "like_count", "comment_count",
        "tags", "topic_categories",
        "default_language", "default_audio_language",
        "live_broadcast_content", "privacy_status", "made_for_kids",
        "source_query",
    ]
    # Keep any new columns too
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    return df


# ---------------------------
# CLI
# ---------------------------
DEFAULT_QUERIES = [
    "machine learning tutorial",
    "introduction to deep learning",
    "data science for beginners",
    "python for machine learning",
    "neural networks explained",
]

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Fetch YouTube video metadata to CSV.")
    parser.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES, help="List of search queries.")
    parser.add_argument("--per_query", type=int, default=cfg["VIDEO_FETCH_PER_KEYWORD"], help="Videos per query.")
    parser.add_argument("--out_csv", default=cfg["VIDEO_META_CSV"], help="Output CSV path.")
    parser.add_argument("--safe_search", default="none", choices=["none", "moderate", "strict"])
    parser.add_argument("--published_after", default=None, help="RFC3339 timestamp, e.g., 2024-01-01T00:00:00Z")
    parser.add_argument("--region_code", default=None, help="e.g., US, GB")
    args = parser.parse_args()

    df = fetch_youtube_videos_for_queries(
        queries=args.queries,
        per_query=args.per_query,
        api_key=cfg["YOUTUBE_API_KEY"],
        delay=cfg["VIDEO_FETCH_DELAY"],
        max_retries=cfg["VIDEO_FETCH_MAX_RETRIES"],
        safe_search=args.safe_search,
        published_after=args.published_after,
        region_code=args.region_code,
    )

    # Make sure output dir exists
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df):,} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
