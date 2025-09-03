# books_metadata_fetcher.py
import os
import time
import math
import requests
import pandas as pd
from typing import List, Optional


GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
PAGE_SIZE = 40  # Google Books maxResults cap

def fetch_books_metadata(
    keywords: List[str],
    per_keyword: int = 200,              # total desired per keyword (multiple pages)
    api_key: Optional[str] = None,       # optional; higher quota if set
    delay: float = 0.5,                  # pause between requests
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch book metadata from Google Books for each keyword, with robust pagination & retries.
    Returns a pandas DataFrame of all results (deduped by volumeId).
    """

    session = requests.Session()
    all_rows = []
    seen_ids = set()

    for kw in keywords:
        pages = math.ceil(per_keyword / PAGE_SIZE)
        fetched = 0

        for p in range(pages):
            start_index = p * PAGE_SIZE
            # Stop early if we already have enough for this keyword
            if fetched >= per_keyword:
                break

            params = {
                "q": kw,
                "startIndex": start_index,
                "maxResults": min(PAGE_SIZE, per_keyword - fetched),
                # Tip: you could add 'printType=books' to avoid magazines, etc.
                # "printType": "books",
                # Use 'fields' to cut payload size:
                "fields": "items(id,volumeInfo(title,authors,description,categories,publishedDate,"
                          "pageCount,language,averageRating,ratingsCount,previewLink,infoLink,industryIdentifiers))"
            }
            if api_key:
                params["key"] = api_key

            # Simple retry with exponential backoff
            for attempt in range(1, max_retries + 1):
                try:
                    resp = session.get(GOOGLE_BOOKS_URL, params=params, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("items", [])
                    if not items:
                        # No more results for this keyword
                        break

                    for item in items:
                        vol_id = item.get("id")
                        if vol_id in seen_ids:
                            continue
                        seen_ids.add(vol_id)

                        info = item.get("volumeInfo", {}) or {}
                        inds = info.get("industryIdentifiers", []) or []
                        # Try to pick ISBN-13 / ISBN-10 if available
                        isbn13 = next((x.get("identifier") for x in inds if x.get("type") == "ISBN_13"), None)
                        isbn10 = next((x.get("identifier") for x in inds if x.get("type") == "ISBN_10"), None)

                        all_rows.append({
                            "keyword": kw,
                            "volume_id": vol_id,
                            "title": info.get("title") or "",
                            "authors": ", ".join(info.get("authors", [])),
                            "description": info.get("description") or "",
                            "categories": ", ".join(info.get("categories", [])),
                            "publishedDate": info.get("publishedDate") or "",
                            "pageCount": info.get("pageCount"),
                            "language": info.get("language") or "",
                            "averageRating": info.get("averageRating"),
                            "ratingsCount": info.get("ratingsCount"),
                            "previewLink": info.get("previewLink") or "",
                            "infoLink": info.get("infoLink") or "",
                            "isbn13": isbn13 or "",
                            "isbn10": isbn10 or "",
                        })
                    fetched += len(items)
                    time.sleep(delay)
                    break  # success: leave retry loop
                except requests.HTTPError as e:
                    # If quota/rate or 5xx, backoff; otherwise re-raise
                    if resp is not None and resp.status_code in (429, 500, 502, 503, 504):
                        if attempt == max_retries:
                            print(f"[{kw}] Reached max retries at startIndex={start_index}: {e}")
                            break
                        sleep_s = delay * (2 ** (attempt - 1))
                        print(f"[{kw}] HTTP {resp.status_code}. Retrying in {sleep_s:.1f}s...")
                        time.sleep(sleep_s)
                        continue
                    else:
                        print(f"[{kw}] HTTP error: {e}")
                        break
                except requests.RequestException as e:
                    if attempt == max_retries:
                        print(f"[{kw}] Network error at startIndex={start_index}: {e}")
                        break
                    sleep_s = delay * (2 ** (attempt - 1))
                    print(f"[{kw}] Network issue. Retrying in {sleep_s:.1f}s...")
                    time.sleep(sleep_s)
                    continue

    df = pd.DataFrame(all_rows)
    # Optional: sort columns for consistency
    preferred_order = [
        "keyword","volume_id","title","authors","description","categories","publishedDate",
        "pageCount","language","averageRating","ratingsCount","previewLink","infoLink","isbn13","isbn10"
    ]
    df = df.reindex(columns=preferred_order)
    return df


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()

    keywords = [
        "machine learning", "data science", "natural language processing",
        "python programming", "deep learning", "neural networks", "AI ethics",
        "computer vision", "data visualization", "statistics for data science",
        "web development", "cybersecurity", "cloud computing", "docker", "sql",
        "algorithms", "software design", "java programming"
    ]

    # API key from .env
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")

    # Resolve paths from env
    project_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    out_dir = Path(os.getenv("BOOKS_FETCH_OUT", project_root / "backend/data/interim_large"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "books_metadata.csv"

    df_books = fetch_books_metadata(
        keywords=keywords,
        per_keyword=int(os.getenv("BOOKS_FETCH_PER_KEYWORD", 200)),  # can override in .env
        api_key=api_key,
        delay=float(os.getenv("BOOKS_FETCH_DELAY", 0.5)),
        max_retries=int(os.getenv("BOOKS_FETCH_MAX_RETRIES", 3)),
    )

    df_books.to_csv(out_path, index=False)
    print(f"Saved {len(df_books)} rows → {out_path}")
