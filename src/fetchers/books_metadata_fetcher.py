import requests
import pandas as pd
import time
import os

def fetch_books_metadata(keywords, max_results_per_keyword=5, delay=1):
    base_url = "https://www.googleapis.com/books/v1/volumes"
    all_books = []

    for keyword in keywords:
        params = {
            "q": keyword,
            "maxResults": max_results_per_keyword
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            for item in data.get("items", []):
                info = item.get("volumeInfo", {})
                all_books.append({
                    "keyword": keyword,
                    "title": info.get("title"),
                    "authors": ", ".join(info.get("authors", [])),
                    "description": info.get("description"),
                    "categories": ", ".join(info.get("categories", [])),
                    "publishedDate": info.get("publishedDate"),
                    "pageCount": info.get("pageCount"),
                    "language": info.get("language"),
                    "averageRating": info.get("averageRating"),
                    "ratingsCount": info.get("ratingsCount"),
                    "previewLink": info.get("previewLink"),
                    "infoLink": info.get("infoLink")
                })

        except Exception as e:
            print(f"Error fetching books for keyword '{keyword}':", e)

        time.sleep(delay)  # Avoid getting rate-limited

    return pd.DataFrame(all_books)

keywords = ["machine learning", "data science", "natural language processing", "python programming"]
df_books = fetch_books_metadata(keywords)

df_books.to_csv("../data/interim/books_metadata.csv")

