from googleapiclient.discovery import build
import pandas as pd
import time

API_KEY = "AIzaSyAVusncGB9Sj7aSmExjj1QElrMWq6f91VU"

def fetch_youtube_videos(query, max_results=50):
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=min(50, max_results)
    )

    response = request.execute()

    items = []
    for item in response['items']:
        vid = item['id']['videoId']
        snippet = item['snippet']
        items.append({
            'video_id': vid,
            'title': snippet['title'],
            'description': snippet.get('description', ''),
            'channel': snippet['channelTitle'],
            'published_at': snippet['publishedAt']
        })

    return pd.DataFrame(items)

if __name__ == "__main__":
    df = fetch_youtube_videos("machine learning tutorial", max_results=50)
    df.to_csv("../data/interim/videos_metadata.csv", index=False)
