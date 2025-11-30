import time
import os
from datetime import datetime
import pandas as pd
import requests

def scrape_lobsters(tag="hot", max_posts=100):
    time.sleep(2)  

    if tag == "hot":
        url = "https://lobste.rs/hottest.json"
    else:
        url = f"https://lobste.rs/t/{tag}.json"

    params = {
        "page": 1,
        "per_page": max_posts
    }

    headers = {
        "User-Agent": "script:web-scraping-project:1.0 (by /u/codemmaann)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error fetching API data for tag '{tag}': {e}")
        return None

    posts = []

    for item in data:
        created_at = item.get("created_at")
        if created_at:
            post_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        else:
            post_time = datetime.utcnow()

        submitter = item.get("submitter_user", {})
        author = submitter["username"] if isinstance(submitter, dict) else submitter

        posts.append({
            "title": item.get("title", "N/A"),
            "score": item.get("score", 0),
            "comments": item.get("comment_count", 0),
            "link": item.get("url", ""),
            "comments_link": f"https://lobste.rs/s/{item.get('short_id', '')}",
            "author": author,
            "post_time": post_time,
            "scraped_at": datetime.utcnow(),
            "tags": item.get("tags", []),
            "lobsters_id": item.get("short_id", ""),
            "source_tag": tag
        })

    return posts


def save_to_csv(posts, filename="raw_data/lobsters.csv"):
    if not posts:
        print("No posts to save")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(posts)
    file_exists = os.path.isfile(filename)

    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    print(f"Saved {len(posts)} posts to {filename}")


def scrape_multiple_tags(tags, max_posts=100, delay=3):
    all_posts = []

    for tag in tags:
        print(f"Scraping tag '{tag}'...")

        posts = scrape_lobsters(tag, max_posts)

        if posts:
            all_posts.extend(posts)
            save_to_csv(posts, f"raw_data/lobsters_{tag}.csv")

        time.sleep(delay)

    return all_posts


if __name__ == "__main__":
    targets = ["hot"]
    
    all_posts = scrape_multiple_tags(targets, max_posts=100, delay=3)
