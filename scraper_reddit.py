import time
import os
from datetime import datetime
import pandas as pd
import requests

def scrape_reddit(subreddit, max_posts=20):
    time.sleep(1.5)  # avoid API rate-limit

    url = f"https://api.reddit.com/r/{subreddit}/hot?limit={max_posts}"

    headers = {
        "User-Agent": "script:web-scraping-project:1.0 (by /u/codemmaann)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error fetching API data for r/{subreddit}: {e}")
        return None

    posts = []

    for item in data["data"]["children"]:
        post = item["data"]

        posts.append({
            "title": post.get("title", "N/A"),
            "upvotes": post.get("ups", 0),
            "comments": post.get("num_comments", 0),
            "link": "https://www.reddit.com" + post.get("permalink", ""),
            "author": post.get("author", "N/A"),
            "post_time": datetime.utcfromtimestamp(post.get("created_utc", 0)),
            "scraped_at": datetime.utcnow(),
            "subreddit": subreddit
        })

    return posts


def save_to_csv(posts, filename="raw_data/reddit.csv"):
    if not posts:
        print("No posts to save")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(posts)
    file_exists = os.path.isfile(filename)

    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    print(f"Saved {len(posts)} posts to {filename}")



def scrape_multiple_subreddits(subreddits, max_posts=15, delay=5):
    all_posts = []

    for subreddit in subreddits:
        print(f"Scraping r/{subreddit}...")

        posts = scrape_reddit(subreddit, max_posts)

        if posts:
            all_posts.extend(posts)
            save_to_csv(posts, f"raw_data/reddit_{subreddit}.csv")

        time.sleep(delay)

    return all_posts


if __name__=="__main__":
    targets = ["technology", "programming"]

    targets = ["technology", "programming"]

    all_posts = scrape_multiple_subreddits(targets)

    if all_posts:
        save_to_csv(all_posts)
        print("\nExample post:")
        print(all_posts[0])