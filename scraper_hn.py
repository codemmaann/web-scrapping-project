import requests
import pandas as pd
from datetime import datetime

def scrape_hn_api():
    top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    story_ids = requests.get(top_stories_url).json()

    rows = []

    for story_id in story_ids[:30]:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        item = requests.get(item_url).json()

        if item is None:
            continue
        
        title = item.get("title", "")
        score = item.get("score", 0)
        comments = item.get("descendants", 0)
        link = item.get("url", f"https://news.ycombinator.com/item?id={story_id}")

        rows.append({
            "title": title,
            "score": score,
            "comments": comments,
            "link": link,
            "timestamp": datetime.utcnow()
        })

    df = pd.DataFrame(rows)
    df.to_csv("raw_data/hn.csv", mode="a", header=False, index=False)

if __name__=="__main__":
    scrape_hn_api()


