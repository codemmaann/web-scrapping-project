import time
import os
import random
from datetime import datetime
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class RedditScraper:
    def __init__(self):
        self.session = self._create_session()
        self.request_count = 0
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0"
        ]
        
    def _create_session(self):
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self):
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    def _random_delay(self, min_delay=4, max_delay=8):
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def scrape_reddit(self, subreddit, max_posts=20):
        
        self._random_delay()
        
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={max_posts}"
        
        try:
            response = self.session.get(
                url, 
                headers=self._get_headers(), 
                timeout=15
            )
            
            if response.status_code == 429:
                time.sleep(60)
                return self.scrape_reddit(subreddit, max_posts)
                
            response.raise_for_status()
            data = response.json()
            
            self.request_count += 1
            
            if self.request_count >= 100:
                time.sleep(300)  
                self.request_count = 0
            
        except requests.exceptions.RequestException as e:
            return None
        except ValueError as e:
            return None
        
        posts = []
        
        for item in data["data"]["children"]:
            post = item["data"]
            
            posts.append({
                "title": post.get("title", "N/A"),
                "upvotes": post.get("ups", 0),
                "comments": post.get("num_comments", 0),
                "link": f"https://www.reddit.com{post.get('permalink', '')}",
                "author": post.get("author", "N/A"),
                "post_time": datetime.utcfromtimestamp(post.get("created_utc", 0)),
                "scraped_at": datetime.utcnow(),
                "subreddit": subreddit,
                "post_id": post.get("id", ""),
                "score": post.get("score", 0),
                "over_18": post.get("over_18", False)
            })
        
        return posts

def save_to_csv(posts, filename="raw_data/reddit.csv"):
    if not posts:
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df = pd.DataFrame(posts)
    file_exists = os.path.isfile(filename)
    
    try:
        df.to_csv(filename, mode='a', header=not file_exists, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Error saving to {filename}: {e}")

def scrape_multiple_subreddits(subreddits, max_posts=15, delay_between_subreddits=5):
    scraper = RedditScraper()
    all_posts = []
    
    random.shuffle(subreddits)
    
    for i, subreddit in enumerate(subreddits):

        
        posts = scraper.scrape_reddit(subreddit, max_posts)
        
        if posts:
            all_posts.extend(posts)
            save_to_csv(posts, f"raw_data/reddit_{subreddit}_{datetime.now().strftime('%Y%m%d')}.csv")
        
        if i < len(subreddits) - 1:
            delay = random.uniform(delay_between_subreddits, delay_between_subreddits + 3)
            print(f"Waiting {delay:.1f} seconds before next subreddit...")
            time.sleep(delay)
    
    return all_posts

def scrape_reddit_pushshift(subreddit, max_posts=20):
    url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size={max_posts}&sort=desc"
    
    headers = {
        "User-Agent": "Research Bot v1.0",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        posts = []
        for post in data["data"]:
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
        
    except Exception as e:
        print(f"Error with Pushshift API for r/{subreddit}: {e}")
        return None

if __name__ == "__main__":
    targets = ["technology", "programming"]
    
    all_posts = scrape_multiple_subreddits(
        subreddits=targets,
        max_posts=15,  
        delay_between_subreddits=10
    )
    
    if all_posts:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_to_csv(all_posts, f"raw_data/reddit_combined_{timestamp}.csv")
        print(f"\nScraped {len(all_posts)} total posts")
        print("\nExample post:")
        print(all_posts[0])