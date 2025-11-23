import pandas as pd
import os

def clean_hn_data():
    hn_df = pd.read_csv('raw_data/hn.csv', names=['title', 'upvotes', 'comments', 'link', 'post_time'])
    hn_df['post_time'] = hn_df['post_time'].str.split('.').str[0]
    hn_df['post_time'] = pd.to_datetime(hn_df['post_time'])
    hn_df['title'] = hn_df['title'].str.replace('"', '')
    hn_df = hn_df.drop_duplicates(subset=['title', 'link'])
    hn_df = hn_df.dropna(subset=['title', 'link'])
    hn_df['platform'] = 'hacker_news'
    return hn_df

def clean_reddit_data():
    reddit_df = pd.read_csv('raw_data/reddit.csv')
    reddit_prog_df = pd.read_csv('raw_data/reddit_programming.csv')
    reddit_tech_df = pd.read_csv('raw_data/reddit_technology.csv')
    
    all_reddit = pd.concat([reddit_df, reddit_prog_df, reddit_tech_df], ignore_index=True)
    all_reddit = all_reddit.drop_duplicates(subset=['title', 'link'])
    all_reddit['post_time'] = pd.to_datetime(all_reddit['post_time'])
    all_reddit['scraped_at'] = pd.to_datetime(all_reddit['scraped_at'])
    all_reddit['title'] = all_reddit['title'].str.replace('"', '')
    all_reddit = all_reddit.dropna(subset=['title', 'link', 'subreddit'])
    all_reddit['platform'] = 'reddit'
    
    return all_reddit

def main():
    hn_clean = clean_hn_data()
    reddit_clean = clean_reddit_data()
    

    folder_path = 'clean_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    hn_clean.to_csv('clean_data/hn_cleaned.csv', index=False)
    reddit_clean.to_csv('clean_data/reddit_cleaned.csv', index=False)
    
    print("Data cleaning complete!")
    print(f"HN: {len(hn_clean)} posts")
    print(f"Reddit: {len(reddit_clean)} posts")
    
    return hn_clean, reddit_clean

hn_cleaned, reddit_cleaned = main()