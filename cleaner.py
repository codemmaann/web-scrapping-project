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

def clean_lobsters_data():
    dfs = []
    for file in ["raw_data/lobsters_hot.csv"]:
        if os.path.exists(file):
            dfs.append(pd.read_csv(file))

    if not dfs:
        print("No Lobsters data found.")
        return pd.DataFrame()

    lob_df = pd.concat(dfs, ignore_index=True)

    for col in ["post_time", "scraped_at"]:
        if col in lob_df.columns:
            lob_df[col] = pd.to_datetime(lob_df[col].astype(str).str.split('.').str[0], errors='coerce')

    if "title" in lob_df.columns:
        lob_df["title"] = lob_df["title"].astype(str).str.replace('"', "").str.strip()

    if "tags" in lob_df.columns:
        lob_df["tags"] = lob_df["tags"].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else x
        )

    if "lobsters_id" in lob_df.columns:
        lob_df = lob_df.drop_duplicates(subset=["lobsters_id"])
    else:
        lob_df = lob_df.drop_duplicates(subset=["title", "link"])

    lob_df = lob_df.dropna(subset=["title", "link"])

    lob_df["platform"] = "lobsters"

    return lob_df


def main():
    hn_clean = clean_hn_data()
    lobsters_clean = clean_lobsters_data()
    

    folder_path = 'clean_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    hn_clean.to_csv('clean_data/hn_cleaned.csv', index=False)
    lobsters_clean.to_csv('clean_data/lobsters_clean.csv', index=False)
    
    print("Data cleaning complete!")
    print(f"HN: {len(hn_clean)} posts")
    print(f"Reddit: {len(lobsters_clean)} posts")
    
    return hn_clean, lobsters_clean

hn_cleaned, lobsters_clean = main()