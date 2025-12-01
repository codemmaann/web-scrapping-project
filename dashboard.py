import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Tech Community Analysis Dashboard",
    layout="wide",
)

st.title("Tech Community Insights Dashboard")
st.write("Interactive analysis of Hacker News and Lobsters posts with NLP-based search capabilities.")

@st.cache_data
def load_data():
    """Load and combine Hacker News and Lobsters data"""
    try:
        hn_df = pd.read_csv("clean_data/hn_cleaned.csv")
        lobsters_df = pd.read_csv("clean_data/lobsters_clean.csv")
        
        lobsters_df = lobsters_df.rename(columns={
            'score': 'upvotes'
        })
        
        if 'post_time' not in lobsters_df.columns:
            lobsters_df['post_time'] = pd.to_datetime(lobsters_df.get('scraped_at', pd.NaT))
        
        common_cols = ['title', 'upvotes', 'comments', 'link', 'post_time', 'platform']
        
        hn_df = hn_df[common_cols]
        lobsters_df = lobsters_df[[c for c in common_cols if c in lobsters_df.columns]]
        
        df = pd.concat([hn_df, lobsters_df], ignore_index=True)
        
        df['post_time'] = pd.to_datetime(df['post_time'], errors='coerce')
        df = df.dropna(subset=['title', 'upvotes', 'comments'])
        
        df['upvotes'] = pd.to_numeric(df['upvotes'], errors='coerce')
        df['comments'] = pd.to_numeric(df['comments'], errors='coerce')
        
        df['engagement_score'] = df['upvotes'] + (df['comments'] * 2)
        df['title_length'] = df['title'].astype(str).str.len()
        df['word_count'] = df['title'].astype(str).str.split().str.len()
        
        df['post_hour'] = df['post_time'].dt.hour
        df['post_day'] = df['post_time'].dt.day_name()
        df['post_date'] = df['post_time'].dt.date
        df = df.dropna(subset=['post_date'])
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data available. Please check if the CSV files exist in the correct location.")
    st.stop()

st.sidebar.header("Filters")

platforms = ["All"] + sorted(df['platform'].dropna().unique().tolist())
selected_platform = st.sidebar.selectbox(
    "Platform",
    options=platforms,
    index=0,
)

if 'post_date' in df.columns:
    min_date = df['post_date'].min()
    max_date = df['post_date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
else:
    start_date, end_date = None, None

col1, col2 = st.sidebar.columns(2)

with col1:
    min_upvotes = st.number_input(
        "Min Upvotes",
        min_value=0,
        max_value=int(df['upvotes'].max()),
        value=0,
        step=10
    )

with col2:
    min_comments = st.number_input(
        "Min Comments",
        min_value=0,
        max_value=int(df['comments'].max()),
        value=0,
        step=5
    )

keyword = st.sidebar.text_input("Keyword in Title").strip().lower()

st.sidebar.subheader("Time Filters")
hour_range = st.sidebar.slider(
    "Posting Hour (24h)",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

base_count = len(df)

if selected_platform != "All":
    filtered_df = df[df['platform'] == selected_platform]
else:
    filtered_df = df.copy()
platform_count = len(filtered_df)

if start_date and end_date:
    filtered_df = filtered_df[
        (filtered_df['post_date'] >= start_date) & 
        (filtered_df['post_date'] <= end_date)
    ]
date_count = len(filtered_df)

filtered_df = filtered_df[
    (filtered_df['upvotes'] >= min_upvotes) &
    (filtered_df['comments'] >= min_comments) &
    (filtered_df['post_hour'] >= hour_range[0]) &
    (filtered_df['post_hour'] <= hour_range[1])
]
engagement_count = len(filtered_df)

if keyword:
    filtered_df = filtered_df[filtered_df['title'].str.lower().str.contains(keyword, na=False)]
final_count = len(filtered_df)

st.sidebar.markdown("---")
st.sidebar.markdown("Filter Results:")
st.sidebar.write(f"Total posts: {base_count}")
st.sidebar.write(f"After platform: {platform_count}")
if start_date and end_date:
    st.sidebar.write(f"After date: {date_count}")
st.sidebar.write(f"After engagement: {engagement_count}")
st.sidebar.write(f"After keyword: {final_count}")

if final_count == 0:
    st.warning("No posts match the current filters. Try relaxing your criteria.")


st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_upvotes = filtered_df['upvotes'].mean() if final_count > 0 else 0
    st.metric(
        "Avg Upvotes",
        f"{avg_upvotes:.1f}",
        delta=None
    )

with col2:
    avg_comments = filtered_df['comments'].mean() if final_count > 0 else 0
    st.metric(
        "Avg Comments",
        f"{avg_comments:.1f}",
        delta=None
    )

with col3:
    total_engagement = filtered_df['engagement_score'].sum() if final_count > 0 else 0
    st.metric(
        "Total Engagement",
        f"{total_engagement:,.0f}",
        delta=None
    )

with col4:
    avg_title_len = filtered_df['title_length'].mean() if final_count > 0 else 0
    st.metric(
        "Avg Title Length",
        f"{avg_title_len:.0f} chars",
        delta=None
    )

st.header("Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Engagement Trends", "Time Analysis", "Platform Comparison", "Top Posts"])

with tab1:
    if final_count > 0:
        fig = px.scatter(
            filtered_df,
            x='comments',
            y='upvotes',
            color='platform',
            size='engagement_score',
            hover_data=['title'],
            title='Engagement Correlation: Comments vs Upvotes',
            labels={'comments': 'Number of Comments', 'upvotes': 'Number of Upvotes'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for visualization")

with tab2:
    if final_count > 0:

        hourly_data = filtered_df.groupby(['platform', 'post_hour']).agg({
            'engagement_score': 'mean'
        }).reset_index()
        
        fig = px.line(
            hourly_data,
            x='post_hour',
            y='engagement_score',
            color='platform',
            title='Average Engagement by Hour of Day',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        daily_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        filtered_df['post_day'] = pd.Categorical(filtered_df['post_day'], categories=daily_order, ordered=True)
        
        fig2 = px.box(
            filtered_df,
            x='post_day',
            y='engagement_score',
            color='platform',
            title='Engagement by Day of Week'
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if final_count > 0:
        col1, col2 = st.columns(2)
        
        with col1:

            platform_counts = filtered_df['platform'].value_counts()
            fig = px.pie(
                values=platform_counts.values,
                names=platform_counts.index,
                title='Post Distribution by Platform'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:

            platform_stats = filtered_df.groupby('platform').agg({
                'upvotes': 'mean',
                'comments': 'mean',
                'engagement_score': 'mean'
            }).reset_index()
            
            fig = px.bar(
                platform_stats,
                x='platform',
                y=['upvotes', 'comments', 'engagement_score'],
                title='Platform Performance Metrics',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    if final_count > 0:

        top_n = st.slider("Number of top posts to show", 5, 50, 10)
        
        top_posts = filtered_df.nlargest(top_n, 'engagement_score')[
            ['title', 'platform', 'upvotes', 'comments', 'engagement_score', 'post_time']
        ]
        
        top_posts_display = top_posts.copy()
        top_posts_display['post_time'] = top_posts_display['post_time'].dt.strftime('%Y-%m-%d %H:%M')
        top_posts_display = top_posts_display.rename(columns={
            'title': 'Title',
            'platform': 'Platform',
            'upvotes': 'Upvotes',
            'comments': 'Comments',
            'engagement_score': 'Engagement Score',
            'post_time': 'Post Time'
        })
        
        st.dataframe(
            top_posts_display,
            use_container_width=True,
            height=400
        )

st.header("NLP Semantic Search")

@st.cache_data
def build_tfidf_for_subset(titles: pd.Series):
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_sub = vec.fit_transform(titles.astype(str).tolist())
    return vec, X_sub

search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    user_query = st.text_input(
        "Search for topics or technologies",
        placeholder="e.g., 'Linux security', 'AI programming', 'Rust vs C++'"
    )

with search_col2:
    top_k = st.number_input(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

run_search = st.button("Search", type="primary")

if run_search:
    if not user_query:
        st.warning("Please enter a search query")
    elif final_count == 0:
        st.warning("No posts available to search within current filters")
    else:
        with st.spinner("Finding relevant posts..."):
            subset_vec, X_sub = build_tfidf_for_subset(filtered_df["title"])
            query_vec = subset_vec.transform([user_query])
            
            sims = cosine_similarity(query_vec, X_sub).flatten()
            k = min(int(top_k), len(filtered_df))
            
            if k <= 0:
                st.warning("No matches found")
            else:
                top_idx_local = np.argsort(sims)[-k:][::-1]
                results = filtered_df.iloc[top_idx_local][
                    ['title', 'platform', 'upvotes', 'comments', 'engagement_score', 'link']
                ].copy().reset_index(drop=True)
                
                results['similarity'] = sims[top_idx_local]
                
                results_display = results.rename(columns={
                    'title': 'Title',
                    'platform': 'Platform',
                    'upvotes': 'Upvotes',
                    'comments': 'Comments',
                    'engagement_score': 'Engagement',
                    'link': 'Link',
                    'similarity': 'Relevance Score'
                })
                
                st.subheader(f"Top {len(results)} Most Relevant Posts")
                
                for idx, row in results.iterrows():
                    with st.expander(f"{row['title']} (Relevance: {sims[top_idx_local][idx]:.2f})"):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"Platform: {row['platform']}")
                            st.write(f"Upvotes: {row['upvotes']} | Comments: {row['comments']}")
                            st.write(f"Engagement Score: {row['engagement_score']:.0f}")
                        
                        with col2:
                            if pd.notna(row['link']) and row['link']:
                                st.markdown(f"[View Original]({row['link']})")
                        
                        with col3:
                            similarity_score = sims[top_idx_local][idx]
                            if similarity_score > 0.3:
                                st.success(f"High Relevance ({similarity_score:.2f})")
                            elif similarity_score > 0.1:
                                st.info(f"Moderate Relevance ({similarity_score:.2f})")
                            else:
                                st.warning(f"Low Relevance ({similarity_score:.2f})")

with st.expander("Data Summary", expanded=False):
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Current Filtered Data:")
        st.dataframe(
            filtered_df[['title', 'platform', 'upvotes', 'comments', 'engagement_score']].head(10),
            use_container_width=True
        )
    
    with col2:
        st.write("Statistics:")
        if final_count > 0:
            stats_df = filtered_df[['upvotes', 'comments', 'engagement_score']].describe().round(2)
            st.dataframe(stats_df, use_container_width=True)
    
    st.write("---")
    if final_count > 0:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"tech_posts_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <i>Tech Community Insights Dashboard • Data from Hacker News & Lobsters • Updated automatically</i>
    </div>
    """,
    unsafe_allow_html=True
)