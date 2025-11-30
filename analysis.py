import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
import plotly.io as pio
import shutil
import numpy as np

nltk.download('vader_lexicon')

pio.renderers.default = "png"

class SocialMediaAnalyzer:
    def __init__(self, lobsters_path, hn_path):
        self.lobsters_df = pd.read_csv(lobsters_path)
        self.hn_df = pd.read_csv(hn_path)
        self.combined_df = None
        self.sia = SentimentIntensityAnalyzer()

    def preprocess_data(self):
        lob = self.lobsters_df.copy()

        # Fix column mapping - use the actual column names from the CSV
        rename_map = {
            "score": "upvotes",
            "comments": "comments",  # This column already exists
            "link": "link"  # This column already exists
        }

        lob = lob.rename(columns=rename_map)
        lob['platform'] = 'lobsters'

        # Ensure required columns exist
        for col in ['upvotes', 'comments', 'link']:
            if col not in lob.columns:
                lob[col] = 0

        # Convert post_time to datetime
        if 'post_time' in lob.columns:
            lob['post_time'] = pd.to_datetime(lob['post_time'], errors='coerce')
        else:
            lob['post_time'] = pd.NaT

        hn = self.hn_df.copy()
        hn['platform'] = 'hacker_news'

        if 'post_time' in hn.columns:
            hn['post_time'] = pd.to_datetime(hn['post_time'], errors='coerce')

        if 'author' not in hn.columns:
            hn['author'] = 'unknown'

        common_cols = ['title', 'upvotes', 'comments', 'link', 'post_time', 'platform']
        lob_clean = lob[common_cols]
        hn_clean = hn[common_cols]

        self.combined_df = pd.concat([lob_clean, hn_clean], ignore_index=True)

        # FIX: Ensure post_time is datetime before using .dt accessor
        self.combined_df['post_time'] = pd.to_datetime(self.combined_df['post_time'], errors='coerce')
        
        # Now safely extract datetime components
        self.combined_df['post_hour'] = self.combined_df['post_time'].dt.hour
        self.combined_df['post_day'] = self.combined_df['post_time'].dt.day_name()
        self.combined_df['title_length'] = self.combined_df['title'].astype(str).str.len()
        self.combined_df['word_count'] = self.combined_df['title'].astype(str).str.split().str.len()

        self.combined_df['engagement_score'] = (
            self.combined_df['upvotes'] + self.combined_df['comments'] * 2
        )

        # Add sentiment analysis
        self.combined_df['sentiment'] = self.combined_df['title'].apply(
            lambda x: self.sia.polarity_scores(str(x))['compound']
        )

        return self.combined_df

    def nlp_analysis(self):
        """Perform NLP analysis including topic modeling"""
        # Filter out very short titles
        titles = [str(title) for title in self.combined_df['title'] if len(str(title)) > 10]
        
        if len(titles) == 0:
            return {'topics': {}}
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(titles)
            
            # Topic Modeling with LDA
            lda = LatentDirichletAllocation(
                n_components=5,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(tfidf_matrix)
            
            # Get top words for each topic
            feature_names = vectorizer.get_feature_names_out()
            topics = {}
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics[f'Topic_{topic_idx+1}'] = top_words
                
            return {'topics': topics, 'vectorizer': vectorizer, 'lda_model': lda}
            
        except Exception as e:
            print(f"NLP analysis error: {e}")
            return {'topics': {}}

    def advanced_analysis(self):
        """Perform advanced statistical analysis"""
        # Correlation matrix
        numeric_cols = ['upvotes', 'comments', 'title_length', 'word_count', 'engagement_score', 'sentiment']
        numeric_data = self.combined_df[numeric_cols].select_dtypes(include=[np.number])
        
        correlation_matrix = numeric_data.corr()
        
        # Platform-specific metrics
        platform_metrics = self.combined_df.groupby('platform').agg({
            'upvotes': ['mean', 'median', 'std'],
            'comments': ['mean', 'median', 'std'],
            'engagement_score': ['mean', 'median', 'std'],
            'sentiment': 'mean'
        }).round(2)
        
        return {
            'correlation_matrix': correlation_matrix,
            'platform_metrics': platform_metrics
        }

# ... rest of the code remains the same (create_all_visualizations, ReportGenerator, save_text_report, and main block)

def create_all_visualizations(analyzer, combined_data, nlp_results, advanced_results):
    
    os.makedirs('visualizations', exist_ok=True)
    
    platform_fig = px.box(combined_data, x='platform', y='upvotes', 
                         title='Upvotes Distribution by Platform',
                         color='platform')
    platform_fig.write_image("visualizations/platform_comparison.png", width=1200, height=800)
    
    engagement_fig = px.scatter(combined_data, x='comments', y='upvotes',
                               color='platform', size='engagement_score',
                               title='Engagement Correlation: Comments vs Upvotes',
                               hover_data=['title'])
    engagement_fig.write_image("visualizations/engagement_trends.png", width=1200, height=800)
    
    sentiment_fig = px.histogram(combined_data, x='sentiment', color='platform',
                                title='Sentiment Distribution of Post Titles',
                                barmode='overlay')
    sentiment_fig.write_image("visualizations/sentiment_analysis.png", width=1200, height=800)
    
    hourly_data = combined_data.groupby(['platform', 'post_hour']).agg({
        'engagement_score': 'mean'
    }).reset_index()
    hourly_fig = px.line(hourly_data, x='post_hour', y='engagement_score',
                        color='platform', title='Average Engagement by Hour of Day')
    hourly_fig.write_image("visualizations/hourly_engagement.png", width=1200, height=800)
    
    daily_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    combined_data['post_day'] = pd.Categorical(combined_data['post_day'], categories=daily_order, ordered=True)
    daily_fig = px.box(combined_data, x='post_day', y='engagement_score', 
                      color='platform', title='Engagement by Day of Week')
    daily_fig.write_image("visualizations/daily_patterns.png", width=1200, height=800)
    
    corr_fig = px.imshow(advanced_results['correlation_matrix'],
                        title='Feature Correlation Matrix',
                        color_continuous_scale='RdBu_r',
                        aspect='auto')
    corr_fig.write_image("visualizations/correlation_matrix.png", width=1000, height=800)
    
    topics = nlp_results['topics']
    topic_data = []
    for topic, words in topics.items():
        for word in words:
            topic_data.append({'topic': topic, 'word': word, 'importance': 1})
    
    if topic_data:  # Only create topic visualization if we have data
        topic_df = pd.DataFrame(topic_data)
        topic_fig = px.treemap(topic_df, path=['topic', 'word'],
                              title='Topic Modeling - Key Words Distribution')
        topic_fig.write_image("visualizations/topic_distribution.png", width=1200, height=800)
    
    platform_metrics = combined_data.groupby('platform').agg({
        'upvotes': 'mean',
        'comments': 'mean',
        'engagement_score': 'mean',
        'sentiment': 'mean',
        'title_length': 'mean'
    }).reset_index()
    
    metrics_normalized = platform_metrics.copy()
    for col in ['upvotes', 'comments', 'engagement_score', 'sentiment', 'title_length']:
        metrics_normalized[col] = (platform_metrics[col] - platform_metrics[col].min()) / (platform_metrics[col].max() - platform_metrics[col].min())
    
    radar_fig = go.Figure()
    for platform in metrics_normalized['platform'].unique():
        platform_data = metrics_normalized[metrics_normalized['platform'] == platform]
        values = platform_data[['upvotes', 'comments', 'engagement_score', 'sentiment', 'title_length']].values[0]
        radar_fig.add_trace(go.Scatterpolar(
            r=values,
            theta=['Upvotes', 'Comments', 'Engagement', 'Sentiment', 'Title Length'],
            fill='toself',
            name=platform
        ))
    
    radar_fig.update_layout(
        title='Platform Performance Radar Chart',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    radar_fig.write_image("visualizations/platform_radar.png", width=1000, height=800)
    
    wordcount_fig = px.scatter(combined_data, x='word_count', y='engagement_score',
                              color='platform', trendline='lowess',
                              title='Word Count vs Engagement Score',
                              labels={'word_count': 'Title Word Count', 'engagement_score': 'Engagement Score'})
    wordcount_fig.write_image("visualizations/wordcount_engagement.png", width=1200, height=800)
    
    top_posts = combined_data.nlargest(10, 'engagement_score')[['title', 'platform', 'upvotes', 'comments', 'engagement_score']]
    top_posts_fig = go.Figure(data=[go.Table(
        header=dict(values=['Title', 'Platform', 'Upvotes', 'Comments', 'Engagement Score'],
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[top_posts['title'], top_posts['platform'], top_posts['upvotes'], 
                          top_posts['comments'], top_posts['engagement_score']],
                  fill_color='lavender',
                  align='left'))
    ])
    top_posts_fig.update_layout(title='Top 10 Posts by Engagement Score')
    top_posts_fig.write_image("visualizations/top_posts_table.png", width=1400, height=600)

class ReportGenerator:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.insights = []
    
    def generate_insights(self):
        """Generate data-driven insights"""
        
        platform_stats = self.analyzer.combined_df.groupby('platform').agg({
            'upvotes': 'mean',
            'comments': 'mean'
        })
        
        best_platform = platform_stats['upvotes'].idxmax()
        most_discussed = platform_stats['comments'].idxmax()
        
        self.insights.append(f"**Platform Performance**: {best_platform} has highest average upvotes")
        self.insights.append(f"**Discussion Leader**: {most_discussed} generates most comments")
        
        best_hour = self.analyzer.combined_df.groupby('post_hour')['engagement_score'].mean().idxmax()
        self.insights.append(f"**Optimal Posting Time**: {best_hour}:00 hour gets highest engagement")
        
        sentiment_by_platform = self.analyzer.combined_df.groupby('platform')['sentiment'].mean()
        most_positive = sentiment_by_platform.idxmax()
        self.insights.append(f"**Sentiment Analysis**: {most_positive} has most positive post titles")
        
        title_engagement_corr = self.analyzer.combined_df[['title_length', 'engagement_score']].corr().iloc[0,1]
        self.insights.append(f"**Title Length Impact**: Correlation between title length and engagement: {title_engagement_corr:.2f}")
        
        return self.insights
    
    def generate_summary_stats(self):
        """Generate summary statistics"""
        stats = {
            'total_posts': len(self.analyzer.combined_df),
            'platform_distribution': self.analyzer.combined_df['platform'].value_counts().to_dict(),
            'avg_upvotes': self.analyzer.combined_df['upvotes'].mean(),
            'avg_comments': self.analyzer.combined_df['comments'].mean(),
            'max_upvotes': self.analyzer.combined_df['upvotes'].max(),
            'max_comments': self.analyzer.combined_df['comments'].max(),
            'date_range': {
                'start': self.analyzer.combined_df['post_time'].min(),
                'end': self.analyzer.combined_df['post_time'].max()
            }
        }
        return stats

def save_text_report(insights, summary_stats):

    with open('visualizations/analysis_report.txt', 'w') as f:
        f.write("="*50 + "\n")
        f.write("COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write(" KEY INSIGHTS:\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
        
        f.write(f"\n SUMMARY STATISTICS:\n")
        f.write(f"-Total Posts Analyzed: {summary_stats['total_posts']}\n")
        f.write(f"-Platform Distribution: {summary_stats['platform_distribution']}\n")
        f.write(f"-Average Upvotes: {summary_stats['avg_upvotes']:.1f}\n")
        f.write(f"-Average Comments: {summary_stats['avg_comments']:.1f}\n")
        f.write(f"-Maximum Upvotes: {summary_stats['max_upvotes']}\n")
        f.write(f"-Maximum Comments: {summary_stats['max_comments']}\n")
        f.write(f"-Date Range: {summary_stats['date_range']['start']} to {summary_stats['date_range']['end']}\n")

if __name__ == '__main__':
    
    if os.path.exists("visualizations"):
        shutil.rmtree("visualizations")

    analyzer = SocialMediaAnalyzer('clean_data/lobsters_clean.csv', 'clean_data/hn_cleaned.csv')
    combined_data = analyzer.preprocess_data()
    nlp_results = analyzer.nlp_analysis()
    advanced_results = analyzer.advanced_analysis()

    create_all_visualizations(analyzer, combined_data, nlp_results, advanced_results)

    report_gen = ReportGenerator(analyzer)
    insights = report_gen.generate_insights()
    summary_stats = report_gen.generate_summary_stats()
    save_text_report(insights, summary_stats)