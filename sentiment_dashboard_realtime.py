import streamlit as st
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# === RSS FEEDS ===
RSS_FEEDS = {
    "Google News (Nifty/Sensex)": "https://news.google.com/rss/search?q=Nifty+OR+Sensex&hl=en-IN&gl=IN&ceid=IN:en",
    "Moneycontrol Market News": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
}

def fetch_rss_articles(rss_sources, max_per_feed=20):
    articles = []
    for source_name, url in rss_sources.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_per_feed]:
            published = entry.get("published", "")
            try:
                pub_date = datetime.strptime(published[:25], "%a, %d %b %Y %H:%M:%S")
            except:
                pub_date = None
            articles.append({
                "Source": source_name,
                "Title": entry.title,
                "Link": entry.link,
                "Published": pub_date,
            })
    return articles

def analyze_sentiment(news_items):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for item in news_items:
        score = analyzer.polarity_scores(item["Title"])["compound"]
        sentiment = "Bullish" if score >= 0.3 else "Bearish" if score <= -0.3 else "Neutral"
        results.append({**item, "Score": score, "Sentiment": sentiment})
    return results

def plot_wordcloud(titles, sentiment):
    text = " ".join(titles)
    if not text.strip():
        st.info(f"No {sentiment} headlines to display word cloud.")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.subheader(f"☁️ Word Cloud - {sentiment} Headlines")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# === Streamlit App ===
st.set_page_config(page_title="📊 Real-Time Indian Market Sentiment", layout="wide")
st.title("📊 Indian Market Sentiment Dashboard (Real-Time Auto Refresh)")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    selected_feeds = st.multiselect("Select News Sources", list(RSS_FEEDS.keys()), default=list(RSS_FEEDS.keys()))
    max_per_feed = st.slider("Articles per Feed", 5, 50, 20)
    enable_refresh = st.checkbox("Enable Auto-Refresh (every 15 min)", value=True)

# Auto-refresh every 15 minutes if enabled
if enable_refresh:
    st_autorefresh(interval=15 * 60 * 1000, key="auto_refresh")
    st.caption("⏱️ Auto-refresh enabled: updates every 15 minutes")

if st.button("🔍 Analyze Now") or enable_refresh:
    with st.spinner("Fetching and analyzing news articles..."):
        selected_sources = {k: RSS_FEEDS[k] for k in selected_feeds}
        articles = fetch_rss_articles(selected_sources, max_per_feed)
        scored = analyze_sentiment(articles)
        df = pd.DataFrame(scored)

        # Process dates
        df["Published"] = pd.to_datetime(df["Published"])
        df["Published Date"] = df["Published"].dt.date

        # Summary
        avg_score = round(df["Score"].mean(), 3) if not df.empty else 0.0
        signal = "📈 Bullish" if avg_score >= 0.3 else "📉 Bearish" if avg_score <= -0.3 else "⚖️ Neutral"

        st.subheader("📈 Summary")
        st.metric("Average Sentiment Score", avg_score)
        st.success(f"Market Signal: {signal}")
        st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Word Clouds
        col1, col2 = st.columns(2)
        with col1:
            bull_titles = df[df["Sentiment"] == "Bullish"]["Title"].tolist()
            plot_wordcloud(bull_titles, "Bullish")

        with col2:
            bear_titles = df[df["Sentiment"] == "Bearish"]["Title"].tolist()
            plot_wordcloud(bear_titles, "Bearish")

        # Sentiment Score Chart
        st.subheader("📉 Sentiment Score Chart")
        sentiment_filter = st.multiselect("Filter by Sentiment", options=df["Sentiment"].unique(), default=df["Sentiment"].unique())
        filtered = df[df["Sentiment"].isin(sentiment_filter)]

        fig, ax = plt.subplots()
        ax.plot(filtered["Score"].values, marker='o')
        ax.axhline(0.3, color='green', linestyle='--', label='Bullish Threshold')
        ax.axhline(-0.3, color='red', linestyle='--', label='Bearish Threshold')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title("Sentiment Scores by Article")
        ax.set_ylabel("Compound Score")
        ax.legend()
        st.pyplot(fig)

        # Sentiment grouped by date
        st.subheader("📅 Sentiment Grouped by Date")
        grouped = df.groupby(["Published Date", "Sentiment"]).size().unstack().fillna(0)
        st.bar_chart(grouped)

        # Sentiment grouped by source
        st.subheader("📊 Sentiment Grouped by Source")
        by_source = df.groupby(["Source", "Sentiment"]).size().unstack().fillna(0)
        st.bar_chart(by_source)

        # Show article table with clickable links
        st.subheader("📰 Articles")
        df["Link"] = df["Link"].apply(lambda url: f'<a href="{url}" target="_blank">🔗</a>')
        table_df = df[["Published", "Source", "Sentiment", "Score", "Title", "Link"]]
        st.write(table_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # CSV Download
        st.download_button("📥 Download CSV", data=df.drop(columns=["Link"]).to_csv(index=False), file_name="sentiment_report.csv")

