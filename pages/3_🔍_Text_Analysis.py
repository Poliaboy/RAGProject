import glob
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_processing import (
    clean_text, get_sentiment, extract_topics, 
    get_key_phrases, create_word_cloud_data
)

st.set_page_config(
    page_title="Advanced Text Analysis",
    page_icon="🔍",
    layout="wide"
)

def load_data():
    data_path = 'TraductionAvisClients'
    all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    dfs = []
    
    for file in all_files:
        df = pd.read_excel(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

st.title('🔍 Advanced Text Analysis')

# Load data
with st.spinner('Loading data...'):
    df = load_data()

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Sentiment Analysis",
    "Topic Modeling",
    "Key Phrases",
    "Word Cloud"
])

with tab1:
    st.header('Sentiment Analysis')
    
    # Sample selection
    selected_company = st.selectbox(
        "Select Insurance Company",
        options=sorted(df['assureur'].unique())
    )
    
    # Filter reviews for selected company
    company_reviews = df[df['assureur'] == selected_company]
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            # Get sentiment for a sample of reviews
            sample_reviews = company_reviews['avis_en'].head(10)  # Analyze first 10 reviews
            sentiments = [get_sentiment(review) for review in sample_reviews]
            
            # Create sentiment DataFrame
            sentiment_df = pd.DataFrame(sentiments)
            
            # Display results
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=sentiment_df, x='score', bins=20)
            plt.title(f'Sentiment Score Distribution for {selected_company}')
            st.pyplot(fig)
            
            # Display sample reviews with sentiment
            st.subheader("Sample Reviews with Sentiment")
            for review, sentiment in zip(sample_reviews, sentiments):
                st.write(f"**Review:** {review[:200]}...")
                st.write(f"**Sentiment:** {sentiment['label']} (Score: {sentiment['score']:.2f})")
                st.write("---")

with tab2:
    st.header('Topic Modeling')
    
    if st.button("Extract Topics"):
        with st.spinner("Extracting topics..."):
            # Get sample of reviews for topic modeling
            sample_reviews = df['avis_en'].sample(n=1000, random_state=42)
            
            # Extract topics
            topic_model, topics, topic_info = extract_topics(sample_reviews)
            
            # Display results
            st.subheader("Topic Distribution")
            st.dataframe(topic_info)
            
            # Visualize topic distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            topic_counts = pd.Series(topics).value_counts()
            topic_counts.plot(kind='bar')
            plt.title('Topic Distribution in Reviews')
            plt.xlabel('Topic ID')
            plt.ylabel('Number of Reviews')
            plt.xticks(rotation=45)
            st.pyplot(fig)

with tab3:
    st.header('Key Phrase Extraction')
    
    # Text input for analysis
    text_input = st.text_area(
        "Enter text for key phrase extraction",
        value=df['avis_en'].iloc[0],
        height=200
    )
    
    if st.button("Extract Key Phrases"):
        with st.spinner("Extracting key phrases..."):
            # Get key phrases
            phrases = get_key_phrases(text_input)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Noun Phrases")
                for phrase in phrases['noun_phrases']:
                    st.write(f"- {phrase}")
            
            with col2:
                st.subheader("Named Entities")
                for entity in phrases['entities']:
                    st.write(f"- {entity}")

with tab4:
    st.header('Word Cloud')
    
    # Company selection for word cloud
    selected_company = st.selectbox(
        "Select Company for Word Cloud",
        options=sorted(df['assureur'].unique()),
        key="wordcloud_company"
    )
    
    if st.button("Generate Word Cloud"):
        with st.spinner("Generating word cloud..."):
            # Get reviews for selected company
            company_reviews = df[df['assureur'] == selected_company]['avis_en']
            
            # Create word cloud
            word_freq = create_word_cloud_data(company_reviews)
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white'
            ).generate_from_frequencies(word_freq)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.title(f'Word Cloud for {selected_company}')
            st.pyplot(fig) 