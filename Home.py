import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

st.set_page_config(
    page_title="Insurance Reviews Analysis",
    page_icon="📊",
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

st.title('📊 Insurance Reviews Analysis Project')

st.markdown("""
## Project Overview

This project analyzes customer reviews of insurance companies using advanced Natural Language Processing (NLP) techniques. 
The analysis includes sentiment analysis, topic modeling, and various machine learning approaches.

### Dataset Description

The dataset contains customer reviews of various insurance companies and products, including:
- Customer ratings
- Review text (in French and English)
- Insurance company names
- Product types
- Publication dates

### Project Components

1. **Data Cleaning & Visualization** 📈
   - Text preprocessing and cleaning
   - Statistical analysis
   - Interactive visualizations
   - N-gram analysis

2. **Word Embeddings** 🔤
   - Word2Vec implementation
   - Embedding visualization
   - Semantic search capabilities
   - Cosine similarity analysis

3. **Supervised Learning** 🤖
   - TF-IDF with classical ML
   - BERT-based classification
   - Model comparison and evaluation
   - Error analysis

4. **RAG Chatbot** 💬
   - Interactive Q&A system
   - Context-aware responses
   - Review summarization
   - Information retrieval
""")

# Load and display data overview
with st.spinner('Loading data...'):
    df = load_data()

st.header('Quick Data Overview')

# Display key metrics in a nice format using columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Reviews", f"{len(df):,}")

with col2:
    st.metric("Insurance Companies", f"{df['assureur'].nunique():,}")

with col3:
    st.metric("Product Types", f"{df['produit'].nunique():,}")

with col4:
    st.metric("Average Rating", f"{df['note'].mean():.2f}/5")

# Show data sample
st.header('Sample Data')
st.dataframe(df.head())

# Display data statistics
st.header('Dataset Statistics')
col1, col2 = st.columns(2)

with col1:
    st.subheader('Review Length Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    df['text_length'] = df['avis'].str.len()
    sns.histplot(data=df, x='text_length', bins=50)
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Number of Characters')
    st.pyplot(fig)

with col2:
    st.subheader('Rating Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='note', bins=5)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    st.pyplot(fig)

# Data quality information
st.header('Data Quality')
missing_data = df.isnull().sum()
st.write("Missing Values per Column:")
st.write(missing_data[missing_data > 0])

# Instructions for navigation
st.markdown("""
### How to Use This Application

1. Use the sidebar to navigate between different analysis pages
2. Each page provides interactive elements to explore the data
3. You can filter and analyze specific companies or products
4. The RAG chatbot can answer questions about the reviews

### Technical Implementation

- **Language**: Python
- **Framework**: Streamlit
- **Key Libraries**: 
  - NLP: transformers, spacy, gensim
  - ML: scikit-learn, pytorch
  - Visualization: matplotlib, seaborn
  - Data Processing: pandas, numpy
""")