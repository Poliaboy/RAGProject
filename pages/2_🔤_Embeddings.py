import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
import sys
import os
import glob

st.set_page_config(
    page_title="Word Embeddings",
    page_icon="🔤",
    layout="wide"
)

def load_data():
    """Load preprocessed data"""
    processed_data_path = 'data/processed_data.pkl'
    if not os.path.exists(processed_data_path):
        st.error("Please run the Data Cleaning & Analysis page first to prepare the data!")
        st.stop()
    return pd.read_pickle(processed_data_path)

st.title('🔤 Word Embeddings Analysis')

# Add explanation of what we're doing
st.markdown("""
### Understanding Word Embeddings
This section explores the semantic relationships between words in our reviews using Word2Vec embeddings:

1. **What are Word Embeddings?**
   - Mathematical representations of words
   - Capture semantic meaning and relationships
   - Convert words into dense vectors
   - Enable similarity comparisons

2. **Why are they useful?**
   - Understand word relationships
   - Find similar words and concepts
   - Capture language patterns
   - Enable semantic search
   - Support machine learning models

3. **What we'll explore:**
   - Train custom Word2Vec models
   - Visualize word relationships
   - Perform semantic search
   - Analyze word clusters

### Language Support
We train separate models for French and English reviews to capture language-specific patterns and meanings.
""")

# Load data
with st.spinner('Loading preprocessed data...'):
    df = load_data()

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs([
    "Word2Vec Training",
    "Embedding Visualization",
    "Semantic Search"
])

with tab1:
    st.header('Word2Vec Model Training')
    st.markdown("""
    Train custom Word2Vec models on our review data:
    - Choose language and text preprocessing
    - Learn word relationships from context
    - Create vector representations
    - Save models for later use
    """)
    
    # Language selection
    language = st.radio("Select Language for Training", ["French", "English"])
    
    # Text version selection
    text_version = st.selectbox(
        "Select Text Version",
        ["Cleaned", "No Stopwords", "Lemmatized", "No Stopwords + Lemmatized"],
        help="Choose which version of the preprocessed text to use for training"
    )
    
    # Map selection to column names
    text_col_map = {
        "Cleaned": "avis_cleaned" if language == "French" else "avis_en_cleaned",
        "No Stopwords": "avis_no_stop" if language == "French" else "avis_en_no_stop",
        "Lemmatized": "avis_lemmatized" if language == "French" else "avis_en_lemmatized",
        "No Stopwords + Lemmatized": "avis_no_stop_and_lemmatized" if language == "French" else "avis_en_no_stop_and_lemmatized"
    }
    
    if st.button(f"Train Word2Vec Model ({language})"):
        with st.spinner(f"Training Word2Vec model for {language}..."):
            # Get selected text column
            text_col = text_col_map[text_version]
            texts = df[text_col].dropna().tolist()
            
            # Tokenize texts
            tokenized_texts = [text.split() for text in texts if isinstance(text, str) and text.strip()]
            
            # Train Word2Vec model
            model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=5, workers=4)
            
            # Save model
            os.makedirs('models', exist_ok=True)
            model_path = f'models/word2vec_{language.lower()}_{text_version.lower().replace(" ", "_")}.model'
            model.save(model_path)
            
            st.success(f"Word2Vec model for {language} trained successfully!")
            
            # Display some example word similarities
            st.subheader("Example Word Similarities")
            example_words = ['insurance', 'customer', 'service', 'price'] if language == "English" else ['assurance', 'client', 'service', 'prix']
            for word in example_words:
                try:
                    similar_words = model.wv.most_similar(word)
                    st.write(f"\nMost similar words to '{word}':")
                    for w, score in similar_words[:5]:
                        st.write(f"- {w}: {score:.4f}")
                except KeyError:
                    st.write(f"Word '{word}' not in vocabulary")

with tab2:
    st.header('Embedding Visualization')
    st.markdown("""
    Visualize word relationships in 2D space:
    - See word clusters
    - Identify related concepts
    - Explore semantic neighborhoods
    - Understand word associations
    """)
    
    # Language selection for visualization
    language = st.radio("Select Language for Visualization", ["French", "English"], key="viz_lang")
    text_version = st.selectbox(
        "Select Text Version",
        ["Cleaned", "No Stopwords", "Lemmatized", "No Stopwords + Lemmatized"],
        key="viz_version",
        help="Choose which version of the preprocessed text to use for visualization"
    )
    
    # Visualization options
    col1, col2 = st.columns(2)
    with col1:
        max_words = st.slider("Number of words to visualize", 100, 2000, 500)
        perplexity = st.slider("t-SNE perplexity", 5, 50, 30)
    with col2:
        point_size = st.slider("Point size", 1, 20, 10)
        opacity = st.slider("Point opacity", 0.1, 1.0, 0.7)
    
    model_path = f'models/word2vec_{language.lower()}_{text_version.lower().replace(" ", "_")}.model'
    
    try:
        # Load pre-trained model
        model = Word2Vec.load(model_path)
        
        # Get word vectors
        words = list(model.wv.index_to_key)[:max_words]  # Limit words based on slider
        vectors = [model.wv[word] for word in words]
        
        # Convert vectors to numpy array
        vectors_array = np.array(vectors)
        
        if len(vectors) > 0:
            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(vectors)-1))
            vectors_2d = tsne.fit_transform(vectors_array)
            
            # Create DataFrame for Plotly
            df_plot = pd.DataFrame({
                'x': vectors_2d[:, 0],
                'y': vectors_2d[:, 1],
                'word': words,
                'freq': [model.wv.get_vecattr(word, "count") for word in words]  # Get word frequencies
            })
            
            # Create interactive plot
            fig = px.scatter(
                df_plot, 
                x='x', 
                y='y',
                text='word',
                size='freq',  # Size points by word frequency
                hover_data={'x': False, 'y': False, 'freq': True},  # Show frequency on hover
                title=f'Interactive Word Embeddings Visualization ({language} - {text_version})'
            )
            
            # Update layout and traces
            fig.update_traces(
                textposition='top center',
                marker=dict(size=point_size, opacity=opacity),
                textfont=dict(size=10)
            )
            fig.update_layout(
                showlegend=False,
                width=800,
                height=600
            )
            
            # Add custom hover template
            fig.update_traces(
                hovertemplate="<b>Word:</b> %{text}<br>" +
                             "<b>Frequency:</b> %{customdata[0]}<br>" +
                             "<extra></extra>"  # This removes the secondary box
            )
            
            # Display the interactive plot
            st.plotly_chart(fig, use_container_width=True)
            
        
            
            # Add clustering analysis
            st.subheader("Word Clusters")
            cluster_words = st.multiselect(
                "Select words to find similar clusters:",
                words,
                max_selections=5
            )
            
            if cluster_words:
                for word in cluster_words:
                    similar_words = model.wv.most_similar(word, topn=10)
                    st.write(f"\nCluster around '{word}':")
                    for similar_word, similarity in similar_words:
                        st.write(f"- {similar_word}: {similarity:.4f}")
            
        else:
            st.warning("No vectors available for visualization. The model vocabulary might be empty.")
        
    except FileNotFoundError:
        st.warning(f"Please train the Word2Vec model for {language} ({text_version}) first!")
    except Exception as e:
        st.error(f"Error during visualization: {str(e)}")
        st.write("Try training the model with more data or different parameters.")

with tab3:
    st.header('Semantic Search')
    st.markdown("""
    Find semantically similar words:
    - Search by meaning, not just spelling
    - Discover related terms
    - Compare word similarities
    - Explore language patterns
    """)
    
    # Language selection for search
    language = st.radio("Select Language for Search", ["French", "English"], key="search_lang")
    text_version = st.selectbox(
        "Select Text Version",
        ["Cleaned", "No Stopwords", "Lemmatized", "No Stopwords + Lemmatized"],
        key="search_version",
        help="Choose which version of the preprocessed text to use for search"
    )
    
    model_path = f'models/word2vec_{language.lower()}_{text_version.lower().replace(" ", "_")}.model'
    
    query = st.text_input("Enter a word to find similar words:")
    
    if query and st.button("Search"):
        try:
            model = Word2Vec.load(model_path)
            
            # Find similar words
            similar_words = model.wv.most_similar(query)
            
            # Display results
            st.subheader("Similar Words")
            for word, score in similar_words:
                st.write(f"- {word}: {score:.4f}")
                
            # Calculate and display cosine similarity
            st.subheader("Cosine Similarity")
            query_vector = model.wv[query]
            
            def cosine_similarity(v1, v2):
                return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            for word, _ in similar_words[:5]:
                word_vector = model.wv[word]
                similarity = cosine_similarity(query_vector, word_vector)
                st.write(f"Cosine similarity between '{query}' and '{word}': {similarity:.4f}")
                
        except FileNotFoundError:
            st.warning(f"Please train the Word2Vec model for {language} ({text_version}) first!")
        except KeyError:
            st.error(f"Word '{query}' not found in vocabulary. Try another word.") 