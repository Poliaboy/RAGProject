import re
import numpy as np
from textblob import TextBlob
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import spacy

def clean_text(text):
    """Clean text by removing special characters and standardizing format"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment(text, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """Get sentiment using a pre-trained model"""
    classifier = pipeline("sentiment-analysis", model=model_name)
    result = classifier(text[:512])[0]  # Truncate to max length
    return {
        'label': result['label'],
        'score': result['score']
    }

def extract_topics(texts, n_topics=10):
    """Extract topics using BERTopic"""
    # Initialize BERTopic
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(texts)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    return topic_model, topics, topic_info

def get_key_phrases(text, lang='en'):
    """Extract key phrases using spaCy"""
    nlp = spacy.load(f'{lang}_core_web_sm')
    doc = nlp(text)
    
    # Extract noun phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    # Extract named entities
    entities = [ent.text for ent in doc.ents]
    
    return {
        'noun_phrases': noun_phrases,
        'entities': entities
    }

def create_word_cloud_data(texts):
    """Create word frequency data for word cloud"""
    # Combine all texts
    text = ' '.join(texts)
    
    # Create word frequency dictionary
    words = text.split()
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    return word_freq 