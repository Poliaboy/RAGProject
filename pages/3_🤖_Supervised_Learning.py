import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import joblib
import json
import os

# Custom Dataset for text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx].split()
        # Pad or truncate text to max_len
        if len(text) < self.max_len:
            text = text + ['<pad>'] * (self.max_len - len(text))
        else:
            text = text[:self.max_len]
        # Convert words to indices
        text = [self.vocab.get(word, self.vocab['<unk>']) for word in text]
        return torch.tensor(text, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Custom RNN model
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        return self.fc(hidden)

# Custom CNN model
class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, 
                     kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# FastText-inspired model
class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        
        # FastText-style averaging of word embeddings
        pooled = torch.mean(embedded, dim=1)  # [batch_size, embedding_dim]
        
        # Feed through fully connected layers
        hidden = self.dropout(F.relu(self.fc1(pooled)))
        output = self.fc2(hidden)
        
        return output

st.set_page_config(
    page_title="Supervised Learning",
    page_icon="🤖",
    layout="wide"
)

st.title('🤖 Supervised Learning Models')

# Add explanation of what we're doing
st.markdown("""
### What are we doing here?
This section implements sentiment analysis on customer reviews using four different machine learning approaches:

1. **TF-IDF + Logistic Regression**: A traditional machine learning approach that:
   - Converts text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
   - Uses logistic regression to classify reviews as negative (1-2 stars) or positive (4-5 stars)
   - Fast to train and interpretable, but doesn't capture word order or context

2. **RNN (Recurrent Neural Network)**: A deep learning approach that:
   - Uses word embeddings to represent words as dense vectors
   - Processes text sequentially, capturing word order and context
   - Better at understanding long-term dependencies in text

3. **CNN (Convolutional Neural Network)**: A deep learning approach that:
   - Also uses word embeddings
   - Applies convolution operations to detect local patterns and features in text
   - Good at capturing local patterns and key phrases

4. **FastText-inspired Model**: A lightweight but effective approach that:
   - Uses word embeddings like the other neural models
   - Averages word embeddings to create document representations
   - Fast to train and good at handling rare words
   - Inspired by Facebook's FastText architecture

### Model Comparison
Compare the performance of different models across:
- Different languages (French/English)
- Different text preprocessing methods
- Various metrics (accuracy, F1-score, etc.)

### Model Inference
Test the trained models by:
- Selecting any trained model
- Entering your own text
- Getting real-time sentiment predictions with confidence scores
""")

# Load training results
try:
    with open('models/training_results.json', 'r') as f:
        st.session_state.model_results = json.load(f)
except FileNotFoundError:
    st.error("No pretrained models found. Please run train_models.py first!")
    st.stop()

# Create tabs for model comparison and inference
tab1, tab2 = st.tabs(["Model Comparison", "Inference"])

with tab1:
    st.header('Model Comparison')
    
    # Create comparison metrics DataFrame
    comparison_data = []
    for model_name, configs in st.session_state.model_results.items():
        if configs:  # Check if there are any configurations for this model
            for config_key, results in configs.items():
                if isinstance(results, dict):  # Ensure results is a dictionary
                    comparison_data.append({
                        'Model': model_name,
                        'Configuration': f"{results.get('language', 'Unknown')}, {results.get('text_version', 'Unknown')}",
                        'Accuracy': results.get('accuracy', 0.0),
                        'Weighted F1': results.get('weighted_f1', 0.0),
                        'Training Time (s)': results.get('training_time', 0.0),
                        'Language': results.get('language', 'Unknown'),
                        'Text Version': results.get('text_version', 'Unknown')
                    })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add filters for language and text version
        col1, col2 = st.columns(2)
        with col1:
            selected_language = st.multiselect(
                "Filter by Language",
                options=comparison_df['Language'].unique(),
                default=comparison_df['Language'].unique()
            )
        with col2:
            selected_text_version = st.multiselect(
                "Filter by Text Version",
                options=comparison_df['Text Version'].unique(),
                default=comparison_df['Text Version'].unique()
            )
        
        # Filter the DataFrame
        filtered_df = comparison_df[
            comparison_df['Language'].isin(selected_language) &
            comparison_df['Text Version'].isin(selected_text_version)
        ]
        
        # Style the comparison table
        styled_comparison = filtered_df.style.format({
            'Accuracy': '{:.2%}',
            'Weighted F1': '{:.2%}',
            'Training Time (s)': '{:.2f}'
        }).background_gradient(
            subset=['Accuracy', 'Weighted F1'],
            cmap='YlOrRd'
        )
        
        # Display comparison table
        st.subheader("Model Performance Comparison")
        st.dataframe(styled_comparison, use_container_width=True)
        
        # Visualization of results
        st.subheader("Performance Visualization")
        
        # Create visualization tabs
        viz_tab1, viz_tab2 = st.tabs(["Accuracy Comparison", "Training Time Comparison"])
        
        with viz_tab1:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=filtered_df, x='Model', y='Accuracy', hue='Configuration', ax=ax)
            plt.title('Model Accuracy Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
        
        with viz_tab2:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=filtered_df, x='Model', y='Training Time (s)', hue='Configuration', ax=ax)
            plt.title('Training Time Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Detailed model results
        st.subheader("Detailed Model Results")
        for model_name, configs in st.session_state.model_results.items():
            if configs:  # Check if there are any configurations for this model
                for config_key, results in configs.items():
                    if isinstance(results, dict):  # Ensure results is a dictionary
                        if (results.get('language') in selected_language and 
                            results.get('text_version') in selected_text_version):
                            with st.expander(f"{model_name} ({results.get('language')}, {results.get('text_version')})"):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # Get class labels based on the model type
                                    if model_name in ['RNN', 'CNN', 'FastText']:
                                        class_labels = ['0', '1']  # Binary numeric labels
                                    else:
                                        class_labels = ['negative', 'positive']  # Binary text labels
                                    
                                    # Classification report
                                    if 'report_dict' in results:
                                        report_df = pd.DataFrame({
                                            'Precision': [results['report_dict'][label]['precision'] for label in class_labels],
                                            'Recall': [results['report_dict'][label]['recall'] for label in class_labels],
                                            'F1-Score': [results['report_dict'][label]['f1-score'] for label in class_labels],
                                            'Support': [results['report_dict'][label]['support'] for label in class_labels]
                                        }, index=['Negative', 'Positive'])
                                        
                                        # Add averages
                                        report_df.loc['Macro Avg'] = [
                                            results['report_dict']['macro avg']['precision'],
                                            results['report_dict']['macro avg']['recall'],
                                            results['report_dict']['macro avg']['f1-score'],
                                            results['report_dict']['macro avg']['support']
                                        ]
                                        
                                        styled_report = report_df.style.format({
                                            'Precision': '{:.2%}',
                                            'Recall': '{:.2%}',
                                            'F1-Score': '{:.2%}',
                                            'Support': '{:.0f}'
                                        }).background_gradient(
                                            subset=['Precision', 'Recall', 'F1-Score'],
                                            cmap='YlOrRd'
                                        )
                                        
                                        st.dataframe(styled_report, use_container_width=True)
                                
                                with col2:
                                    # Confusion matrix
                                    if 'confusion_matrix' in results:
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                                                  xticklabels=['Negative', 'Positive'],
                                                  yticklabels=['Negative', 'Positive'])
                                        plt.title(f'Confusion Matrix - {model_name}')
                                        plt.xlabel('Predicted')
                                        plt.ylabel('True')
                                        st.pyplot(fig)
        
        # Best configurations analysis
        st.subheader("Best Configurations Analysis")
        
        # Best by accuracy
        best_accuracy = filtered_df.loc[filtered_df['Accuracy'].idxmax()]
        st.write(f"**Best Accuracy:** {best_accuracy['Model']} ({best_accuracy['Configuration']}) - {best_accuracy['Accuracy']:.2%}")
        
        # Best by F1-score
        best_f1 = filtered_df.loc[filtered_df['Weighted F1'].idxmax()]
        st.write(f"**Best F1-Score:** {best_f1['Model']} ({best_f1['Configuration']}) - {best_f1['Weighted F1']:.2%}")
        
        # Fastest training
        fastest = filtered_df.loc[filtered_df['Training Time (s)'].idxmin()]
        st.write(f"**Fastest Training:** {fastest['Model']} ({fastest['Configuration']}) - {fastest['Training Time (s)']:.2f}s")

with tab2:
    st.header('Model Inference')
    
    st.markdown("""
    ### Test Your Trained Models
    
    Use this section to test your trained models on new text. You can:
    1. Select a trained model
    2. Enter your own text
    3. Get sentiment predictions
    
    The model will classify the text as negative (1-2 stars) or positive (4-5 stars).
    """)
    
    # Create a list of available models
    model_options = []
    for model_name, configs in st.session_state.model_results.items():
        for config_key, results in configs.items():
            if isinstance(results, dict):
                model_options.append({
                    'name': f"{model_name} ({results.get('language', 'Unknown')}, {results.get('text_version', 'Unknown')})",
                    'type': model_name,
                    'language': results.get('language', 'Unknown'),
                    'text_version': results.get('text_version', 'Unknown'),
                    'config_key': config_key
                })
    
    # Model selection
    selected_model = st.selectbox(
        "Select a trained model",
        options=range(len(model_options)),
        format_func=lambda x: model_options[x]['name']
    )
    
    model_info = model_options[selected_model]
    
    # Text input
    user_text = st.text_area(
        "Enter your text for sentiment analysis",
        height=150,
        help="Enter the text you want to analyze. The model will predict its sentiment."
    )
    
    if st.button("Predict Sentiment"):
        if user_text.strip() == "":
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Analyzing sentiment..."):
                # Load the appropriate model based on type
                if model_info['type'] == 'TF-IDF':
                    try:
                        vectorizer = joblib.load(f"models/tfidf_vectorizer_{model_info['config_key']}.joblib")
                        model = joblib.load(f"models/tfidf_model_{model_info['config_key']}.joblib")
                        
                        # Transform the user's text
                        user_text_transformed = vectorizer.transform([user_text])
                        
                        # Make prediction
                        prediction = model.predict(user_text_transformed)
                        probabilities = model.predict_proba(user_text_transformed)
                        
                        # For TF-IDF model, prediction is already a string ('negative', 'positive')
                        predicted_sentiment = prediction[0]
                        
                    except FileNotFoundError:
                        st.error("Model files not found. Please retrain the TF-IDF model.")
                        st.stop()
                
                elif model_info['type'] in ['RNN', 'CNN']:
                    try:
                        # Load vocabulary
                        with open(f"models/{model_info['type'].lower()}_vocab_{model_info['config_key']}.json", 'r') as f:
                            vocab = json.load(f)
                        
                        # Create dataset from user text
                        user_dataset = TextDataset([user_text], [0], vocab)  # Label doesn't matter for inference
                        user_loader = DataLoader(user_dataset, batch_size=1)
                        
                        # Load the appropriate model
                        device = torch.device('mps' if torch.mps.is_available() else 'cpu')
                        if model_info['type'] == 'RNN':
                            model = SentimentRNN(len(vocab), 100, 128, 2, 2, True, 0.3)
                        else:  # CNN
                            model = SentimentCNN(len(vocab), 100, 100, [3, 4, 5], 2, 0.3)
                        
                        model.load_state_dict(torch.load(f"models/{model_info['type'].lower()}_{model_info['config_key']}.pt"))
                        model = model.to(device)
                        model.eval()
                        
                        # Make prediction
                        with torch.no_grad():
                            text_tensor = next(iter(user_loader))[0].to(device)
                            outputs = model(text_tensor)
                            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
                            prediction = np.argmax(probabilities, axis=1)
                            
                            # For neural models, convert numeric prediction to sentiment
                            sentiment_map = {0: "Negative", 1: "Positive"}
                            predicted_sentiment = sentiment_map[prediction[0]]
                    
                    except FileNotFoundError:
                        st.error(f"Model files not found. Please retrain the {model_info['type']} model.")
                        st.stop()
                
                elif model_info['type'] == 'FastText':
                    try:
                        # Load vocabulary
                        with open(f"models/fasttext_vocab_{model_info['config_key']}.json", 'r') as f:
                            vocab = json.load(f)
                        
                        # Create dataset from user text
                        user_dataset = TextDataset([user_text], [0], vocab)  # Label doesn't matter for inference
                        user_loader = DataLoader(user_dataset, batch_size=1)
                        
                        # Load the appropriate model
                        device = torch.device('mps' if torch.mps.is_available() else 'cpu')
                        model = FastTextClassifier(len(vocab), 100, 128, 2, 0.3)
                        model.load_state_dict(torch.load(f"models/fasttext_{model_info['config_key']}.pt"))
                        model = model.to(device)
                        model.eval()
                        
                        # Make prediction
                        with torch.no_grad():
                            text_tensor = next(iter(user_loader))[0].to(device)
                            outputs = model(text_tensor)
                            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
                            prediction = np.argmax(probabilities, axis=1)
                            
                            # Convert numeric prediction to sentiment
                            sentiment_map = {0: "Negative", 1: "Positive"}
                            predicted_sentiment = sentiment_map[prediction[0]]
                    
                    except FileNotFoundError:
                        st.error("FastText model files not found. Please retrain the model.")
                        st.stop()
                
                # Display results
                st.subheader("Prediction Results")
                
                # Display the prediction with appropriate color
                sentiment_color = {
                    "Negative": "red",
                    "Positive": "green",
                    "negative": "red",
                    "positive": "green"
                }
                
                st.markdown(f"""
                <div style='padding: 20px; border-radius: 10px; background-color: {sentiment_color[predicted_sentiment.lower()]}20;'>
                    <h3 style='color: {sentiment_color[predicted_sentiment.lower()]}; margin:0;'>
                        {predicted_sentiment} Sentiment
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display probability distribution
                st.subheader("Confidence Scores")
                
                # Handle probabilities based on model type
                if model_info['type'] == 'TF-IDF':
                    sentiments = ['Negative', 'Positive']
                    # Map probabilities to correct sentiment order based on model's classes
                    prob_dict = {label: prob for label, prob in zip(model.classes_, probabilities[0])}
                    ordered_probs = [prob_dict[label.lower()] for label in sentiments]
                    probs_df = pd.DataFrame({
                        'Sentiment': sentiments,
                        'Probability': ordered_probs
                    })
                else:
                    probs_df = pd.DataFrame({
                        'Sentiment': ['Negative', 'Positive'],
                        'Probability': probabilities[0]
                    })
                
                # Create a bar plot
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(probs_df['Sentiment'], probs_df['Probability'])
                
                # Color the bars based on sentiment
                for bar, sentiment in zip(bars, probs_df['Sentiment']):
                    bar.set_color(sentiment_color[sentiment])
                    bar.set_alpha(0.7)
                
                ax.set_ylim(0, 1)
                plt.title('Prediction Probabilities')
                plt.ylabel('Probability')
                
                # Add percentage labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1%}',
                           ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Add interpretation
                st.subheader("Interpretation")
                confidence = max(probabilities[0])
                
                if confidence > 0.8:
                    confidence_text = "very confident"
                elif confidence > 0.6:
                    confidence_text = "moderately confident"
                else:
                    confidence_text = "uncertain"
                
                st.write(f"""
                The model is {confidence_text} about this prediction:
                - Predicted sentiment: **{predicted_sentiment}**
                - Confidence: **{confidence:.1%}**
                
                This means the text is most likely expressing a {predicted_sentiment.lower()} sentiment.
                """) 