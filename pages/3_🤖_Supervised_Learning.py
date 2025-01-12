import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import tensorflow as tf
import tensorflow_hub as hub
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
import os
import glob
import time
import json

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
        # text shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))  # [batch_size, seq_len, embedding_dim]
        
        # Pack sequence for RNN
        output, (hidden, cell) = self.rnn(embedded)
        
        if self.rnn.bidirectional:
            # Concatenate the final forward and backward hidden states
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

def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label)
        acc = accuracy_score(batch.label.cpu().numpy(), predictions.argmax(1).cpu().numpy())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def create_vocabulary(texts, min_freq=2):
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.xavier_uniform_(m.weight)

st.set_page_config(
    page_title="Supervised Learning",
    page_icon="🤖",
    layout="wide"
)

st.title('🤖 Supervised Learning Models')

# Add explanation of what we're doing
st.markdown("""
### What are we doing here?
This section implements sentiment analysis on customer reviews using three different machine learning approaches:

1. **TF-IDF + Logistic Regression**: A traditional machine learning approach that:
   - Converts text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
   - Uses logistic regression to classify reviews into negative (1-2 stars), neutral (3 stars), or positive (4-5 stars)
   - Fast to train and interpretable, but doesn't capture word order or context

2. **RNN (Recurrent Neural Network)**: A deep learning approach that:
   - Uses word embeddings to represent words as dense vectors
   - Processes text sequentially, capturing word order and context
   - Better at understanding long-term dependencies in text

3. **CNN (Convolutional Neural Network)**: A deep learning approach that:
   - Also uses word embeddings
   - Applies convolution operations to detect local patterns and features in text
   - Good at capturing local patterns and key phrases

### Training Data
We train these models on customer reviews with their associated ratings:
- Input: Review text (in either French or English)
- Output: Sentiment classification (negative/neutral/positive)
- The models learn to predict sentiment based on the text content

### Why Multiple Models?
Each model has its strengths:
- TF-IDF + LogReg: Fast, interpretable baseline
- RNN: Good for capturing sequential patterns and context
- CNN: Excellent at detecting key phrases and local patterns

Choose the model based on your needs for accuracy, speed, and interpretability.
""")

def load_data():
    """Load preprocessed data"""
    processed_data_path = 'data/processed_data.pkl'
    if not os.path.exists(processed_data_path):
        st.error("Please run the Data Cleaning & Analysis page first to prepare the data!")
        st.stop()
    return pd.read_pickle(processed_data_path)

# Load data
with st.spinner('Loading preprocessed data...'):
    df = load_data()

# Language selection
language = st.radio("Select Language for Analysis", ["French", "English"])

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

# Initialize session state for model results
if 'model_results' not in st.session_state:
    st.session_state.model_results = {
        'TF-IDF': {},  # Will store results by config: {config_key: results}
        'RNN': {},
        'CNN': {}
    }

# Create tabs for different models
tab1, tab2, tab3, tab4 = st.tabs([
    "TF-IDF + Classical ML",
    "RNN with Embeddings",
    "CNN with Embeddings",
    "Model Comparison"
])

with tab1:
    st.header('TF-IDF with Logistic Regression')
    
    if st.button("Train Classical Model"):
        with st.spinner("Training classical model..."):
            # Start timing
            start_time = time.time()
            
            # Get selected text column
            text_col = text_col_map[text_version]
            
            # Prepare data
            # Convert ratings to sentiment classes (1-2: negative, 3: neutral, 4-5: positive)
            df['sentiment'] = pd.cut(df['note'], 
                                   bins=[-np.inf, 2.5, 3.5, np.inf], 
                                   labels=['negative', 'neutral', 'positive'])
            
            # Remove rows with missing values
            df_clean = df.dropna(subset=['note', text_col])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df_clean[text_col],
                df_clean['sentiment'],
                test_size=0.2,
                random_state=42
            )
            
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            
            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_tfidf, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_tfidf)
            
            # Get classification report as dict
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            
            # Create DataFrame for the report
            report_df = pd.DataFrame({
                'Precision': [report_dict[label]['precision'] for label in ['negative', 'neutral', 'positive']],
                'Recall': [report_dict[label]['recall'] for label in ['negative', 'neutral', 'positive']],
                'F1-Score': [report_dict[label]['f1-score'] for label in ['negative', 'neutral', 'positive']],
                'Support': [report_dict[label]['support'] for label in ['negative', 'neutral', 'positive']]
            }, index=['Negative', 'Neutral', 'Positive'])
            
            # Add accuracy row
            accuracy_row = pd.DataFrame({
                'Precision': [report_dict['accuracy']],
                'Recall': [report_dict['accuracy']],
                'F1-Score': [report_dict['accuracy']],
                'Support': [report_dict['macro avg']['support']]
            }, index=['Accuracy'])
            
            # Add average rows
            macro_avg = pd.DataFrame({
                'Precision': [report_dict['macro avg']['precision']],
                'Recall': [report_dict['macro avg']['recall']],
                'F1-Score': [report_dict['macro avg']['f1-score']],
                'Support': [report_dict['macro avg']['support']]
            }, index=['Macro Avg'])
            
            weighted_avg = pd.DataFrame({
                'Precision': [report_dict['weighted avg']['precision']],
                'Recall': [report_dict['weighted avg']['recall']],
                'F1-Score': [report_dict['weighted avg']['f1-score']],
                'Support': [report_dict['weighted avg']['support']]
            }, index=['Weighted Avg'])
            
            # Combine all rows
            final_report_df = pd.concat([report_df, accuracy_row, macro_avg, weighted_avg])
            
            # Style the DataFrame
            styled_df = final_report_df.style.format({
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'Support': '{:.0f}'
            }).background_gradient(
                subset=['Precision', 'Recall', 'F1-Score'],
                cmap='YlOrRd'
            )
            
            # Display results
            st.subheader(f"Classification Report ({language} - {text_version})")
            
            # Create two columns for metrics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=250
                )
            
            with col2:
                # Display key metrics
                st.metric(
                    label="Overall Accuracy",
                    value=f"{report_dict['accuracy']:.2%}",
                )
                
                st.metric(
                    label="Weighted F1-Score",
                    value=f"{report_dict['weighted avg']['f1-score']:.2%}"
                )

            # Plot confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Neutral', 'Positive'],
                       yticklabels=['Negative', 'Neutral', 'Positive'])
            plt.title(f'Confusion Matrix ({language} - {text_version})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            st.pyplot(fig)
            
            # Feature importance analysis
            st.subheader("Most Important Features")
            feature_names = vectorizer.get_feature_names_out()
            for i, sentiment_class in enumerate(model.classes_):
                top_features_idx = np.argsort(model.coef_[i])[-10:]
                st.write(f"\nTop features for {sentiment_class}:")
                for idx in top_features_idx:
                    st.write(f"- {feature_names[idx]}: {model.coef_[i][idx]:.4f}")

            # Store results in session state with configuration key
            config_key = f"{language}_{text_version}"
            if 'TF-IDF' not in st.session_state.model_results:
                st.session_state.model_results['TF-IDF'] = {}
            st.session_state.model_results['TF-IDF'][config_key] = {
                'accuracy': report_dict['accuracy'],
                'weighted_f1': report_dict['weighted avg']['f1-score'],
                'training_time': time.time() - start_time,
                'report_dict': report_dict,
                'confusion_matrix': cm,
                'language': language,
                'text_version': text_version
            }

with tab2:
    st.header('RNN with Embeddings')
    
    # Add parameters in expander
    with st.expander("Show/Hide Model Parameters"):
        st.markdown("""
        **RNN Model Parameters:**
        Adjust these parameters to control the model's architecture and training behavior.
        """)
        # Model parameters
        embedding_dim = st.slider("Embedding dimension", 50, 300, 100)
        hidden_dim = st.slider("Hidden dimension", 32, 256, 128)
        n_layers = st.slider("Number of layers", 1, 4, 2)
        bidirectional = st.checkbox("Bidirectional", value=True)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.3)
        batch_size = st.slider("Batch Size", 16, 128, 32)
    
    if st.button("Train RNN Model"):
        with st.spinner("Training RNN model..."):
            # Get selected text column
            text_col = text_col_map[text_version]
            
            # Prepare data
            df_clean = df.dropna(subset=['note', text_col])
            df_clean['sentiment'] = pd.cut(df_clean['note'], 
                                         bins=[-np.inf, 2.5, 3.5, np.inf], 
                                         labels=[0, 1, 2])
            
            # Create vocabulary
            vocab = create_vocabulary(df_clean[text_col])
            vocab_size = len(vocab)
            
            # Create datasets
            X_train, X_test, y_train, y_test = train_test_split(
                df_clean[text_col], df_clean['sentiment'], test_size=0.2, random_state=42
            )
            
            train_dataset = TextDataset(X_train.values, y_train.values, vocab)
            test_dataset = TextDataset(X_test.values, y_test.values, vocab)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
            
            # Create model
            device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
            model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, 3, n_layers, bidirectional, dropout)
            model = model.to(device)
            model.apply(init_weights)
            
            # Setup training
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # TensorBoard setup
            log_dir = f"runs/rnn_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            writer = SummaryWriter(log_dir)
            
            # Training loop
            n_epochs = 5
            best_acc = 0
            
            progress_bar = st.progress(0)
            metrics_placeholder = st.empty()
            
            for epoch in range(n_epochs):
                start_time = time.time()
                
                # Training
                model.train()
                train_loss = 0
                train_acc = 0
                for batch_idx, (text, labels) in enumerate(train_loader):
                    # Move tensors to device
                    text, labels = text.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    predictions = model(text)
                    loss = criterion(predictions, labels)
                    acc = accuracy_score(labels.cpu().numpy(), predictions.argmax(1).cpu().numpy())
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_acc += acc
                    
                    # Update progress
                    progress = (epoch * len(train_loader) + batch_idx) / (n_epochs * len(train_loader))
                    progress_bar.progress(progress)
                
                # Validation
                model.eval()
                val_loss = 0
                val_acc = 0
                predictions_list = []
                labels_list = []
                
                with torch.no_grad():
                    for text, labels in test_loader:
                        # Move tensors to device
                        text, labels = text.to(device), labels.to(device)
                        
                        predictions = model(text)
                        loss = criterion(predictions, labels)
                        acc = accuracy_score(labels.cpu().numpy(), predictions.argmax(1).cpu().numpy())
                        
                        val_loss += loss.item()
                        val_acc += acc
                        
                        predictions_list.extend(predictions.argmax(1).cpu().numpy())
                        labels_list.extend(labels.cpu().numpy())
                
                # Calculate epoch metrics
                train_loss = train_loss / len(train_loader)
                train_acc = train_acc / len(train_loader)
                val_loss = val_loss / len(test_loader)
                val_acc = val_acc / len(test_loader)

                # Log to TensorBoard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                
                # Log embeddings
                if epoch == n_epochs - 1:
                    writer.add_embedding(model.embedding.weight)
                
                # Update metrics display
                metrics_placeholder.write(f"""
                Epoch {epoch+1}/{n_epochs}:
                - Train Loss: {train_loss:.4f}
                - Train Acc: {train_acc:.4f}
                - Val Loss: {val_loss:.4f}
                - Val Acc: {val_acc:.4f}
                - Time: {time.time() - start_time:.2f}s
                """)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    # Save best model
                    torch.save(model.state_dict(), f'models/rnn_{language}_{text_version}.pt')
            
            writer.close()
            st.success(f"Training completed! Best validation accuracy: {best_acc:.4f}")
            st.write("Model architecture:")
            st.write(model)
            
            # Visualization of embeddings
            st.subheader("Embedding Visualization")
            st.write("You can view the embeddings visualization in TensorBoard:")
            st.code(f"tensorboard --logdir={log_dir}")

            # Create classification report
            model.eval()
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for text, labels in test_loader:
                    text, labels = text.to(device), labels.to(device)
                    predictions = model(text)
                    all_predictions.extend(predictions.argmax(1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate final metrics
            report_dict = classification_report(all_labels, all_predictions, output_dict=True)
            
            # Create and display metrics similar to TF-IDF section
            report_df = pd.DataFrame({
                'Precision': [report_dict[str(i)]['precision'] for i in range(3)],
                'Recall': [report_dict[str(i)]['recall'] for i in range(3)],
                'F1-Score': [report_dict[str(i)]['f1-score'] for i in range(3)],
                'Support': [report_dict[str(i)]['support'] for i in range(3)]
            }, index=['Negative', 'Neutral', 'Positive'])
            
            # Add accuracy and averages rows (similar to previous sections)
            accuracy_row = pd.DataFrame({
                'Precision': [report_dict['accuracy']],
                'Recall': [report_dict['accuracy']],
                'F1-Score': [report_dict['accuracy']],
                'Support': [report_dict['macro avg']['support']]
            }, index=['Accuracy'])
            
            macro_avg = pd.DataFrame({
                'Precision': [report_dict['macro avg']['precision']],
                'Recall': [report_dict['macro avg']['recall']],
                'F1-Score': [report_dict['macro avg']['f1-score']],
                'Support': [report_dict['macro avg']['support']]
            }, index=['Macro Avg'])
            
            weighted_avg = pd.DataFrame({
                'Precision': [report_dict['weighted avg']['precision']],
                'Recall': [report_dict['weighted avg']['recall']],
                'F1-Score': [report_dict['weighted avg']['f1-score']],
                'Support': [report_dict['weighted avg']['support']]
            }, index=['Weighted Avg'])
            
            # Combine all rows
            final_report_df = pd.concat([report_df, accuracy_row, macro_avg, weighted_avg])
            
            # Style the DataFrame
            styled_df = final_report_df.style.format({
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'Support': '{:.0f}'
            }).background_gradient(
                subset=['Precision', 'Recall', 'F1-Score'],
                cmap='YlOrRd'
            )
            
            # Display metrics
            st.subheader("Final Model Performance")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(styled_df, use_container_width=True, height=250)
            
            with col2:
                st.metric(
                    label="Best Validation Accuracy",
                    value=f"{best_acc:.2%}",
                    delta=f"{best_acc - 0.33:.2%} vs random"
                )
                st.metric(
                    label="Training Time",
                    value=f"{time.time() - start_time:.2f}s"
                )

            # Store RNN results with configuration key
            config_key = f"{language}_{text_version}"
            if 'RNN' not in st.session_state.model_results:
                st.session_state.model_results['RNN'] = {}
            st.session_state.model_results['RNN'][config_key] = {
                'accuracy': best_acc,
                'weighted_f1': report_dict['weighted avg']['f1-score'],
                'training_time': time.time() - start_time,
                'report_dict': report_dict,
                'confusion_matrix': confusion_matrix(all_labels, all_predictions),
                'language': language,
                'text_version': text_version
            }

with tab3:
    st.header('CNN with Embeddings')
    
    # Add parameters in expander
    with st.expander("Show/Hide Model Parameters"):
        st.markdown("""
        **CNN Model Parameters:**
        Adjust these parameters to control the model's architecture and training behavior.
        """)
        # Model parameters
        embedding_dim = st.slider("Embedding dimension", 50, 300, 100, key="cnn_embed")
        n_filters = st.slider("Number of filters", 32, 256, 100)
        filter_sizes = st.multiselect("Filter sizes", [2, 3, 4, 5], default=[3, 4, 5])
        dropout = st.slider("Dropout", 0.0, 0.5, 0.3, key="cnn_dropout")
    
    if st.button("Train CNN Model"):
        with st.spinner("Training CNN model..."):
            # Get selected text column
            text_col = text_col_map[text_version]
            
            # Prepare data
            df_clean = df.dropna(subset=['note', text_col])
            df_clean['sentiment'] = pd.cut(df_clean['note'], 
                                         bins=[-np.inf, 2.5, 3.5, np.inf], 
                                         labels=[0, 1, 2])
            
            # Create vocabulary
            vocab = create_vocabulary(df_clean[text_col])
            vocab_size = len(vocab)
            
            # Create datasets
            X_train, X_test, y_train, y_test = train_test_split(
                df_clean[text_col], df_clean['sentiment'], test_size=0.2, random_state=42
            )
            
            train_dataset = TextDataset(X_train.values, y_train.values, vocab)
            test_dataset = TextDataset(X_test.values, y_test.values, vocab)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            # Create model
            model = SentimentCNN(vocab_size, embedding_dim, n_filters, filter_sizes, 3, dropout)
            model.apply(init_weights)
            
            # Setup training
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # TensorBoard setup
            log_dir = f"runs/cnn_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            writer = SummaryWriter(log_dir)
            
            # Training loop
            n_epochs = 5
            best_acc = 0
            
            progress_bar = st.progress(0)
            metrics_placeholder = st.empty()
            
            for epoch in range(n_epochs):
                start_time = time.time()
                
                # Training
                model.train()
                train_loss = 0
                train_acc = 0
                for batch_idx, (text, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    predictions = model(text)
                    loss = criterion(predictions, labels)
                    acc = accuracy_score(labels.numpy(), predictions.argmax(1).numpy())
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_acc += acc
                    
                    # Update progress
                    progress = (epoch * len(train_loader) + batch_idx) / (n_epochs * len(train_loader))
                    progress_bar.progress(progress)
                
                # Validation
                model.eval()
                val_loss = 0
                val_acc = 0
                predictions_list = []
                labels_list = []
                with torch.no_grad():
                    for text, labels in test_loader:
                        predictions = model(text)
                        loss = criterion(predictions, labels)
                        acc = accuracy_score(labels.numpy(), predictions.argmax(1).numpy())
                        val_loss += loss.item()
                        val_acc += acc
                        
                        predictions_list.extend(predictions.argmax(1).numpy())
                        labels_list.extend(labels.numpy())
                
                # Calculate epoch metrics
                train_loss = train_loss / len(train_loader)
                train_acc = train_acc / len(train_loader)
                val_loss = val_loss / len(test_loader)
                val_acc = val_acc / len(test_loader)
                
                # Log to TensorBoard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                
                # Log embeddings and feature maps
                if epoch == n_epochs - 1:
                    writer.add_embedding(model.embedding.weight)
                    
                    # Log sample feature maps
                    sample_text = next(iter(test_loader))[0][:4]  # Get 4 samples
                    embedded = model.embedding(sample_text).unsqueeze(1)
                    for i, conv in enumerate(model.convs):
                        feature_maps = conv(embedded).squeeze(3)
                        for j in range(min(4, feature_maps.size(1))):  # Log first 4 feature maps
                            writer.add_image(f'Feature_Maps/conv{i+1}_filter{j+1}', 
                                          feature_maps[0, j:j+1].unsqueeze(0), 
                                          epoch)
                
                # Update metrics display
                metrics_placeholder.write(f"""
                Epoch {epoch+1}/{n_epochs}:
                - Train Loss: {train_loss:.4f}
                - Train Acc: {train_acc:.4f}
                - Val Loss: {val_loss:.4f}
                - Val Acc: {val_acc:.4f}
                - Time: {time.time() - start_time:.2f}s
                """)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    # Save best model
                    torch.save(model.state_dict(), f'models/cnn_{language}_{text_version}.pt')
                    
                    # Create and save confusion matrix
                    cm = confusion_matrix(labels_list, predictions_list)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                              xticklabels=['Negative', 'Neutral', 'Positive'],
                              yticklabels=['Negative', 'Neutral', 'Positive'])
                    plt.title(f'Confusion Matrix - CNN ({language})')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    writer.add_figure('Confusion_Matrix', fig, epoch)
            
            writer.close()
            st.success(f"Training completed! Best validation accuracy: {best_acc:.4f}")
            
            # Display model architecture and performance
            st.write("Model architecture:")
            st.write(model)
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            st.pyplot(fig)
            
            # Visualization of embeddings
            st.subheader("Embedding Visualization")
            st.write("You can view the embeddings and feature maps visualization in TensorBoard:")
            st.code(f"tensorboard --logdir={log_dir}")
            
            # Feature importance analysis
            st.subheader("Filter Visualization")
            st.write("The model learned the following filter patterns:")
            for i, size in enumerate(filter_sizes):
                st.write(f"Filter size {size}:")
                weights = model.convs[i].weight.data.squeeze()
                fig, ax = plt.subplots(figsize=(10, 2))
                sns.heatmap(weights[0].detach().numpy(), cmap='viridis')
                st.pyplot(fig)

            # Create classification report
            model.eval()
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for text, labels in test_loader:
                    text, labels = text.to(device), labels.to(device)
                    predictions = model(text)
                    all_predictions.extend(predictions.argmax(1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate final metrics
            report_dict = classification_report(all_labels, all_predictions, output_dict=True)
            
            # Create and display metrics similar to TF-IDF section
            report_df = pd.DataFrame({
                'Precision': [report_dict[str(i)]['precision'] for i in range(3)],
                'Recall': [report_dict[str(i)]['recall'] for i in range(3)],
                'F1-Score': [report_dict[str(i)]['f1-score'] for i in range(3)],
                'Support': [report_dict[str(i)]['support'] for i in range(3)]
            }, index=['Negative', 'Neutral', 'Positive'])
            
            # Add accuracy and averages rows (similar to previous sections)
            accuracy_row = pd.DataFrame({
                'Precision': [report_dict['accuracy']],
                'Recall': [report_dict['accuracy']],
                'F1-Score': [report_dict['accuracy']],
                'Support': [report_dict['macro avg']['support']]
            }, index=['Accuracy'])
            
            macro_avg = pd.DataFrame({
                'Precision': [report_dict['macro avg']['precision']],
                'Recall': [report_dict['macro avg']['recall']],
                'F1-Score': [report_dict['macro avg']['f1-score']],
                'Support': [report_dict['macro avg']['support']]
            }, index=['Macro Avg'])
            
            weighted_avg = pd.DataFrame({
                'Precision': [report_dict['weighted avg']['precision']],
                'Recall': [report_dict['weighted avg']['recall']],
                'F1-Score': [report_dict['weighted avg']['f1-score']],
                'Support': [report_dict['weighted avg']['support']]
            }, index=['Weighted Avg'])
            
            # Combine all rows
            final_report_df = pd.concat([report_df, accuracy_row, macro_avg, weighted_avg])
            
            # Style the DataFrame
            styled_df = final_report_df.style.format({
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'Support': '{:.0f}'
            }).background_gradient(
                subset=['Precision', 'Recall', 'F1-Score'],
                cmap='YlOrRd'
            )
            
            # Display metrics
            st.subheader("Final Model Performance")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(styled_df, use_container_width=True, height=250)
            
            with col2:
                st.metric(
                    label="Best Validation Accuracy",
                    value=f"{best_acc:.2%}",
                    delta=f"{best_acc - 0.33:.2%} vs random"
                )
                st.metric(
                    label="Training Time",
                    value=f"{time.time() - start_time:.2f}s"
                )

            # Store CNN results with configuration key
            config_key = f"{language}_{text_version}"
            if 'CNN' not in st.session_state.model_results:
                st.session_state.model_results['CNN'] = {}
            st.session_state.model_results['CNN'][config_key] = {
                'accuracy': best_acc,
                'weighted_f1': report_dict['weighted avg']['f1-score'],
                'training_time': time.time() - start_time,
                'report_dict': report_dict,
                'confusion_matrix': confusion_matrix(all_labels, all_predictions),
                'language': language,
                'text_version': text_version
            }

with tab4:
    st.header('Model Comparison')
    
    # Add explanation of what we're comparing
    st.markdown("""
    ### Understanding the Comparison
    
    This section helps you compare the performance of different models across various metrics:
    
    **Key Metrics:**
    - **Accuracy**: How often the model correctly predicts the sentiment
    - **Weighted F1-Score**: Balance between precision and recall, weighted by class size
    - **Training Time**: How long it takes to train each model
    
    **What to Look For:**
    1. **Best Overall Performance**: Which model achieves the highest accuracy/F1-score?
    2. **Speed vs Accuracy Tradeoff**: Is the slight improvement in accuracy worth the longer training time?
    3. **Language Impact**: How do models perform across different languages?
    4. **Text Processing Impact**: How do different text preprocessing steps affect performance?
    
    Use these comparisons to choose the best model for your specific needs, considering:
    - Required accuracy
    - Available computational resources
    - Need for real-time predictions
    - Interpretability requirements
    """)
    
    # Check if we have any results
    has_results = any(bool(configs) for configs in st.session_state.model_results.values())
    
    if not has_results:
        st.info("Train some models to see their comparison!")
    else:
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
                                        if model_name in ['RNN', 'CNN']:
                                            class_labels = ['0', '1', '2']  # Numeric labels
                                        else:
                                            class_labels = ['negative', 'neutral', 'positive']  # Text labels
                                        
                                        # Classification report
                                        if 'report_dict' in results:
                                            report_df = pd.DataFrame({
                                                'Precision': [results['report_dict'][label]['precision'] for label in class_labels],
                                                'Recall': [results['report_dict'][label]['recall'] for label in class_labels],
                                                'F1-Score': [results['report_dict'][label]['f1-score'] for label in class_labels],
                                                'Support': [results['report_dict'][label]['support'] for label in class_labels]
                                            }, index=['Negative', 'Neutral', 'Positive'])
                                            
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
                                                      xticklabels=['Negative', 'Neutral', 'Positive'],
                                                      yticklabels=['Negative', 'Neutral', 'Positive'])
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
            
            # Model characteristics
            st.subheader("Model Characteristics")
            characteristics_data = {
                'Model': ['TF-IDF + LogReg', 'RNN', 'CNN'],
                'Training Speed': ['Very Fast', 'Moderate', 'Moderate'],
                'Inference Speed': ['Very Fast', 'Fast', 'Fast'],
                'Memory Usage': ['Low', 'Moderate', 'Moderate'],
                'Interpretability': ['High', 'Medium', 'Medium'],
                'Best Use Case': [
                    'Quick analysis & baseline',
                    'Sequential patterns',
                    'Local patterns & features'
                ]
            }
            st.table(pd.DataFrame(characteristics_data)) 