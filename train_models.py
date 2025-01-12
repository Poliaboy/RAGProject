import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import time
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

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

def train_tfidf_model(df, text_col, language, text_version):
    print(f"\nTraining TF-IDF model for {language} - {text_version}")
    start_time = time.time()
    
    # Convert ratings to sentiment classes
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
    
    # Save the model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/tfidf_model_{language}_{text_version}.joblib')
    joblib.dump(vectorizer, f'models/tfidf_vectorizer_{language}_{text_version}.joblib')
    
    # Calculate metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Save results
    results = {
        'accuracy': report_dict['accuracy'],
        'weighted_f1': report_dict['weighted avg']['f1-score'],
        'training_time': time.time() - start_time,
        'report_dict': report_dict,
        'confusion_matrix': cm.tolist(),
        'language': language,
        'text_version': text_version
    }
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Training time: {results['training_time']:.2f}s")
    return results

def train_neural_model(model_type, df, text_col, language, text_version, device):
    print(f"\nTraining {model_type} model for {language} - {text_version}")
    start_time = time.time()
    
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
    if model_type == 'RNN':
        model = SentimentRNN(vocab_size, 100, 128, 3, 2, True, 0.3)
    else:  # CNN
        model = SentimentCNN(vocab_size, 100, 100, [3, 4, 5], 3, 0.3)
    
    model = model.to(device)
    model.apply(init_weights)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    n_epochs = 5
    best_acc = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (text, labels) in enumerate(train_loader):
            text, labels = text.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(text)
            loss = criterion(predictions, labels)
            acc = accuracy_score(labels.cpu().numpy(), predictions.argmax(1).cpu().numpy())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        predictions_list = []
        labels_list = []
        
        with torch.no_grad():
            for text, labels in test_loader:
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
        
        print(f'Epoch: {epoch+1}')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'\tVal. Loss: {val_loss:.4f} | Val. Acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save best model
            torch.save(model.state_dict(), f'models/{model_type.lower()}_{language}_{text_version}.pt')
            # Save vocabulary
            with open(f'models/{model_type.lower()}_vocab_{language}_{text_version}.json', 'w') as f:
                json.dump(vocab, f)
    
    # Calculate final metrics
    report_dict = classification_report(labels_list, predictions_list, output_dict=True)
    cm = confusion_matrix(labels_list, predictions_list)
    
    # Save results
    results = {
        'accuracy': best_acc,
        'weighted_f1': report_dict['weighted avg']['f1-score'],
        'training_time': time.time() - start_time,
        'report_dict': report_dict,
        'confusion_matrix': cm.tolist(),
        'language': language,
        'text_version': text_version
    }
    
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Training time: {results['training_time']:.2f}s")
    return results

def main():
    # Load data
    print("Loading data...")
    df = pd.read_pickle('data/processed_data.pkl')
    
    # Define configurations
    languages = ["French", "English"]
    text_versions = ["Cleaned", "No Stopwords", "Lemmatized", "No Stopwords + Lemmatized"]
    
    # Map text versions to column names
    text_col_map = {
        ("French", "Cleaned"): "avis_cleaned",
        ("French", "No Stopwords"): "avis_no_stop",
        ("French", "Lemmatized"): "avis_lemmatized",
        ("French", "No Stopwords + Lemmatized"): "avis_no_stop_and_lemmatized",
        ("English", "Cleaned"): "avis_en_cleaned",
        ("English", "No Stopwords"): "avis_en_no_stop",
        ("English", "Lemmatized"): "avis_en_lemmatized",
        ("English", "No Stopwords + Lemmatized"): "avis_en_no_stop_and_lemmatized"
    }
    
    # Initialize results dictionary
    all_results = {
        'TF-IDF': {},
        'RNN': {},
        'CNN': {}
    }
    
    # Set device
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train models for each configuration
    for language in languages:
        for text_version in text_versions:
            text_col = text_col_map[(language, text_version)]
            config_key = f"{language}_{text_version}"
            
            # Train TF-IDF
            results = train_tfidf_model(df, text_col, language, text_version)
            all_results['TF-IDF'][config_key] = results
            
            # Train RNN
            results = train_neural_model('RNN', df, text_col, language, text_version, device)
            all_results['RNN'][config_key] = results
            
            # Train CNN
            results = train_neural_model('CNN', df, text_col, language, text_version, device)
            all_results['CNN'][config_key] = results
    
    # Save all results
    with open('models/training_results.json', 'w') as f:
        json.dump(all_results, f)
    
    print("\nAll models trained successfully!")
    print("Results saved to models/training_results.json")

if __name__ == "__main__":
    main() 