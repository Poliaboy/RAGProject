import pandas as pd
import numpy as np
import os
import glob
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_and_process_data():
    """Load data and create documents"""
    data_path = 'TraductionAvisClients'
    all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    dfs = []
    
    print("Loading Excel files...")
    for file in all_files:
        df = pd.read_excel(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total reviews loaded: {len(df)}")
    
    # Prepare documents from reviews only
    documents = []
    for _, row in df.iterrows():
        try:
            # Get only English review
            english_review = row['avis_en']
            rating = row['note']
            company = row['assureur']
            product = row['produit']
            author = row['auteur']
            
            # Create document with English version only
            content = (
                f"Insurance Company: {company}\n"
                f"Product: {product}\n"
                f"Rating: {rating}\n"
                f"Author: {author}\n"
                f"Review: {english_review}"
            )
            metadata = {
                "type": "review",
                "company": company,
                "product": product,
                "rating": float(rating),
                "author": author
            }
            documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"Skipping row due to error: {str(e)}")
            continue
    
    print(f"Created {len(documents)} documents")
    
    # Split documents with larger chunk size to reduce number of embeddings
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks")
    
    return docs

def create_vector_store():
    """Create and save vector store using local embeddings"""
    # Load documents
    docs = load_and_process_data()
    
    # Initialize local embeddings model with MPS acceleration
    print("Initializing local embeddings model...")
    
    # Check if MPS is available
    import torch
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Metal Performance Shaders) acceleration")
    else:
        device = 'cpu'
        print("MPS not available, falling back to CPU")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller, English-focused model
        model_kwargs={'device': device}
    )
    
    # Create vector store
    print("Creating vector store...")
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Save vector store
    cache_dir = "data/vector_store_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "faiss_index")
    
    print(f"Saving vector store to {cache_path}...")
    vector_store.save_local(cache_path, allow_dangerous_deserialization=True)
    print("Vector store saved successfully!")
    
    return vector_store

if __name__ == "__main__":
    create_vector_store() 