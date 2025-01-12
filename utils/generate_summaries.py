import pandas as pd
import numpy as np
import os
import glob
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json

def load_data():
    """Load and preprocess the review data"""
    data_path = 'TraductionAvisClients'
    all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    dfs = []
    
    print("Loading Excel files...")
    for file in all_files:
        df = pd.read_excel(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total reviews loaded: {len(df)}")
    return df

def generate_company_summaries(df, llm):
    """Generate summaries for each company"""
    print("\nGenerating company summaries...")
    company_summaries = {}
    
    for company in df['assureur'].unique():
        company_reviews = df[df['assureur'] == company]
        
        # Calculate statistics
        total_reviews = len(company_reviews)
        avg_rating = company_reviews['note'].mean()
        rating_dist = company_reviews['note'].value_counts().sort_index().to_dict()
        products = company_reviews['produit'].value_counts().to_dict()
        
        # Prepare context for the LLM
        context = f"""
        Company: {company}
        Total Reviews: {total_reviews}
        Average Rating: {avg_rating:.2f}
        Rating Distribution: {rating_dist}
        Products: {products}
        """
        
        prompt = f"""Based on the following statistics about an insurance company, provide a comprehensive summary:

{context}

Generate a detailed summary that covers:
1. Overall customer satisfaction and rating trends
2. Key products and their popularity
3. Main strengths and areas for improvement
4. Notable patterns in customer feedback

Keep the summary factual and objective. Focus on the most significant insights."""

        print(f"Generating summary for {company}...")
        response = llm.invoke(prompt)
        company_summaries[company] = {
            "summary": response.content,
            "stats": {
                "total_reviews": total_reviews,
                "avg_rating": float(avg_rating),
                "rating_distribution": rating_dist,
                "products": products
            }
        }
    
    return company_summaries

def generate_product_summaries(df, llm):
    """Generate summaries for each product type"""
    print("\nGenerating product summaries...")
    product_summaries = {}
    
    for product in df['produit'].unique():
        product_reviews = df[df['produit'] == product]
        
        # Calculate statistics
        total_reviews = len(product_reviews)
        avg_rating = product_reviews['note'].mean()
        rating_dist = product_reviews['note'].value_counts().sort_index().to_dict()
        companies = product_reviews['assureur'].value_counts().to_dict()
        
        # Prepare context for the LLM
        context = f"""
        Product: {product}
        Total Reviews: {total_reviews}
        Average Rating: {avg_rating:.2f}
        Rating Distribution: {rating_dist}
        Companies Offering: {companies}
        """
        
        prompt = f"""Based on the following statistics about an insurance product, provide a comprehensive summary:

{context}

Generate a detailed summary that covers:
1. Overall customer satisfaction with this type of insurance
2. Key companies offering this product and their performance
3. Common themes in customer feedback
4. Industry-wide patterns for this product type

Keep the summary factual and objective. Focus on the most significant insights."""

        print(f"Generating summary for {product}...")
        response = llm.invoke(prompt)
        product_summaries[product] = {
            "summary": response.content,
            "stats": {
                "total_reviews": total_reviews,
                "avg_rating": float(avg_rating),
                "rating_distribution": rating_dist,
                "companies": companies
            }
        }
    
    return product_summaries

def create_summary_documents(company_summaries, product_summaries):
    """Create documents from summaries for embedding"""
    documents = []
    
    # Add company summaries
    for company, data in company_summaries.items():
        content = f"""Company Summary: {company}
Average Rating: {data['stats']['avg_rating']:.2f}
Total Reviews: {data['stats']['total_reviews']}

{data['summary']}

Statistics:
- Rating Distribution: {data['stats']['rating_distribution']}
- Products Offered: {data['stats']['products']}"""
        
        metadata = {
            "type": "company_summary",
            "company": company,
            "avg_rating": data['stats']['avg_rating'],
            "total_reviews": data['stats']['total_reviews']
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Add product summaries
    for product, data in product_summaries.items():
        content = f"""Product Summary: {product}
Average Rating: {data['stats']['avg_rating']:.2f}
Total Reviews: {data['stats']['total_reviews']}

{data['summary']}

Statistics:
- Rating Distribution: {data['stats']['rating_distribution']}
- Companies Offering: {data['stats']['companies']}"""
        
        metadata = {
            "type": "product_summary",
            "product": product,
            "avg_rating": data['stats']['avg_rating'],
            "total_reviews": data['stats']['total_reviews']
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

def create_and_save_summaries(mistral_api_key):
    """Main function to create and save summaries and embeddings"""
    company_summaries = {}
    product_summaries = {}
    
    # Check if summaries already exist
    if os.path.exists('data/summaries/company_summaries.json') and os.path.exists('data/summaries/product_summaries.json'):
        print("Loading existing summaries...")
        with open('data/summaries/company_summaries.json', 'r') as f:
            company_summaries = json.load(f)
        with open('data/summaries/product_summaries.json', 'r') as f:
            product_summaries = json.load(f)
        print("Loaded existing summaries successfully!")
    else:
        # Initialize Mistral LLM
        llm = ChatMistralAI(
            temperature=0.7,
            mistral_api_key=mistral_api_key,
            model="mistral-medium"
        )
        
        # Load data
        df = load_data()
        
        # Generate summaries
        company_summaries = generate_company_summaries(df, llm)
        product_summaries = generate_product_summaries(df, llm)
        
        # Save raw summaries
        os.makedirs('data/summaries', exist_ok=True)
        with open('data/summaries/company_summaries.json', 'w') as f:
            json.dump(company_summaries, f, indent=2)
        with open('data/summaries/product_summaries.json', 'w') as f:
            json.dump(product_summaries, f, indent=2)
        print("\nSaved raw summaries to data/summaries/")
    
    # Create documents for embedding
    print("\nCreating documents for embedding...")
    documents = create_summary_documents(company_summaries, product_summaries)
    
    # Initialize local embeddings model with MPS acceleration
    print("\nInitializing local embeddings model...")
    import torch
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS acceleration")
    else:
        device = 'cpu'
        print("Using CPU")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    
    # Create and save vector store
    print("Creating vector store...")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    cache_dir = "data/vector_store_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "faiss_index")
    
    print(f"Saving vector store to {cache_path}...")
    vector_store.save_local(cache_path)
    print("Vector store saved successfully!")
    
    return company_summaries, product_summaries

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate_summaries.py MISTRAL_API_KEY")
        sys.exit(1)
    
    mistral_api_key = sys.argv[1]
    create_and_save_summaries(mistral_api_key) 