from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import pandas as pd

def generate_company_summary(reviews_df, company_name, openai_api_key):
    """Generate a comprehensive summary for a company"""
    company_reviews = reviews_df[reviews_df['assureur'] == company_name]
    if len(company_reviews) == 0:
        return None
    
    llm = ChatOpenAI(temperature=0.5, api_key=openai_api_key, model="gpt-4o-mini")
    
    prompt_template = """Write a comprehensive summary of the following insurance company reviews:
    {text}
    
    Focus on:
    1. Common themes in customer feedback
    2. Strengths and weaknesses
    3. Overall customer satisfaction
    4. Areas for improvement
    
    Summary:"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=prompt
    )
    
    # Prepare documents from reviews
    docs = [Document(page_content=review) for review in company_reviews['avis_en'].head(50)]
    summary = chain.run(docs)
    
    return summary

def generate_product_summary(reviews_df, product_type, openai_api_key):
    """Generate a summary for a specific product type"""
    product_reviews = reviews_df[reviews_df['produit'] == product_type]
    if len(product_reviews) == 0:
        return None
    
    llm = ChatOpenAI(temperature=0.5, api_key=openai_api_key)
    
    prompt_template = """Write a comprehensive summary of customer reviews for this insurance product type:
    {text}
    
    Focus on:
    1. Common customer needs and expectations
    2. Typical issues and concerns
    3. Overall satisfaction levels
    4. Price-value perception
    
    Summary:"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=prompt
    )
    
    docs = [Document(page_content=review) for review in product_reviews['avis_en'].head(50)]
    summary = chain.run(docs)
    
    return summary

def prepare_rag_documents(reviews_df, summaries_dict, openai_api_key):
    """Prepare documents for RAG including reviews and summaries"""
    documents = []
    
    # Add company summaries
    for company, summary in summaries_dict['company_summaries'].items():
        if summary:
            doc_text = f"COMPANY SUMMARY - {company}:\n{summary}"
            documents.append(Document(page_content=doc_text, metadata={'type': 'company_summary', 'company': company}))
    
    # Add product summaries
    for product, summary in summaries_dict['product_summaries'].items():
        if summary:
            doc_text = f"PRODUCT SUMMARY - {product}:\n{summary}"
            documents.append(Document(page_content=doc_text, metadata={'type': 'product_summary', 'product': product}))
    
    # Add individual reviews with metadata
    for _, row in reviews_df.iterrows():
        doc_text = f"Review for {row['assureur']} about {row['produit']} with rating {row['note']}: {row['avis_en']}"
        documents.append(Document(
            page_content=doc_text,
            metadata={
                'type': 'review',
                'company': row['assureur'],
                'product': row['produit'],
                'rating': row['note']
            }
        ))
    
    return documents

def generate_all_summaries(reviews_df, openai_api_key):
    """Generate summaries for all companies and products"""
    summaries = {
        'company_summaries': {},
        'product_summaries': {}
    }
    
    # Generate company summaries
    for company in reviews_df['assureur'].unique():
        summary = generate_company_summary(reviews_df, company, openai_api_key)
        summaries['company_summaries'][company] = summary
    
    # Generate product summaries
    for product in reviews_df['produit'].unique():
        summary = generate_product_summary(reviews_df, product, openai_api_key)
        summaries['product_summaries'][product] = summary
    
    return summaries 