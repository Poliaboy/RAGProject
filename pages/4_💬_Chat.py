import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_mistralai.chat_models import ChatMistralAI
import pandas as pd
import numpy as np
import os
import glob
import sys
from langchain.schema import Document

def initialize_vector_store():
    """Initialize FAISS vector store with embeddings"""
    try:
        # Load cached vector store with local embeddings
        cache_dir = "data/vector_store_cache"
        cache_path = os.path.join(cache_dir, "faiss_index")
        
        if not os.path.exists(cache_path):
            st.error("""
            Vector store not found! Please run the following command first:
            ```
            python utils/generate_embeddings.py
            ```
            This will create the vector store using local embeddings with MPS acceleration.
            """)
            return None
        
        # Initialize local embeddings model with MPS acceleration
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
            st.success("Using MPS (Metal Performance Shaders) acceleration")
        else:
            device = 'cpu'
            st.info("MPS not available, falling back to CPU")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        
        # Load the vector store
        vector_store = FAISS.load_local(
            cache_path,
            embeddings,
            allow_dangerous_deserialization=True  # Safe since we created the cache locally
        )
        st.success("Successfully loaded vector store with local embeddings!")
        return vector_store
        
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def get_conversation_chain(vector_store, mistral_api_key):
    """Create conversation chain with RAG using Mistral"""
    try:
        llm = ChatMistralAI(
            temperature=0.7,
            mistral_api_key=mistral_api_key,
            model="mistral-medium"  # Using Mistral's medium model for better performance
        )
        
        # Custom prompt for summary-based analysis
        prompt_template = """You are an AI assistant helping analyze insurance company reviews.
        Use the following summaries to answer the question. The summaries include:
        - Company summaries with overall ratings, trends, and key insights
        - Product summaries with market analysis and customer satisfaction data
        
        When answering:
        1. Use the statistical data provided in the summaries
        2. Reference specific companies or products when relevant
        3. Be balanced and objective in your assessment
        4. Support your points with the provided metrics
        
        You can analyze:
        - Company performance and trends
        - Product satisfaction across companies
        - Market patterns and insights
        - Customer satisfaction metrics
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        
        Answer:"""
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_kwargs={
                    "k": 3,  # Retrieve top 3 most relevant summaries
                    "filter": None
                }
            ),
            return_source_documents=True,
            verbose=True
        )
        
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.set_page_config(
    page_title="Chat Interface",
    page_icon="💬",
    layout="wide"
)

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    mistral_api_key = st.text_input("Mistral API Key", type="password", key="api_key")
    if not mistral_api_key:
        st.warning('Please enter your Mistral API key to continue!', icon='⚠️')

# Main chat interface
st.title('💬 Enhanced Chat Interface')

# Add explanation of what we're doing
st.markdown("""
### Intelligent Insurance Market Analysis
This interface provides insights from a comprehensive analysis of insurance reviews:

1. **What insights can you get?**
   - Company performance analysis
   - Product satisfaction trends
   - Market-wide patterns
   - Statistical insights from customer feedback

2. **Key Features:**
   - **Company Analysis**: Get insights about specific insurance companies
   - **Product Analysis**: Understand performance across insurance types
   - **Market Trends**: See patterns across the industry
   - **Statistical Backing**: All insights supported by real data
   - **Powered by Mistral AI**: High-quality analysis and insights

3. **Example Questions:**
   - "How does [company] perform across different products?"
   - "What are the market trends in auto insurance?"
   - "Which companies have the highest customer satisfaction?"
   - "Compare the performance of different insurance types"
   - "What are the common strengths of top-rated companies?"
""")

# Initialize vector store if not already initialized
if st.session_state.vector_store is None:
    with st.spinner("Initializing the knowledge base..."):
        st.session_state.vector_store = initialize_vector_store()

# Chat interface
st.divider()
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if mistral_api_key and st.session_state.vector_store:
    prompt = st.chat_input("Ask me anything about the customer reviews...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Get conversation chain
                conversation = get_conversation_chain(st.session_state.vector_store, mistral_api_key)
                
                if conversation is None:
                    st.session_state.messages.pop()  # Remove user message if error occurs
                else:
                    # Get response from conversation chain
                    response = conversation({
                        "question": prompt,
                        "chat_history": [(msg["content"], "") for msg in st.session_state.messages[:-1] if msg["role"] == "user"]
                    })
                    
                    # Display response
                    message_placeholder.markdown(response["answer"])
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    
                    # Display source documents in expander
                    with st.expander("View Source Summary", expanded=False):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**{doc.metadata.get('type', 'unknown').replace('_', ' ').title()}**")
                            st.write(doc.page_content)
                            if doc.metadata.get('type') == 'company_summary':
                                st.metric("Average Rating", f"{doc.metadata['avg_rating']:.2f} ⭐")
                                st.metric("Total Reviews", doc.metadata['total_reviews'])
                            elif doc.metadata.get('type') == 'product_summary':
                                st.metric("Average Rating", f"{doc.metadata['avg_rating']:.2f} ⭐")
                                st.metric("Total Reviews", doc.metadata['total_reviews'])
                            st.divider()
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.session_state.messages.pop()  # Remove user message if error occurs
else:
    st.info("Please provide your Mistral API key in the sidebar to start chatting!", icon="ℹ️")

# Add a clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun() 