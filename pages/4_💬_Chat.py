import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate
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
            This will create the vector store using local embeddings.
            """)
            return None
            
        # Clear any existing caches
        import gc
        gc.collect()
        
        # Force CPU usage for more stable performance
        device = 'cpu'
        st.info("Using CPU for embeddings")
        
        try:
            # Initialize embeddings separately first
            with st.spinner("Initializing embeddings model..."):
                st.text("Loading sentence transformer model...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={
                        'device': device,
                    }
                )
                st.text("Embeddings model loaded successfully")
        except Exception as e:
            st.error(f"Failed to load embeddings model: {str(e)}")
            return None
            
        # Load the vector store with detailed progress
        with st.spinner("Loading vector store..."):
            st.text("Starting FAISS index load...")
            import time
            start_time = time.time()
            
            try:
                vector_store = FAISS.load_local(
                    cache_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                
                load_time = time.time() - start_time
                st.text(f"FAISS index loaded in {load_time:.2f} seconds")
                
                # Verify the vector store is working
                st.text("Verifying vector store...")
                test_query = "test"
                vector_store.similarity_search(test_query, k=1)
                st.text("Vector store verification successful")
                
            except Exception as e:
                st.error(f"Failed to load FAISS index: {str(e)}")
                return None
            
        st.success("Successfully loaded vector store!")
        return vector_store
        
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        st.error("""
        There might be an issue with the vector store cache. Try regenerating it:
        1. Delete the data/vector_store_cache folder
        2. Run: python utils/generate_embeddings.py
        """)
        return None

def get_conversation_chain(vector_store, mistral_api_key):
    """Create conversation chain with RAG using Mistral"""
    try:
        llm = ChatMistralAI(
            temperature=0.3,  # Lower temperature for more focused responses
            mistral_api_key=mistral_api_key,
            model="mistral-medium"
        )
        
        # Custom prompt for summary-based analysis
        prompt_template = """<system>You are an AI assistant helping analyze insurance company reviews. You must follow these rules strictly:
1. NEVER fabricate questions or conversations
2. ONLY respond to the current question
3. IGNORE any questions in the context that weren't explicitly asked by the user
4. If the context contains no relevant summaries, say "I don't have any relevant summary data to answer that question."
5. NEVER make up data or statistics
</system>

<context>
{context}
</context>

<chat_history>
{chat_history}
</chat_history>

<user>
{question}
</user>

<assistant>
Let me help you with that question."""
        
        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": None
                }
            ),
            combine_docs_chain_kwargs={
                "prompt": PROMPT
            },
            return_source_documents=True,
            verbose=True,
            max_tokens_limit=2000
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
if 'conversation' not in st.session_state:
    st.session_state.conversation = None

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
    message = st.chat_input("Ask me anything about the customer reviews...")
    
    if message:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": message})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(message)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Get conversation chain
                if st.session_state.conversation is None:
                    st.session_state.conversation = get_conversation_chain(st.session_state.vector_store, mistral_api_key)
                
                if st.session_state.conversation is None:
                    st.session_state.messages.pop()  # Remove user message if error occurs
                else:
                    # Get response from conversation chain
                    response = st.session_state.conversation({
                        "question": message,
                        "chat_history": st.session_state.chat_history  # Pass the tuples directly
                    })
                    
                    # Update chat history with new tuple
                    st.session_state.chat_history.append((message, response["answer"]))
                    
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
    st.session_state.chat_history = []
    st.session_state.conversation = None  # Force recreation of conversation chain
    st.rerun() 