import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
import numpy as np
import os
import glob
import sys
import pickle

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rag_utils import generate_all_summaries, prepare_rag_documents

st.set_page_config(
    page_title="Chat Interface",
    page_icon="💬",
    layout="wide"
)

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'summaries' not in st.session_state:
    st.session_state.summaries = None

st.title('💬 Enhanced Chat Interface')

# Get OpenAI API key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def load_and_process_data():
    """Load data and create vector store"""
    if st.session_state.df is None:
        data_path = 'TraductionAvisClients'
        all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
        dfs = []
        
        for file in all_files:
            df = pd.read_excel(file)
            dfs.append(df)
        
        st.session_state.df = pd.concat(dfs, ignore_index=True)
    
    # Try to load pre-generated summaries
    if st.session_state.summaries is None:
        try:
            with open('data/summaries.pkl', 'rb') as f:
                st.session_state.summaries = pickle.load(f)
        except FileNotFoundError:
            # Generate new summaries
            with st.spinner("Generating summaries for all companies and products..."):
                st.session_state.summaries = generate_all_summaries(st.session_state.df, openai_api_key)
                # Save summaries for future use
                os.makedirs('data', exist_ok=True)
                with open('data/summaries.pkl', 'wb') as f:
                    pickle.dump(st.session_state.summaries, f)
    
    # Prepare documents including summaries
    documents = prepare_rag_documents(
        st.session_state.df,
        st.session_state.summaries,
        openai_api_key
    )
    
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    return docs

def initialize_vector_store():
    """Initialize FAISS vector store with embeddings"""
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
        return None
    
    try:
        docs = load_and_process_data()
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None

def get_conversation_chain(vector_store):
    """Create conversation chain with RAG"""
    llm = ChatOpenAI(
        temperature=0.7,
        api_key=openai_api_key,
        model_name='gpt-3.5-turbo'
    )
    
    # Custom prompt to handle both summaries and reviews
    prompt_template = """You are an AI assistant helping with insurance company reviews.
    Use the following pieces of context to answer the question. The context includes both high-level summaries
    and specific reviews. When answering:
    1. Start with relevant high-level insights from summaries if available
    2. Support your points with specific examples from reviews
    3. Be balanced and objective in your assessment
    
    Context: {context}
    
    Chat History: {chat_history}
    
    Question: {question}
    
    Answer:"""
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={
                "k": 8,  # Increased to get more context
                "filter": None  # Can be used to filter by metadata
            }
        ),
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

# Initialize vector store if not already done
if st.session_state.vector_store is None and openai_api_key:
    with st.spinner("Initializing the knowledge base..."):
        st.session_state.vector_store = initialize_vector_store()

# Sidebar for exploring summaries
st.sidebar.header("Pre-generated Insights")
if st.session_state.summaries:
    summary_type = st.sidebar.radio(
        "View summaries by:",
        ["Companies", "Products"]
    )
    
    if summary_type == "Companies":
        company = st.sidebar.selectbox(
            "Select a company",
            options=sorted(st.session_state.summaries['company_summaries'].keys())
        )
        if company:
            st.sidebar.markdown(f"### Summary for {company}")
            st.sidebar.write(st.session_state.summaries['company_summaries'][company])
    else:
        product = st.sidebar.selectbox(
            "Select a product type",
            options=sorted(st.session_state.summaries['product_summaries'].keys())
        )
        if product:
            st.sidebar.markdown(f"### Summary for {product}")
            st.sidebar.write(st.session_state.summaries['product_summaries'][product])

# Main chat interface
st.markdown("""
This enhanced chat interface allows you to:
1. Ask questions about customer reviews
2. Get company-specific insights
3. Compare different insurance providers
4. Analyze sentiment and trends
5. Explore pre-generated summaries
""")

# Chat input
user_question = st.text_input("Ask a question about the customer reviews:", key="user_input")

if user_question and openai_api_key and st.session_state.vector_store:
    try:
        # Get conversation chain
        conversation = get_conversation_chain(st.session_state.vector_store)
        
        # Get response from conversation chain
        response = conversation({
            "question": user_question,
            "chat_history": [(msg[0], msg[1]) for msg in st.session_state.chat_history]
        })
        
        # Add to chat history
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Assistant", response["answer"]))
        
        # Display source documents with metadata
        with st.expander("View Sources"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Source {i+1} ({doc.metadata.get('type', 'unknown')}):**")
                st.write(doc.page_content)
                if doc.metadata.get('type') == 'review':
                    st.write(f"Company: {doc.metadata['company']}")
                    st.write(f"Product: {doc.metadata['product']}")
                    st.write(f"Rating: {doc.metadata['rating']}")
                st.write("---")
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

# Display chat history
st.subheader("Chat History")
for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Assistant:** {message}")

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun() 