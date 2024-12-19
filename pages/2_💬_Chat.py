import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
import os
import glob
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

st.title('💬 Chat Interface')

# Get OpenAI API key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def load_and_process_data():
    """Load data and create vector store"""
    data_path = 'TraductionAvisClients'
    all_files = glob.glob(os.path.join(data_path, '*.xlsx'))
    dfs = []
    
    for file in all_files:
        df = pd.read_excel(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Prepare documents for vectorization
    documents = []
    for _, row in combined_df.iterrows():
        doc_text = f"Review for {row['assureur']} about {row['produit']} with rating {row['note']}: {row['avis']}"
        documents.append(doc_text)
    
    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents(documents)
    
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
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

# Initialize vector store if not already done
if st.session_state.vector_store is None and openai_api_key:
    with st.spinner("Initializing the knowledge base..."):
        st.session_state.vector_store = initialize_vector_store()

# Chat interface
st.markdown("""
This chat interface allows you to ask questions about customer reviews.
The AI will analyze the reviews database to provide relevant answers.
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