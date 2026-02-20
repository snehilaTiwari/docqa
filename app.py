"""Streamlit web interface for the Document Q&A system."""
import streamlit as st
import os
import sys
from pathlib import Path
from typing import List

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_loader import load_multiple_documents
from src.text_splitter import split_documents
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline


# Page configuration
st.set_page_config(
    page_title="DocuQA - AI Document Assistant",
    page_icon="📄",
    layout="wide"
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "vectorstore_manager" not in st.session_state:
        st.session_state.vectorstore_manager = None
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False


def process_uploaded_files(uploaded_files: List) -> bool:
    """
    Process uploaded files and create vector store.
    
    Args:
        uploaded_files: List of Streamlit uploaded files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if vector store already exists
        vectorstore_manager = VectorStoreManager()
        
        # Try to load existing vector store
        with st.spinner("Checking for existing knowledge base..."):
            existing_store = vectorstore_manager.load_vectorstore()
        
        if existing_store:
            st.success("Found existing knowledge base! Loading...")
            rag_pipeline = RAGPipeline(vectorstore_manager)
            st.session_state.vectorstore_manager = vectorstore_manager
            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.documents_loaded = True
            st.session_state.chat_history = []
            return True
        
        # Save uploaded files temporarily
        temp_paths = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Loading file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            temp_path = f"./data/{uploaded_file.name}"
            os.makedirs("./data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_paths.append(temp_path)
            progress_bar.progress((i+1) / (len(uploaded_files) * 3))
        
        # Load documents
        status_text.text("Processing documents...")
        documents = load_multiple_documents(temp_paths)
        progress_bar.progress(2 / 3)
        
        # Split documents
        status_text.text("Creating embeddings...")
        chunks = split_documents(documents, chunk_size=1500, chunk_overlap=300)
        
        # Create vector store
        status_text.text("Building knowledge base (this may take a while)...")
        vectorstore_manager.create_vectorstore(chunks)
        progress_bar.progress(1.0)
        status_text.text("Done!")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(vectorstore_manager)
        
        # Store in session state
        st.session_state.vectorstore_manager = vectorstore_manager
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.documents_loaded = True
        st.session_state.chat_history = []
        
        # Clean up temp files
        for path in temp_paths:
            os.remove(path)
        
        return True
    
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False


def add_more_documents(uploaded_files: List) -> bool:
    """
    Add more documents to existing vector store.
    
    Args:
        uploaded_files: List of Streamlit uploaded files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        vectorstore_manager = st.session_state.vectorstore_manager
        
        # Save uploaded files temporarily
        temp_paths = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Loading file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            temp_path = f"./data/{uploaded_file.name}"
            os.makedirs("./data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_paths.append(temp_path)
            progress_bar.progress((i+1) / (len(uploaded_files) * 3))
        
        # Load documents
        status_text.text("Processing documents...")
        documents = load_multiple_documents(temp_paths)
        progress_bar.progress(2 / 3)
        
        # Split documents
        status_text.text("Creating embeddings...")
        chunks = split_documents(documents, chunk_size=1500, chunk_overlap=300)
        
        # Add to existing vector store
        status_text.text("Adding to knowledge base...")
        vectorstore_manager.add_documents(chunks)
        progress_bar.progress(1.0)
        status_text.text("Done!")
        
        # Clear chat history (optional - keeps context cleaner)
        st.session_state.chat_history = []
        
        # Clean up temp files
        for path in temp_paths:
            os.remove(path)
        
        st.success(f"Added {len(uploaded_files)} document(s) to knowledge base!")
        return True
    
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return False


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Header
    st.title("📄 DocuQA - AI Document Assistant")
    st.markdown("""
    Upload your documents and ask questions. The AI will answer based on the content.
    """)
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Documents")
        st.markdown("Supported: PDF, TXT, CSV")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.session_state.documents_loaded:
                if st.button("Add to Knowledge Base", type="primary"):
                    add_more_documents(uploaded_files)
            else:
                if st.button("Process Documents", type="primary"):
                    process_uploaded_files(uploaded_files)
        
        st.divider()
        
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded!")
            if st.button("Clear All & Rebuild"):
                import shutil
                if os.path.exists("./data/chroma_db"):
                    shutil.rmtree("./data/chroma_db")
                st.session_state.vectorstore_manager = None
                st.session_state.rag_pipeline = None
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.rerun()
    
    # Main content
    if not st.session_state.documents_loaded:
        st.info("👈 Please upload documents in the sidebar to get started!")
        
        # Show example
        st.markdown("### Example Use Cases")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📋 Legal** - Contract analysis")
        with col2:
            st.markdown("**💼 Business** - Report Q&A")
        with col3:
            st.markdown("**🔬 Research** - Paper summary")
    else:
        # Question input
        question = st.chat_input("Ask a question about your documents...")
        
        last_result = None
        
        if question:
            try:
                result = st.session_state.rag_pipeline.ask(
                    question,
                    st.session_state.chat_history
                )
                # Add to chat history
                st.session_state.chat_history.append((question, result["answer"]))
                last_result = result
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Display chat history
        st.markdown("### 💬 Conversation")
        
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
                
                # Show sources for last message
                if i == len(st.session_state.chat_history) - 1 and last_result:
                    with st.expander("📚 View Sources"):
                        for doc in last_result.get("source_documents", []):
                            source = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "")
                            st.markdown(f"**{source}** (Page {page})")
                            st.write(doc.page_content[:300] + "...")


if __name__ == "__main__":
    main()
