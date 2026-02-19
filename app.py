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
        # Save uploaded files temporarily
        temp_paths = []
        for uploaded_file in uploaded_files:
            temp_path = f"./data/{uploaded_file.name}"
            os.makedirs("./data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_paths.append(temp_path)
        
        # Load documents
        with st.spinner("Loading documents..."):
            documents = load_multiple_documents(temp_paths)
        
        # Split documents
        with st.spinner("Processing documents..."):
            chunks = split_documents(documents)
        
        # Create vector store
        with st.spinner("Creating knowledge base..."):
            vectorstore_manager = VectorStoreManager()
            vectorstore_manager.create_vectorstore(chunks)
        
        # Initialize RAG pipeline
        with st.spinner("Setting up AI..."):
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
            if st.button("Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
        
        st.divider()
        
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded!")
            if st.button("Clear All"):
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
