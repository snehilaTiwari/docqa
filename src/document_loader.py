"""Document loading utilities for various file types."""
from typing import List
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    CSVLoader
)
from langchain.schema import Document


def load_document(file_path: str) -> List[Document]:
    """
    Load a document from a file path.
    Supports PDF, TXT, and CSV files.
    
    Args:
        file_path: Path to the document
        
    Returns:
        List of Document objects
    """
    if file_path.lower().endswith('.pdf'):
        loader = PDFPlumberLoader(file_path)
    elif file_path.lower().endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_path.lower().endswith('.csv'):
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return loader.load()


def load_multiple_documents(file_paths: List[str]) -> List[Document]:
    """
    Load multiple documents.
    
    Args:
        file_paths: List of paths to documents
        
    Returns:
        Combined list of Document objects
    """
    documents = []
    for path in file_paths:
        try:
            docs = load_document(path)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return documents
