"""Vector store and embedding management using ChromaDB."""
import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


class VectorStoreManager:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Where to store the vector database
            model_name: HuggingFace model for embeddings
        """
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create a vector store from documents.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            Chroma vector store
        """
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return self.vectorstore
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Load an existing vector store.
        
        Returns:
            Chroma vector store or None
        """
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self.vectorstore
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load or create first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load or create first.")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def add_documents(self, documents: List[Document]) -> Chroma:
        """
        Add documents to existing vector store.

        Args:
            documents: List of Document objects to add
            
        Returns:
            Updated Chroma vector store
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load or create first.")
        
        self.vectorstore.add_documents(documents=documents)
        return self.vectorstore
