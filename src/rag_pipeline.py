"""RAG pipeline using HuggingFace via OpenAI-compatible API."""
import os
import logging
from openai import OpenAI
from typing import List, Optional, Dict
from langchain.schema import Document

from src.vector_store import VectorStoreManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Handles question answering using RAG with HuggingFace Inference API."""
    
    def __init__(
        self,
        vectorstore_manager: VectorStoreManager,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct:novita",
        hf_token: Optional[str] = None,
        temperature: float = 0.0
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vectorstore_manager: VectorStoreManager instance
            model_name: HuggingFace model to use
            hf_token: HuggingFace token
            temperature: Temperature for generation
        """
        self.vectorstore_manager = vectorstore_manager
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.temperature = temperature
        
        # Use HuggingFace router with OpenAI client
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.hf_token
        )
    
    def _get_relevant_docs(self, question: str, k: int = 8) -> List[Document]:
        """Get relevant documents from vector store."""
        if self.vectorstore_manager.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore_manager.similarity_search(question, k=k)
    
    def _create_prompt(self, question: str, docs: List[Document]) -> str:
        """Create prompt from question and documents."""
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Based on the following context, answer the question. If you cannot find the answer in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace via OpenAI-compatible API."""
        logger.info(f"Calling HF with model: {self.model_name}")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=512
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from model")
        logger.info(f"Response received: {content[:100]}...")
        return content
    
    def ask(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None
    ) -> Dict[str, any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            chat_history: List of (question, answer) tuples (unused in this simple version)
            
        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Processing question: {question}")
        
        # Get relevant documents
        docs = self._get_relevant_docs(question)
        logger.info(f"Found {len(docs)} relevant documents")
        
        if not docs:
            return {
                "answer": "No relevant documents found.",
                "source_documents": [],
                "chat_history": chat_history or []
            }
        
        # Create prompt
        prompt = self._create_prompt(question, docs)
        
        # Get answer from HuggingFace
        try:
            answer = self._call_huggingface(prompt)
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "source_documents": docs,
                "chat_history": chat_history or []
            }
        
        return {
            "answer": answer,
            "source_documents": docs,
            "chat_history": chat_history or []
        }
    
    def get_sources_formatted(self, source_docs: List[Document]) -> str:
        """Format source documents for display."""
        sources = []
        for i, doc in enumerate(source_docs, 1):
            source_name = doc.metadata.get("source", f"Source {i}")
            page = doc.metadata.get("page", "")
            sources.append(f"{i}. {source_name} (Page {page})")
        return "\n".join(sources)
