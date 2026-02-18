"""RAG (Retrieval-Augmented Generation) pipeline for question answering."""
import os
from typing import List, Optional, Dict
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from src.vector_store import VectorStoreManager


# Default prompt template for question answering
QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the answer is not in the context, say "I don't have enough information to answer that"
- Be concise and accurate
- Cite the source when possible

Answer:""",
    input_variables=["context", "question"]
)


class RAGPipeline:
    """Handles question answering using RAG architecture."""
    
    def __init__(
        self,
        vectorstore_manager: VectorStoreManager,
        model_name: str = "microsoft/Phi-3-mini-128k-instruct",
        hf_token: Optional[str] = None,
        temperature: float = 0.0
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vectorstore_manager: VectorStoreManager instance
            model_name: HuggingFace model to use
            hf_token: HuggingFace token (optional, for rate limit increase)
            temperature: Temperature for generation
        """
        self.vectorstore_manager = vectorstore_manager
        self.llm = HuggingFaceHub(
            repo_id=model_name,
            task="text-generation",
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": 512,
                "top_p": 0.7,
            },
            huggingfacehub_api_token=hf_token or os.getenv("HF_TOKEN")
        )
        self.chat_model = ChatHuggingFace(llm=self.llm)
        self.qa_chain = None
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the retrieval chain."""
        if self.vectorstore_manager.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self.vectorstore_manager.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            ),
            return_source_documents=True
        )
    
    def ask(
        self,
        question: str,
        chat_history: list[tuple[str, str]] | None = None
    ) -> dict[str, any]:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            chat_history: List of (question, answer) tuples
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized")
        
        chat_history = chat_history or []
        result = self.qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        
        return {
            "answer": result["answer"],
            "source_documents": result.get("source_documents", []),
            "chat_history": chat_history + [(question, result["answer"])]
        }
    
    def get_sources_formatted(self, source_docs: List[Document]) -> str:
        """Format source documents for display."""
        sources = []
        for i, doc in enumerate(source_docs, 1):
            source_name = doc.metadata.get("source", f"Source {i}")
            page = doc.metadata.get("page", "")
            sources.append(f"{i}. {source_name} (Page {page})")
        return "\n".join(sources)
