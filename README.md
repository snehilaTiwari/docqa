# DocuQA - AI Document Question Answering System

An AI-powered system that allows users to upload documents (PDF, TXT, CSV) and ask questions in plain language. The system retrieves relevant information and generates accurate answers using Retrieval-Augmented Generation (RAG).

## Features

- 📄 **Multi-format Support**: PDF, TXT, CSV files
- 🤖 **AI-Powered Q&A**: Natural language question answering
- 💾 **Vector Storage**: ChromaDB for efficient similarity search
- 🎨 **User-Friendly UI**: Built with Streamlit

## Tech Stack

- **Python** - Core language
- **LangChain** - LLM framework & RAG pipeline
- **ChromaDB** - Vector database
- **HuggingFace** - Free LLM inference (Phi-3, Mistral, etc.)
- **Sentence-Transformers** - Free embeddings
- **Streamlit** - Web UI

## Installation

1. Clone and setup:
```bash
git clone <repo-url>
cd docqa
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

2. (Optional) Get HuggingFace token for higher rate limits:
   - https://huggingface.co/settings/tokens

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open the application in your browser (usually http://localhost:8501)
2. Upload documents via the sidebar (PDF, TXT, or CSV)
3. Click "Process Documents" to create the knowledge base
4. Ask questions about your documents in the chat input

## Cost

**Free!** Uses HuggingFace Inference API free tier.

## Project Structure

```
docqa/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Document loading utilities
│   ├── text_splitter.py      # Text chunking
│   ├── vector_store.py       # ChromaDB management
│   └── rag_pipeline.py       # RAG pipeline
└── data/                     # Data storage (created at runtime)
```

## License

MIT
```

## Usage

1. Open the application in your browser (usually http://localhost:8501)
2. Upload documents via the sidebar (PDF, TXT, or CSV)
3. Click "Process Documents" to create the knowledge base
4. Ask questions about your documents in the chat input

## Project Structure

```
docqa/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Document loading utilities
│   ├── text_splitter.py      # Text chunking
│   ├── vector_store.py       # ChromaDB management
│   └── rag_pipeline.py       # RAG pipeline
└── data/                     # Data storage (created at runtime)
```

## Cost

**$0** - Everything runs locally on your machine!

## License

MIT

## Author

[Snehi] - [snehi@example.com]
