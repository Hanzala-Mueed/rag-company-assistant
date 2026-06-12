# RAG Company Assistant

An advanced Retrieval-Augmented Generation (RAG) application that enables users to query organizational knowledge using natural language. The system combines semantic search, vector databases, local LLMs, evaluation metrics, and interactive visualizations to provide accurate and explainable responses.

---

## Overview

Enterprise RAG Assistant is a production-style Retrieval-Augmented Generation system built with LangChain, ChromaDB, Sentence Transformers, Ollama, and Gradio.

The application allows users to:

* Query enterprise knowledge bases using natural language
* Retrieve semantically relevant document chunks
* Generate context-aware responses using a local LLM
* Visualize vector embeddings in 2D and 3D
* Evaluate retrieval quality using information retrieval metrics
* Evaluate generated answers against reference responses
* Analyze vector database statistics through an interactive dashboard

The project demonstrates the complete RAG pipeline from document ingestion to evaluation and visualization.

---

## Features

### Knowledge Base Management

* Markdown document ingestion
* Automatic metadata tagging
* Multi-category document organization
* Support for structured enterprise knowledge

### Chunking Pipeline

* Recursive Character Text Splitting
* Configurable chunk size
* Configurable overlap
* Metadata preservation across chunks

### Vector Database

* ChromaDB integration
* Persistent vector storage
* Fast similarity search
* Metadata filtering support

### Semantic Search

* Sentence Transformer embeddings
* Dense vector retrieval
* Context aggregation
* Top-k retrieval strategy

### Local LLM Integration

* Ollama-powered inference
* Fully local execution
* No external API costs
* Privacy-friendly deployment

### Conversational RAG

* Multi-turn conversation support
* Query expansion from chat history
* Context-aware responses

### Retrieval Evaluation

* Mean Reciprocal Rank (MRR)
* Normalized Discounted Cumulative Gain (nDCG)
* Keyword Coverage Analysis

### Answer Evaluation

* Semantic Similarity Scoring
* Reference Answer Comparison
* Accuracy Assessment
* Completeness Assessment
* Relevance Assessment

### Vector Visualization

* t-SNE dimensionality reduction
* Interactive 2D visualization
* Interactive 3D visualization
* Category-aware clustering

### Analytics Dashboard

* Total vector count
* Chunk distribution
* Category statistics
* Knowledge base insights

---

## Technology Stack

### Backend

* Python 3.11+
* LangChain
* ChromaDB
* Ollama

### Machine Learning

* Sentence Transformers
* all-MiniLM-L6-v2
* Scikit-Learn
* NumPy

### Visualization

* Plotly
* t-SNE

### Frontend

* Gradio

### Data Processing

* Pandas

---

## Project Architecture

```text
User Question
       │
       ▼
 Query Expansion
       │
       ▼
 Semantic Retrieval
       │
       ▼
 Relevant Chunks
       │
       ▼
 Context Construction
       │
       ▼
      LLM
       │
       ▼
 Generated Answer
       │
       ├────────► Retrieval Evaluation
       │
       ├────────► Answer Evaluation
       │
       └────────► Analytics Dashboard
```

---

## Project Structure

```text
rag-company-assistant/
│
├── app/
│   │
│   ├── chunking/
│   │   └── text_splitter.py
│   │
│   ├── config/
│   │   ├── settings.py
│   │   ├── prompts.py
│   │   └── logging_config.py
│   │
│   ├── embeddings/
│   │   └── embedding_model.py
│   │
│   ├── evaluation/
│   │   ├── answer_metrics.py
│   │   ├── evaluator.py
│   │   ├── prompts.py
│   │   ├── retrieval_metrics.py
│   │   ├── judge_models.py
│   │   └── test_cases.py
│   │
│   ├── loaders/
│   │   └── document_loader.py
│   │
│   ├── llm/
│   │   ├── ollama_llm.py
│   │   └── huggingface_llm.py
│   │
│   ├── rag/
│   │   ├── rag_pipeline.py
│   │   ├── context_builder.py
│   │   └── prompt_builder.py
│   │
│   ├── retrieval/
│   │   ├── query_builder.py
│   │   └── retriever.py
│   │
│   ├── services/
│   │   ├── chat_service.py
│   │   └── ingest_service.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   └── helpers.py
│   │
│   ├── vectorstore/
│   │   └── chroma_store.py
│   │
│   └── visualization/
│        ├── vector_visualizer.py
│        ├── tsne_visualizer.py
│        └── vector_loader.py
│
├── knowledge_base/
│   ├── company/
│   ├── employees/
│   ├── products/
│   └── contracts/
│
├── scripts/
│   ├── ingest_documents.py
│   ├── run_answer_eval.py
│   ├── run_retrieval_eval.py
│   └── check_metadata.py
│
├── ui/
│   └── gradio_app.py
│
├── vector_db/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd rag-company-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download and install Ollama:

https://ollama.com

Verify installation:

```bash
ollama --version
```

Pull the model used by the project:

```bash
ollama pull llama3
```

Run the model:

```bash
ollama run llama3
```

---

## Build Vector Database

After adding documents to the knowledge base:

```bash
python -m scripts.ingest_documents
```

This step will:

* Load markdown documents
* Generate embeddings
* Create chunks
* Store vectors in ChromaDB

---

## Run Retrieval Evaluation

```bash
python -m scripts.run_retrieval_eval
```

Metrics:

* MRR
* nDCG
* Keyword Coverage

---

## Run Answer Evaluation

```bash
python -m scripts.run_answer_eval
```

Evaluates generated answers against reference responses.

---

## Launch Application

```bash
python -m ui.gradio_app
```

Open:

```text
http://localhost:7860
```

---

## Using the Application

### AI Assistant

Ask questions related to your knowledge base.

Examples:

* Who founded the company?
* What products are offered?
* What employee benefits are available?
* What contracts are currently active?

### Vector Visualization

Generate:

* 2D embedding visualization
* 3D embedding visualization

### Retrieval Evaluation

Evaluate retrieval quality using:

* MRR
* nDCG
* Keyword Coverage

### Answer Evaluation

Compare generated answers with reference answers and calculate:

* Semantic Similarity
* Accuracy
* Completeness
* Relevance

### Analytics

View:

* Total vectors
* Chunk counts
* Category distribution

---

## Future Improvements

* Hybrid Search (BM25 + Dense Retrieval)
* Reranking Models
* PDF Ingestion
* Multi-file Upload Support
* Source Citations
* Conversation Memory Storage
* User Authentication
* REST API Integration
* Docker Deployment
* Kubernetes Deployment

---

## Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* LangChain
* ChromaDB
* Vector Search
* Embedding Models
* Ollama
* Local LLM Deployment
* Information Retrieval Metrics
* Semantic Similarity Evaluation
* Gradio UI Development
* Production-style AI System Design

---

## License

This project is intended for educational and portfolio purposes.
