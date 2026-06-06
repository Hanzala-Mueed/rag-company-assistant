from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Data directories
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
LOG_DIR = BASE_DIR / "logs"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

# Retrieval configuration
RETRIEVAL_K = 10

# Chroma collection
COLLECTION_NAME = "company_documents"

# Visualization
TSNE_RANDOM_STATE = 42