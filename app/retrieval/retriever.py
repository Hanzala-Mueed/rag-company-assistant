from langchain_chroma import Chroma

from app.config.settings import VECTOR_DB_DIR, COLLECTION_NAME, RETRIEVAL_K

from app.embeddings.embedding_model import get_embedding_model

from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentRetriever:
    """
    Retrieve relevant chunks from Chroma.
    """

    def __init__(self):

        self.vectorstore = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedding_model()
        )

    def retrieve(self, query: str):

        logger.info(
            f"Retrieving context for query: {query}"
        )

        docs = self.vectorstore.similarity_search(
            query=query,
            k=RETRIEVAL_K
        )

        logger.info(
            f"Retrieved {len(docs)} chunks"
        )

        return docs