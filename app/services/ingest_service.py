from app.loaders.document_loader import DocumentLoader

from app.chunking.text_splitter import TextChunker

from app.vectorstore.chroma_store import ChromaStore

from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestService:

    def run(self):

        logger.info(
            "Starting ingestion pipeline"
        )

        documents = (
            DocumentLoader()
            .load_documents()
        )

        chunks = (
            TextChunker()
            .create_chunks(documents)
        )

        vectorstore = (
            ChromaStore()
            .create_vectorstore(chunks)
        )

        logger.info(
            "Ingestion completed"
        )

        return vectorstore