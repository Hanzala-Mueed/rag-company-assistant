from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import (CHUNK_SIZE, CHUNK_OVERLAP,)

from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """
    Split documents into overlapping chunks.
    """

    def __init__(self):

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        )

    def create_chunks(self, documents):

        chunks = self.splitter.split_documents(
            documents
        )

        logger.info(
            f"Created {len(chunks)} chunks"
        )

        return chunks