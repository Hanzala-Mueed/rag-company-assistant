import os

from langchain_chroma import Chroma

from app.config.settings import (VECTOR_DB_DIR, COLLECTION_NAME,)

from app.embeddings.embedding_model import get_embedding_model

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaStore:

    def __init__(self):

        self.embeddings = get_embedding_model()

    def create_vectorstore(self, chunks):

        if VECTOR_DB_DIR.exists():

            try:
                Chroma(
                    persist_directory=str(VECTOR_DB_DIR),
                    embedding_function=self.embeddings
                ).delete_collection()

                logger.info("Previous collection deleted")

            except Exception:
                pass

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(VECTOR_DB_DIR),
            collection_name=COLLECTION_NAME
        )

        count = vectorstore._collection.count()

        logger.info(f"Stored {count} vectors")

        return vectorstore

    def load_vectorstore(self):
        """
        Load existing Chroma database.
        """

        return Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )

    def get_collection_data(self):
        """
        Return embeddings, metadata and documents
        for visualization.
        """

        vectorstore = self.load_vectorstore()

        collection = vectorstore._collection

        data = collection.get(
            include=[
                "embeddings",
                "documents",
                "metadatas"
            ]
        )

        return {
            "embeddings": data["embeddings"],
            "documents": data["documents"],
            "metadatas": data["metadatas"]
        }
    

    def load_vectorstore(self):

        return Chroma(
            persist_directory=str(
                VECTOR_DB_DIR
            ),
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )