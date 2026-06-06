from pathlib import Path

from langchain_community.document_loaders import (DirectoryLoader, TextLoader,)

from app.config.settings import KNOWLEDGE_BASE_DIR
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """
    Load markdown documents from knowledge base.
    """

    def load_documents(self):
        documents = []

        folders = [
            folder
            for folder in KNOWLEDGE_BASE_DIR.iterdir()
            if folder.is_dir()
        ]

        logger.info(
            f"Found {len(folders)} document categories"
        )

        for folder in folders:

            doc_type = folder.name

            loader = DirectoryLoader(
                path=str(folder),
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={
                    "encoding": "utf-8"
                }
            )

            folder_docs = loader.load()

            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type

            documents.extend(folder_docs)

            logger.info(
                f"Loaded {len(folder_docs)} files from '{doc_type}'"
            )

        logger.info(
            f"Total documents loaded: {len(documents)}"
        )

        return documents