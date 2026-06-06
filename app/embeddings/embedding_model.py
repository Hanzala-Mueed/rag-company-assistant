from langchain_huggingface import (HuggingFaceEmbeddings)

from app.config.settings import (EMBEDDING_MODEL)


def get_embedding_model():
    """
    Return embedding model instance.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )