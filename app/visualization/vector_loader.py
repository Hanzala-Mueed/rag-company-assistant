import numpy as np

from app.vectorstore.chroma_store import get_vectorstore


def load_vectors():

    vectorstore = get_vectorstore()

    collection = vectorstore._collection

    data = collection.get(
        include=[
            "embeddings",
            "documents",
            "metadatas"
        ]
    )

    vectors = np.array(
        data["embeddings"]
    )

    documents = data["documents"]

    metadatas = data["metadatas"]

    return (
        vectors,
        documents,
        metadatas
    )