import numpy as np

from sklearn.manifold import TSNE

import plotly.graph_objects as go

from app.vectorstore.chroma_store import ChromaStore

from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorVisualizer:

    def __init__(self):

        self.store = ChromaStore()

    def get_embeddings(self):

        vectorstore = self.store.load_vectorstore()

        collection = vectorstore._collection

        result = collection.get(
            include=[
                "embeddings",
                "documents",
                "metadatas",
            ]
        )

        return (
            np.array(result["embeddings"]),
            result["documents"],
            result["metadatas"]
        )
    
    def create_2d_plot(self):

        vectors, documents, metadata = (
            self.get_embeddings()
        )

        tsne = TSNE(
            n_components=2,
            random_state=42
        )

        reduced = tsne.fit_transform(vectors)

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode="markers",
                    text=[
                        doc[:100]
                        for doc in documents
                    ],
                    hoverinfo="text"
                )
            ]
        )

        fig.update_layout(
            title="Vector Space (2D)"
        )

        return fig
    
    def create_3d_plot(self):

        vectors, documents, metadata = (
            self.get_embeddings()
        )

        tsne = TSNE(
            n_components=3,
            random_state=42
        )

        reduced = tsne.fit_transform(vectors)

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode="markers",
                    text=[
                        doc[:100]
                        for doc in documents
                    ],
                    hoverinfo="text"
                )
            ]
        )

        fig.update_layout(
            title="Vector Space (3D)"
        )

        return fig