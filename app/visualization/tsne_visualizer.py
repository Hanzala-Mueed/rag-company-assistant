import numpy as np

from sklearn.manifold import TSNE

from app.vectorstore.chroma_store import ChromaStore

from app.utils.logger import get_logger

logger = get_logger(__name__)


class TSNEVisualizer:

    def __init__(self):

        self.store = ChromaStore()

    def generate_2d(self):

        data = self.store.get_collection_data()

        vectors = np.array(
            data["embeddings"]
        )

        tsne = TSNE(
            n_components=2,
            random_state=42
        )

        reduced = tsne.fit_transform(
            vectors
        )

        logger.info(
            "Generated 2D t-SNE projection"
        )

        return reduced, data

    def generate_3d(self):

        data = self.store.get_collection_data()

        vectors = np.array(
            data["embeddings"]
        )

        tsne = TSNE(
            n_components=3,
            random_state=42
        )

        reduced = tsne.fit_transform(
            vectors
        )

        logger.info(
            "Generated 3D t-SNE projection"
        )

        return reduced, data