from app.retrieval.retriever import DocumentRetriever

from app.rag.context_builder import ContextBuilder

from app.rag.prompt_builder import PromptBuilder

from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:

    def __init__(self):

        self.retriever = (
            DocumentRetriever()
        )

    def retrieve_context(
        self,
        query: str
    ):

        docs = (
            self.retriever.retrieve(query)
        )

        context = (
            ContextBuilder.build(docs)
        )

        return docs, context

    def build_system_prompt(
        self,
        context: str
    ):

        return PromptBuilder.build(
            context
        )