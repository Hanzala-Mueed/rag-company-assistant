from typing import List

from langchain_core.documents import Document


def build_context(documents: List[Document]) -> str:
    """
    Convert retrieved documents into a single context string.
    """

    return "\n\n".join(
        doc.page_content
        for doc in documents
    )