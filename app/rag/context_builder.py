from langchain_core.documents import Document


class ContextBuilder:

    @staticmethod
    def build(
        documents: list[Document]
    ) -> str:

        return "\n\n".join(
            doc.page_content
            for doc in documents
        )