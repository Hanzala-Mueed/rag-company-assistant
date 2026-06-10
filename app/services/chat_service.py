from app.rag.rag_pipeline import RAGPipeline

from app.llm.ollama_llm import OllamaLLM

from app.utils.logger import get_logger

from app.retrieval.query_builder import (
    build_combined_query
)

logger = get_logger(__name__)


class ChatService:

    def __init__(self):

        self.rag = RAGPipeline()
        self.llm = OllamaLLM()

        # Store latest chat session data
        self.last_question = ""
        self.last_answer = ""
        self.last_docs = []

    @staticmethod
    def combine_questions(
        question: str,
        history: list[dict]
    ):

        previous = "\n".join(
            msg["content"]
            for msg in history
            if msg["role"] == "user"
        )

        return previous + "\n" + question

    def answer(
        self,
        question: str,
        history: list[dict]
    ):
        
        logger.info(
            f"Current question: {question}"
        )

        logger.info(
            f"History messages: {len(history)}"
        )


        combined_query = (
            build_combined_query(
                question,
                history
            )
        )

        logger.info(
            f"Combined query:\n{combined_query}"
        )

        docs, context = (
            self.rag.retrieve_context(
                combined_query
            )
        )
        logger.info(
            f"Retrieved {len(docs)} documents"
        )


        system_prompt = (
            self.rag.build_system_prompt(
                context
            )
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        answer = (
            self.llm.generate(messages)
        )

        self.last_question = question
        self.last_answer = answer
        self.last_docs = docs

        logger.info(
            "Response generated successfully"
        )

        return answer, docs
    

chat_services = ChatService()