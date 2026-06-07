from app.rag.rag_pipeline import RAGPipeline

# from app.llm.huggingface_llm import HuggingFaceLLM
from app.llm.ollama_llm import OllamaLLM

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChatService:

    def __init__(self):

        self.rag = RAGPipeline()

        # self.llm = HuggingFaceLLM(
        #     api_key=HF_API_KEY
        # )
        self.llm = OllamaLLM()

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

        combined_query = (
            self.combine_questions(
                question,
                history
            )
        )

        docs, context = (
            self.rag.retrieve_context(
                combined_query
            )
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

        return answer, docs