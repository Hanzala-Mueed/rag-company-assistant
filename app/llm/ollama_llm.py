from langchain_ollama import ChatOllama
from app.utils.logger import get_logger
import json

logger = get_logger(__name__)


class OllamaLLM:

    def __init__(self):

        self.model_name = "llama3.2"

        self.llm = ChatOllama(
            model=self.model_name,
            temperature=0
        )

        logger.info(f"Ollama model loaded: {self.model_name}")


    def generate(self, messages: list) -> str:

        response = self.llm.invoke(messages)
        return response.content

    def generate_json(self, messages: list):

        response = self.generate(messages)
        return json.loads(response)