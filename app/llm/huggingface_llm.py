from huggingface_hub import InferenceClient

from app.utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceLLM:

    def __init__(self, api_key: str):

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key
        )

        self.model = (
            "Qwen/Qwen2.5-7B-Instruct"
        )

    def generate(
        self,
        messages: list
    ) -> str:

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
        )