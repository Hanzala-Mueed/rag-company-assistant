from app.config.prompts import (
    SYSTEM_PROMPT
)


class PromptBuilder:

    @staticmethod
    def build(context: str):

        return SYSTEM_PROMPT.format(
            context=context
        )