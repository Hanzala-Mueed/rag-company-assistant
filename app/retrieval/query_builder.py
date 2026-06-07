# from app.utils.logger import get_logger

# logger = get_logger(__name__)


# def build_combined_query(
#     question: str,
#     history: list
# ) -> str:
#     """
#     Combine previous user messages
#     with current question.
#     """

#     previous_questions = "\n".join(
#         msg["content"]
#         for msg in history
#         if msg["role"] == "user"
#     )

#     combined_query = (
#         previous_questions
#         + "\n"
#         + question
#     )

#     logger.info(
#         "Combined query built successfully"
#     )

#     return combined_query

from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_combined_query(
    question: str,
    history: list[dict]
) -> str:

    previous_questions = "\n".join(
        msg["content"]
        for msg in history
        if msg["role"] == "user"
    )

    if previous_questions:
        combined_query = (
            previous_questions
            + "\n"
            + question
        )
    else:
        combined_query = question

    logger.info(
        f"Combined query built:\n{combined_query}"
    )

    return combined_query