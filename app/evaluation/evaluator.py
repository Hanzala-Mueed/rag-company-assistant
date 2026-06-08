from app.services.chat_service import ChatService
from app.evaluation.retrieval_metrics import (
    RetrievalResult,
    calculate_mrr,
    calculate_ndcg,
    calculate_keyword_coverage
)

from app.evaluation.test_cases import load_tests
from app.evaluation.answer_metrics import AnswerResult
from app.evaluation.prompts import ANSWER_EVALUATION_PROMPT
from app.evaluation.judge_models import EvaluationResponse
from app.llm.ollama_llm import OllamaLLM
import json

chat_service = ChatService()
judge_llm = OllamaLLM()

def evaluate_retrieval(test_case):

    docs, _ = (
        chat_service.rag.retrieve_context(
            test_case.question
        )
    )

    mrr = calculate_mrr(
        docs,
        test_case.keywords
    )

    ndcg = calculate_ndcg(
        docs,
        test_case.keywords
    )

    coverage = calculate_keyword_coverage(
        docs,
        test_case.keywords
    )

    return RetrievalResult(
        mrr=mrr,
        ndcg=ndcg,
        keyword_coverage=coverage
    )

def evaluate_all_retrieval():

    tests = load_tests()

    for index, test in enumerate(tests):

        result = (
            evaluate_retrieval(
                test
            )
        )

        yield (
            test,
            result,
            (index + 1) / len(tests)
        )


def evaluate_answer(test_case):
    generated_answer, _ = (
        chat_service.answer(
            question=test_case.question,
            history=[]
        )
    )

    prompt = (
        ANSWER_EVALUATION_PROMPT.format(
            question=test_case.question,
            reference_answer=test_case.reference_answer,
            generated_answer=generated_answer
        )
    )

    response = judge_llm.generate(
        [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    try:

        data = json.loads(
            response
        )

    except Exception:

        data = {
            "accuracy": 1,
            "completeness": 1,
            "relevance": 1,
            "feedback": response
        }

    return (
        AnswerResult(
            accuracy=float(
                data["accuracy"]
            ),
            completeness=float(
                data["completeness"]
            ),
            relevance=float(
                data["relevance"]
            ),
            feedback=data["feedback"]
        ),
        generated_answer
    )

def evaluate_all_answers():

    tests = load_tests()

    for index, test in enumerate(tests):

        result, answer = (
            evaluate_answer(
                test
            )
        )

        yield (
            test,
            result,
            answer,
            (index + 1) / len(tests)
        )