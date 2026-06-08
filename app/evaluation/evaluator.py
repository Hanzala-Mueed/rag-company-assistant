from app.services.chat_service import ChatService

from app.evaluation.retrieval_metrics import (
    RetrievalResult,
    calculate_mrr,
    calculate_ndcg,
    calculate_keyword_coverage
)

from app.evaluation.test_cases import (
    load_tests
)


chat_service = ChatService()

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