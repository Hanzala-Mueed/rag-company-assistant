from app.services.chat_service import chat_services
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
#import re

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

judge_llm = OllamaLLM()

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# functions for testing used in scripts/run_retrieval_eval.py and scripts/run_answer_eval.py
def evaluate_retrieval(test_case):

    docs, _ = (
        chat_services.rag.retrieve_context(
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
        chat_services.answer(
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




# Gradio UI live evaluation functions - using llm as judge technique

# def evaluate_live_answer(reference_answer: str):

#     question = chat_service.last_question

#     generated_answer = chat_service.last_answer

#     prompt = ANSWER_EVALUATION_PROMPT.format(
#         question=question,
#         reference_answer=reference_answer,
#         generated_answer=generated_answer
#     )

#     print("\nQUESTION:")
#     print(question)

#     print("\nREFERENCE:")
#     print(reference_answer)

#     print("\nGENERATED:")
#     print(generated_answer)

#     print("\nPROMPT:")
#     print(prompt)

#     response = judge_llm.generate(
#         [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )

#     # try:

#     #     data = json.loads(response)

#     # except Exception:

#     #     data = {
#     #         "accuracy": 1,
#     #         "completeness": 1,
#     #         "relevance": 1,
#     #         "feedback": response
#     #     }



#     try:

#         match = re.search(
#             r"\{.*\}",
#             response,
#             re.DOTALL
#         )

#         if match:

#             data = json.loads(
#                 match.group()
#             )

#         else:

#             raise ValueError(
#                 "No JSON found"
#             )

#     except Exception:

#         data = {
#             "accuracy": 1,
#             "completeness": 1,
#             "relevance": 1,
#             "feedback": response
#         }

#     return {
#         "Question": question,
#         "Accuracy": data["accuracy"],
#         "Completeness": data["completeness"],
#         "Relevance": data["relevance"],
#         "Feedback": data["feedback"]
#     }


# for live evaluation during Gradio sessions using semantic similarity 
def calculate_semantic_similarity(
    reference_answer: str,
    generated_answer: str
):
    """
    Compute cosine similarity
    between reference and generated answer.
    """

    print("\nREFERENCE:")
    print(repr(reference_answer))

    print("\nGENERATED:")
    print(repr(generated_answer))


    embeddings = embedding_model.encode(
        [
            reference_answer,
            generated_answer
        ]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    print("\nSIMILARITY:")
    print(similarity)


    return float(similarity)

    
def evaluate_live_answer(
    reference_answer: str
):

    question = (
        chat_services.last_question
    )

    generated_answer = (
        chat_services.last_answer
    )
    print(chat_services.last_answer)

    similarity = (
        calculate_semantic_similarity(
            reference_answer,
            generated_answer
        )
    )

    score = round(
        similarity * 5,
        2
    )

    if similarity >= 0.90:

        feedback = (
            "Excellent match with "
            "reference answer."
        )

    elif similarity >= 0.75:

        feedback = (
            "Good answer with minor "
            "differences."
        )

    elif similarity >= 0.60:

        feedback = (
            "Partially correct answer."
        )

    else:

        feedback = (
            "Low similarity to "
            "reference answer."
        )

    print("\nQUESTION:")
    print(repr(question))

    print("\nREFERENCE:")
    print(repr(reference_answer))

    print("\nGENERATED:")
    print(repr(generated_answer))

    return {
        "Question": question,
        "Similarity": round(
            similarity,
            4
        ),
        "Accuracy": score,
        "Completeness": score,
        "Relevance": score,
        "Feedback": feedback
    }




def evaluate_live_retrieval(keywords: list[str]):

    docs = chat_services.last_docs

    result = RetrievalResult(
        mrr=calculate_mrr(
            docs,
            keywords
        ),
        ndcg=calculate_ndcg(
            docs,
            keywords
        ),
        keyword_coverage=calculate_keyword_coverage(
            docs,
            keywords
        )
    )

    return {
        "Question": chat_services.last_question,
        "MRR": result.mrr,
        "NDCG": result.ndcg,
        "Keyword Coverage": result.keyword_coverage
    }