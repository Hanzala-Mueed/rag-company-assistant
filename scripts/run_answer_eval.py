from statistics import mean
from app.evaluation.test_cases import load_tests
from app.evaluation.evaluator import evaluate_answer

tests = load_tests()
accuracy_scores = []
completeness_scores = []
relevance_scores = []

for test in tests:

    result, answer = (
        evaluate_answer(
            test
        )
    )

    print("\n" + "=" * 80)

    print(f"Question: {test.question}")

    print()

    print(f"Generated Answer:\n{answer}")

    print()

    print(f"Accuracy: {result.accuracy}")
    print(f"Completeness: {result.completeness}")
    print(f"Relevance: {result.relevance}")

    print()

    print(f"Feedback: {result.feedback}")

    accuracy_scores.append(
        result.accuracy
    )

    completeness_scores.append(
        result.completeness
    )

    relevance_scores.append(
        result.relevance
    )

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"Average Accuracy: {mean(accuracy_scores):.2f}")
print(f"Average Completeness: {mean(completeness_scores):.2f}")
print(f"Average Relevance: {mean(relevance_scores):.2f}")