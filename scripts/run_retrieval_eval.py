from statistics import mean
from app.evaluation.test_cases import (load_tests)
from app.evaluation.evaluator import (evaluate_retrieval)

tests = load_tests()
mrr_scores = []
ndcg_scores = []
coverage_scores = []

for test in tests:

    result = (
        evaluate_retrieval(
            test
        )
    )

    print("\n" + "=" * 60)
    print(f"Question: {test.question}")
    print(f"MRR: {result.mrr:.4f}")
    print(f"nDCG: {result.ndcg:.4f}")
    print(f"Coverage: {result.keyword_coverage:.2f}%")

    mrr_scores.append(result.mrr)

    ndcg_scores.append(result.ndcg)

    coverage_scores.append(result.keyword_coverage)
    

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Average MRR: {mean(mrr_scores):.4f}")
print(f"Average nDCG: {mean(ndcg_scores):.4f}")
print(f"Average Coverage: {mean(coverage_scores):.2f}%")