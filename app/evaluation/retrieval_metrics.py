from dataclasses import dataclass

from sklearn.metrics import ndcg_score


@dataclass
class RetrievalResult:

    mrr: float
    ndcg: float
    keyword_coverage: float

def calculate_mrr(docs, keywords):

        for idx, doc in enumerate(docs):

            text = doc.page_content.lower()

            if any(
                keyword.lower() in text for keyword in keywords):
                
                return 1 / (idx + 1)

        return 0.0

def calculate_keyword_coverage(docs, keywords):

    text = " ".join(
        doc.page_content
        for doc in docs
    ).lower()

    found = sum(
        1
        for keyword in keywords
        if keyword.lower() in text
    )

    return (
        found / len(keywords)
    ) * 100

def calculate_ndcg(docs, keywords):

    y_true = []
    y_score = []

    total_docs = len(docs)

    for idx, doc in enumerate(docs):

        text = doc.page_content.lower()

        relevant = any(
            keyword.lower() in text
            for keyword in keywords
        )

        y_true.append(1 if relevant else 0)

        y_score.append(total_docs - idx)

    if sum(y_true) == 0: return 0.0

    return float(
        ndcg_score(
            [y_true],
            [y_score]
        )
    )