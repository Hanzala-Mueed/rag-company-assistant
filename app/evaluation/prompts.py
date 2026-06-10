# ANSWER_EVALUATION_PROMPT = """
# You are an expert RAG evaluator.

# Evaluate the generated answer.

# Question:
# {question}

# Reference Answer:
# {reference_answer}

# Generated Answer:
# {generated_answer}

# Return ONLY valid JSON.

# Format:

# {{
#     "accuracy": 1-5,
#     "completeness": 1-5,
#     "relevance": 1-5,
#     "feedback": "short explanation"
# }}
# """

ANSWER_EVALUATION_PROMPT = """
You are an expert evaluator.

Compare the Generated Answer against the Reference Answer.

Scoring:

5 = perfect match
4 = mostly correct
3 = partially correct
2 = weak answer
1 = incorrect
0 = completely wrong

Evaluate:

1. Accuracy
2. Completeness
3. Relevance

Question:
{question}

Reference Answer:
{reference_answer}

Generated Answer:
{generated_answer}

If the generated answer conveys the same meaning as the reference answer,
give high scores even if wording differs.

Return ONLY valid JSON.

{{
    "accuracy": 0,
    "completeness": 0,
    "relevance": 0,
    "feedback": ""
}}
"""