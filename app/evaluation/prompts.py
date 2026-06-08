ANSWER_EVALUATION_PROMPT = """
You are an expert RAG evaluator.

Evaluate the generated answer.

Question:
{question}

Reference Answer:
{reference_answer}

Generated Answer:
{generated_answer}

Return ONLY valid JSON.

Format:

{{
    "accuracy": 1-5,
    "completeness": 1-5,
    "relevance": 1-5,
    "feedback": "short explanation"
}}
"""