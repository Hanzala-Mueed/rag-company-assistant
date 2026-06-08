from dataclasses import dataclass


@dataclass
class AnswerResult:

    accuracy: float
    completeness: float
    relevance: float
    feedback: str