from pydantic import BaseModel


class EvaluationResponse(BaseModel):

    accuracy: float

    completeness: float

    relevance: float

    feedback: str