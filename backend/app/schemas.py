from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DiagnosisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    crop: str
    class_id: str
    display_name: str
    model_score: float
    status: str
    created_at: datetime


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class SourceResponse(BaseModel):
    reference: int
    title: str
    url: str


class GuidanceResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]