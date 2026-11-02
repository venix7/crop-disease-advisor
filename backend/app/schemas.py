from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DiagnosisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    crop: str
    class_id: str
    display_name: str
    model_score: float
    status: str
    created_at: datetime