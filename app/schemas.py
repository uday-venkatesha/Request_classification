from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["food", "shelter", "medical", "legal", "education", "mental_health"]


class Classification(BaseModel):
    """What the LLM must return for every request (structured output)."""
    reason: str = Field(description="One short sentence on why this category was chosen")
    category: Category
    urgency: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
