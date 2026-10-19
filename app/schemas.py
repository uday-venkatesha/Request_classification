from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["food", "shelter", "medical", "legal", "education", "mental_health"]


class Classification(BaseModel):
    """What the LLM must return for every request (structured output)."""
    reason: str = Field(description="One short sentence on why this category was chosen")
    category: Category
    urgency: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)


class RequestIn(BaseModel):
    text: str = Field(min_length=10, examples=["We were evicted today and have nowhere to sleep."])
    source: Literal["web_form", "phone", "walk_in"] = "web_form"
    city: str | None = Field(default=None, examples=["Austin"])


class RequestOut(BaseModel):
    id: int
    category: Category
    urgency: str
    confidence: float
    reason: str
    status: str
    queue: str
