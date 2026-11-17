from datetime import datetime

from sqlalchemy import ARRAY, JSON, Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base, engine

CATEGORIES = ["food", "shelter", "medical", "legal", "education", "mental_health"]
CITY_COORDS = {
    "Austin": (30.27, -97.74),
    "Round Rock": (30.51, -97.68),
    "San Marcos": (29.88, -97.94),
    "Georgetown": (30.63, -97.68),
    "Pflugerville": (30.44, -97.62),
}


class AidRequest(Base):
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[str | None] = mapped_column(String(20), unique=True)  # id from the intake form
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(20))  # web_form, phone, walk_in
    city: Mapped[str | None] = mapped_column(String(50))
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    category: Mapped[str | None] = mapped_column(String(20))
    urgency: Mapped[str | None] = mapped_column(String(10))  # low, medium, high
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="new")  # new, routed, in_review, assigned, closed
    review_reason: Mapped[str | None] = mapped_column(Text)  # why the agent sent it to a human
    contact: Mapped[str | None] = mapped_column(String(100))  # phone or email, never sent to the LLM
    summary: Mapped[str | None] = mapped_column(Text)  # the agent's notes for staff and the volunteer
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Volunteer(Base):
    __tablename__ = "volunteers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100), unique=True)
    skills: Mapped[list[str]] = mapped_column(ARRAY(String))  # same values as CATEGORIES
    availability: Mapped[list[str]] = mapped_column(ARRAY(String))  # weekdays, weekends, evenings
    city: Mapped[str | None] = mapped_column(String(50))
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_ref: Mapped[str | None] = mapped_column(String(20), unique=True)  # id from the old case system
    request_id: Mapped[int] = mapped_column(ForeignKey("requests.id"))
    volunteer_id: Mapped[int | None] = mapped_column(ForeignKey("volunteers.id"))
    status: Mapped[str] = mapped_column(String(20), default="open")  # open, accepted, declined, closed
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int | None] = mapped_column(ForeignKey("requests.id"))
    step: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(50))  # tool name or "human_review"
    input: Mapped[dict | None] = mapped_column(JSON)
    output: Mapped[dict | None] = mapped_column(JSON)
    model: Mapped[str | None] = mapped_column(String(50))
    prompt_version: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MatchModel(Base):
    """Trained volunteer match models. Kept in the database so the API and the
    retraining job share them without needing a separate file store."""
    __tablename__ = "match_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    auc: Mapped[float | None] = mapped_column(Float)
    trained_on_rows: Mapped[int | None] = mapped_column(Integer)
    data: Mapped[bytes] = mapped_column(LargeBinary)  # the model saved with joblib


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("tables created:", list(Base.metadata.tables))
