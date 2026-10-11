from datetime import datetime

from sqlalchemy import ARRAY, JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base, engine

CATEGORIES = ["food", "shelter", "medical", "legal", "education", "mental_health"]


class AidRequest(Base):
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(20))  # web_form, phone, walk_in
    city: Mapped[str | None] = mapped_column(String(50))
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    category: Mapped[str | None] = mapped_column(String(20))
    urgency: Mapped[str | None] = mapped_column(String(10))  # low, medium, high
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="new")  # new, routed, in_review, assigned, closed
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


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("tables created:", list(Base.metadata.tables))
