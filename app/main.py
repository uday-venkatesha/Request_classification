from typing import Literal

from fastapi import FastAPI, HTTPException
from sqlalchemy import select

from app.classifier import classify
from app.db import SessionLocal
from app.models import CITY_COORDS, AidRequest
from app.schemas import Category, RequestIn, RequestOut

app = FastAPI(title="AidRoute", description="Classifies and routes aid requests for a nonprofit.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/requests", response_model=RequestOut)
def create_request(body: RequestIn, model: Literal["gpt-4o", "llama3"] = "gpt-4o"):
    try:
        result = classify(body.text, model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Classifier failed: {e}")

    city = body.city.strip().title() if body.city else None
    lat, lon = CITY_COORDS.get(city, (None, None))

    with SessionLocal() as session:
        request = AidRequest(
            text=body.text, source=body.source, city=city, lat=lat, lon=lon,
            category=result.category, urgency=result.urgency, confidence=result.confidence,
            status="routed",
        )
        session.add(request)
        session.commit()
        return RequestOut(
            id=request.id, category=result.category, urgency=result.urgency,
            confidence=result.confidence, reason=result.reason,
            status=request.status, queue=f"{result.category}_team",
        )


@app.get("/queues/{category}")
def get_queue(category: Category):
    """Open requests waiting for one team, most urgent first."""
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    with SessionLocal() as session:
        rows = session.scalars(
            select(AidRequest).where(AidRequest.category == category, AidRequest.status == "routed")
        ).all()
        rows = sorted(rows, key=lambda r: (urgency_order.get(r.urgency, 3), r.created_at))
        return [{"id": r.id, "urgency": r.urgency, "city": r.city, "text": r.text[:120]} for r in rows]
