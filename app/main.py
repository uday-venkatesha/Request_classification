import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from app import agent, rag
from app.classifier import classify
from app.db import SessionLocal, engine
from app.matching import top_volunteers
from app.models import CITY_COORDS, AidRequest, Base, Case, Volunteer
from app.schemas import AskIn, CaseResponseIn, Category, RequestIn, RequestOut, ReviewIn

ModelName = Literal["gpt-4o", "llama3"]
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app):
    # Creates any missing tables when the service starts. On a database that is
    # already set up this does nothing, so a fresh deployment works without manual steps.
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="AidRoute", description="Classifies and routes aid requests for a nonprofit.",
              lifespan=lifespan)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def request_page():
    # The public form for people who need help
    return FileResponse(STATIC / "index.html")


@app.get("/volunteer", include_in_schema=False)
def volunteer_page():
    # Where volunteers see the cases assigned to them
    return FileResponse(STATIC / "volunteer.html")


@app.get("/health")
def health():
    return {"status": "ok"}


def new_request(body):
    # Cities are cleaned the same way as in the ETL, so "  round rock" and "Round Rock" match
    city = body.city.strip().title() if body.city else None
    lat, lon = CITY_COORDS.get(city, (None, None))
    return AidRequest(text=body.text, source=body.source, city=city, lat=lat, lon=lon, contact=body.contact)


@app.post("/requests", response_model=RequestOut)
def create_request(body: RequestIn, model: ModelName = "gpt-4o"):
    """Classifies a request and puts it in the right team's queue."""
    try:
        result = classify(body.text, model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Classifier failed: {e}")

    with SessionLocal() as session:
        request = new_request(body)
        request.category = result.category
        request.urgency = result.urgency
        request.confidence = result.confidence
        request.status = "routed"
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


@app.post("/ask")
def ask(body: AskIn, model: ModelName = "gpt-4o", stream: bool = True):
    """Answers a question from the program documents.

    With stream=true the answer arrives word by word as server-sent events.
    With stream=false it comes back as one JSON object."""
    if not stream:
        try:
            return rag.answer(body.question, model)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Assistant failed: {e}")

    def events():
        # Each server-sent event is one line starting with "data: " followed by a blank line
        try:
            for event in rag.answer_stream(body.question, model):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/intake")
def intake(body: RequestIn, background: BackgroundTasks):
    """Used by the public request form. The request is saved and the person gets an answer
    immediately. Classification, matching and assignment happen in the background."""
    with SessionLocal() as session:
        request = new_request(body)
        session.add(request)
        session.commit()
        request_id = request.id

    background.add_task(agent.run_in_background, request_id)

    # Someone in crisis must see the hotline right away, without waiting for the agent
    crisis_message = rag.CRISIS_ANSWER if rag.is_crisis(body.text) else None
    return {"request_id": request_id, "crisis_message": crisis_message}


@app.post("/triage")
def triage(body: RequestIn):
    """Runs the full agent on a request: classify, look up program rules, match a volunteer,
    then either assign the case or send it to a human reviewer. Every step is logged."""
    with SessionLocal() as session:
        request = new_request(body)
        session.add(request)
        session.commit()  # saved before the agent starts, so the request is never lost
        request_id = request.id

    try:
        return agent.run_agent(request_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent failed, request {request_id} is saved: {e}")


@app.get("/review")
def review_queue():
    """Requests waiting for a human decision, most urgent first, with suggested volunteers."""
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    with SessionLocal() as session:
        rows = session.scalars(select(AidRequest).where(AidRequest.status == "in_review")).all()
        rows = sorted(rows, key=lambda r: (urgency_order.get(r.urgency, 3), r.created_at))
        return [
            {"id": r.id, "category": r.category, "urgency": r.urgency, "confidence": r.confidence,
             "review_reason": r.review_reason, "city": r.city, "text": r.text,
             "suggested_volunteers": top_volunteers(r) if r.category else []}
            for r in rows
        ]


@app.post("/review/{request_id}")
def review(request_id: int, body: ReviewIn):
    """A reviewer approves (a case is created) or rejects (the request is closed)."""
    try:
        return agent.apply_review(request_id, body.approve, body.volunteer_id, body.category, body.note)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/requests/{request_id}/audit")
def audit(request_id: int):
    """Every step taken on a request, by the agent or by a person."""
    with SessionLocal() as session:
        return agent.audit_trail(session, request_id)


@app.get("/volunteers")
def volunteers():
    """Active volunteers with the number of new cases waiting for each, used by the portal's sign-in list."""
    with SessionLocal() as session:
        new_cases = dict(session.execute(
            select(Case.volunteer_id, func.count()).where(Case.status == "open").group_by(Case.volunteer_id)
        ).all())
        rows = session.scalars(select(Volunteer).where(Volunteer.active).order_by(Volunteer.name)).all()
        return [{"id": v.id, "name": v.name, "city": v.city, "new_cases": new_cases.get(v.id, 0)} for v in rows]


@app.get("/volunteers/{volunteer_id}/cases")
def volunteer_cases(volunteer_id: int):
    """A volunteer's new cases and the ones they are working on, newest first."""
    with SessionLocal() as session:
        rows = session.execute(
            select(Case, AidRequest)
            .join(AidRequest, AidRequest.id == Case.request_id)
            .where(Case.volunteer_id == volunteer_id, Case.closed_at.is_(None), Case.status.in_(["open", "accepted"]))
            .order_by(Case.created_at.desc())
        ).all()
        return [
            {"case_id": case.id, "status": case.status,
             "assigned_at": case.created_at.isoformat() + "Z",  # the database keeps UTC; "Z" lets the browser show local time
             "category": request.category, "urgency": request.urgency, "city": request.city,
             "text": request.text, "contact": request.contact, "notes": request.summary}
            for case, request in rows
        ]


@app.post("/cases/{case_id}/respond")
def respond(case_id: int, body: CaseResponseIn):
    """A volunteer accepts or declines a new case, or marks an accepted case as done."""
    try:
        return agent.respond_to_case(case_id, body.action)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
