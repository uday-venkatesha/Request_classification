"""Tests for the API and the safety rules. No test calls a real LLM:
the model calls are replaced with fakes, so the tests are free, fast and repeatable."""
import json
import os
from types import SimpleNamespace

# Fake keys so the app can start without real ones. Must be set before the app is imported.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("MESH_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app import agent, main, rag
from app.db import SessionLocal
from app.llm import mask_pii
from app.matching import make_features
from app.models import AidRequest, AuditLog, Case, Volunteer
from app.schemas import Classification


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as test_client:  # "with" runs the startup step that creates the tables
        yield test_client


def fake_llm(reply):
    # Stands in for the OpenAI client and always answers with the given text
    message = SimpleNamespace(content=reply)
    completions = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), "fake-model"


def fake_chunks(distance):
    chunk = SimpleNamespace(page_content="Food Pantry Program - How often can I visit?\nOnce every 7 days.",
                            metadata={"source": "food_pantry.md"})
    return [(chunk, distance)]


# ---------- API ----------

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_pages_load(client):
    assert "Ask for help" in client.get("/").text
    assert "Volunteer portal" in client.get("/volunteer").text


def test_intake_answers_at_once_and_runs_the_agent_afterwards(client, monkeypatch):
    handled = []
    monkeypatch.setattr(main.agent, "run_in_background", handled.append)

    body = client.post("/intake", json={"text": "We need groceries this week.", "city": "Austin"}).json()
    assert body["crisis_message"] is None
    assert handled == [body["request_id"]]  # the agent was started for this request

    with SessionLocal() as session:
        session.delete(session.get(AidRequest, body["request_id"]))
        session.commit()


def test_intake_shows_the_hotline_for_crisis_language(client, monkeypatch):
    monkeypatch.setattr(main.agent, "run_in_background", lambda request_id: None)

    body = client.post("/intake", json={"text": "I have nothing left and I want to end my life."}).json()
    assert "988" in body["crisis_message"]

    with SessionLocal() as session:
        session.delete(session.get(AidRequest, body["request_id"]))
        session.commit()


def test_volunteer_accepts_and_finishes_a_case(client):
    # Set up one volunteer with one new case
    with SessionLocal() as session:
        volunteer = Volunteer(name="Test Volunteer", email="test.volunteer@example.org", skills=["food"],
                              availability=["weekends"], city="Austin", active=True)
        request = AidRequest(text="We need groceries this week.", source="web_form", city="Austin",
                             category="food", urgency="medium", status="assigned", contact="512-555-0199")
        session.add_all([volunteer, request])
        session.flush()
        case = Case(request_id=request.id, volunteer_id=volunteer.id, status="open")
        session.add(case)
        session.commit()
        volunteer_id, request_id, case_id = volunteer.id, request.id, case.id

    try:
        listed = client.get(f"/volunteers/{volunteer_id}/cases").json()
        assert listed[0]["case_id"] == case_id
        assert listed[0]["contact"] == "512-555-0199"

        assert client.post(f"/cases/{case_id}/respond", json={"action": "accept"}).json()["case_status"] == "accepted"
        # A case can only be accepted once
        assert client.post(f"/cases/{case_id}/respond", json={"action": "accept"}).status_code == 409

        finished = client.post(f"/cases/{case_id}/respond", json={"action": "done"}).json()
        assert finished["request_status"] == "closed"
        assert client.get(f"/volunteers/{volunteer_id}/cases").json() == []  # finished cases leave the list
    finally:
        with SessionLocal() as session:
            session.execute(delete(AuditLog).where(AuditLog.request_id == request_id))
            session.execute(delete(Case).where(Case.id == case_id))
            session.execute(delete(AidRequest).where(AidRequest.id == request_id))
            session.execute(delete(Volunteer).where(Volunteer.id == volunteer_id))
            session.commit()


def test_request_is_classified_saved_and_routed(client, monkeypatch):
    result = Classification(reason="No food at home.", category="food", urgency="high", confidence=0.9)
    monkeypatch.setattr(main, "classify", lambda text, model: result)

    response = client.post("/requests", json={"text": "We have nothing to eat at home.", "city": " round rock"})
    body = response.json()
    assert response.status_code == 200
    assert body["queue"] == "food_team"

    with SessionLocal() as session:
        saved = session.get(AidRequest, body["id"])
        assert saved.city == "Round Rock"  # cleaned the same way as in the ETL
        assert saved.lat is not None
        session.delete(saved)  # leave the database as we found it
        session.commit()


def test_too_short_request_is_rejected(client):
    assert client.post("/requests", json={"text": "help"}).status_code == 422


def test_unknown_queue_is_rejected(client):
    assert client.get("/queues/pizza").status_code == 422


# ---------- end to end: request form -> agent -> volunteer portal ----------

def scripted_llm(steps):
    # A fake LLM for the agent. Each call returns the next step of the script:
    # a (tool name, arguments) pair means "call this tool", plain text is the final summary.
    remaining = list(steps)

    def create(**kwargs):
        step = remaining.pop(0)
        if isinstance(step, str):
            message = SimpleNamespace(content=step, tool_calls=None)
        else:
            name, arguments = step
            function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call-1", function=function)])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), "fake-model"


@pytest.fixture
def volunteer_id():
    # One volunteer for the test, removed again afterwards together with everything assigned to them
    with SessionLocal() as session:
        volunteer = Volunteer(name="Test Volunteer", email="e2e.volunteer@example.org", skills=["food", "medical"],
                              availability=["weekends"], city="Austin", active=True)
        session.add(volunteer)
        session.commit()
        new_id = volunteer.id
    yield new_id
    with SessionLocal() as session:
        session.execute(delete(Case).where(Case.volunteer_id == new_id))
        session.execute(delete(Volunteer).where(Volunteer.id == new_id))
        session.commit()


def remove_request(request_id):
    with SessionLocal() as session:
        session.execute(delete(AuditLog).where(AuditLog.request_id == request_id))
        session.execute(delete(Case).where(Case.request_id == request_id))
        session.execute(delete(AidRequest).where(AidRequest.id == request_id))
        session.commit()


def test_request_from_the_form_reaches_the_volunteer_portal(client, monkeypatch, volunteer_id):
    food = Classification(reason="No food at home.", category="food", urgency="high", confidence=0.9)
    monkeypatch.setattr(agent, "classify", lambda text: food)
    monkeypatch.setattr(agent, "top_volunteers", lambda request: [{"volunteer_id": volunteer_id}])
    monkeypatch.setattr(agent, "get_client", lambda name: scripted_llm([
        ("classify_request", {}),
        ("find_volunteers", {}),
        ("create_case", {"volunteer_id": volunteer_id}),
        "Family needs groceries. Assigned to Test Volunteer.",
    ]))

    # 1. Someone fills in the request form
    form = {"text": "We have no food at home for the kids.", "city": "Austin", "contact": "512-555-0199"}
    request_id = client.post("/intake", json=form).json()["request_id"]

    try:
        # 2. The agent has run in the background and the case is on the volunteer's page
        cases = client.get(f"/volunteers/{volunteer_id}/cases").json()
        assert len(cases) == 1
        assert cases[0]["category"] == "food"
        assert cases[0]["contact"] == "512-555-0199"
        assert cases[0]["notes"] == "Family needs groceries. Assigned to Test Volunteer."

        # 3. Every step was written to the audit log
        steps = [step["action"] for step in client.get(f"/requests/{request_id}/audit").json()]
        assert steps == ["classify_request", "find_volunteers", "create_case"]

        # 4. The volunteer accepts
        accepted = client.post(f"/cases/{cases[0]['case_id']}/respond", json={"action": "accept"}).json()
        assert accepted["case_status"] == "accepted"
    finally:
        remove_request(request_id)


def test_medical_request_from_the_form_waits_for_a_reviewer(client, monkeypatch, volunteer_id):
    # The fake LLM tries to assign a medical case directly. The rules in code must stop it.
    medical = Classification(reason="Out of insulin.", category="medical", urgency="high", confidence=0.95)
    monkeypatch.setattr(agent, "classify", lambda text: medical)
    monkeypatch.setattr(main, "top_volunteers", lambda request: [])  # the review page asks for suggestions
    monkeypatch.setattr(agent, "get_client", lambda name: scripted_llm([
        ("classify_request", {}),
        ("create_case", {"volunteer_id": volunteer_id}),
        "Assigned to Test Volunteer.",
    ]))

    request_id = client.post("/intake", json={"text": "I ran out of insulin yesterday.", "city": "Austin"}).json()["request_id"]

    try:
        assert client.get(f"/volunteers/{volunteer_id}/cases").json() == []  # nothing reached the volunteer

        waiting = [r for r in client.get("/review").json() if r["id"] == request_id]
        assert "sensitive category" in waiting[0]["review_reason"]
    finally:
        remove_request(request_id)


# ---------- guardrails ----------

def test_personal_data_is_masked():
    masked = mask_pii("Call (512) 867-5309 or mail maria.g@gmail.com, SSN 123-45-6789")
    assert masked == "Call [phone] or mail [email], SSN [ssn]"


def test_crisis_question_gets_the_hotline(client):
    # No fakes needed: the crisis check answers before the search or the LLM are used
    body = client.post("/ask?stream=false", json={"question": "I want to end my life"}).json()
    assert body["guardrail"] == "crisis"
    assert "988" in body["answer"]


def test_off_topic_question_is_refused_without_calling_the_llm(monkeypatch):
    def llm_must_not_be_called(name):
        raise AssertionError("the LLM was called for an off-topic question")

    monkeypatch.setattr(rag, "retrieve", lambda question, k=4: fake_chunks(distance=0.95))
    monkeypatch.setattr(rag, "get_client", llm_must_not_be_called)

    result = rag.answer("Who won the super bowl?")
    assert result["guardrail"] == "off_topic"
    assert result["answer"] == rag.FALLBACK_ANSWER


def test_answer_without_a_citation_is_not_shown(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question, k=4: fake_chunks(distance=0.3))
    monkeypatch.setattr(rag, "get_client", lambda name: fake_llm("You can visit every single day."))

    result = rag.answer("How often can I visit the pantry?")
    assert result["guardrail"] == "no_citation"
    assert result["answer"] == rag.FALLBACK_ANSWER


def test_answer_with_a_citation_is_shown(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question, k=4: fake_chunks(distance=0.3))
    monkeypatch.setattr(rag, "get_client", lambda name: fake_llm("Once every 7 days [1]."))

    result = rag.answer("How often can I visit the pantry?")
    assert result["guardrail"] is None
    assert result["sources"][0]["file"] == "food_pantry.md"


# ---------- human review rules ----------

def aid_request(**changes):
    values = {"text": "We need groceries this week.", "category": "food", "confidence": 0.9, "lat": 30.27}
    values.update(changes)
    return SimpleNamespace(**values)


def test_routine_request_needs_no_review():
    assert agent.needs_review(aid_request()) is None


@pytest.mark.parametrize("changes", [
    {"category": "medical"},
    {"category": "legal"},
    {"category": "mental_health"},
    {"confidence": 0.5},
    {"lat": None},                                    # city missing or outside the service area
    {"text": "I have no food and I want to die"},     # crisis language
])
def test_risky_request_needs_review(changes):
    assert agent.needs_review(aid_request(**changes)) is not None


def test_agent_cannot_assign_a_sensitive_case_by_itself():
    # Even if the LLM asks for it, create_case refuses and points to human review
    result = agent.create_case(session=None, request=aid_request(category="legal"), args={"volunteer_id": 1})
    assert "send_to_review" in result["error"]


# ---------- matching features ----------

def test_match_features():
    request = SimpleNamespace(category="food", urgency="high", city="Austin", lat=30.27, lon=-97.74)
    volunteer = SimpleNamespace(skills=["food", "shelter"], availability=["weekends"], city="Austin",
                                lat=30.27, lon=-97.74)
    features = make_features(request, volunteer, past_acceptance_rate=0.5)

    assert features["skill_match"] == 1
    assert features["same_city"] == 1
    assert features["distance_km"] == 0
    assert features["high_urgency"] == 1
