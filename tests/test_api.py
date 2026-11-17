"""Tests for the API and the safety rules. No test calls a real LLM:
the model calls are replaced with fakes, so the tests are free, fast and repeatable."""
import os
from types import SimpleNamespace

# Fake keys so the app can start without real ones. Must be set before the app is imported.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("MESH_API_KEY", "test-key")

import pytest
from fastapi.testclient import TestClient

from app import agent, main, rag
from app.db import SessionLocal
from app.llm import mask_pii
from app.matching import make_features
from app.models import AidRequest
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
