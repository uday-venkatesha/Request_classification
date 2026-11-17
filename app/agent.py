"""Triage agent: prepares an incoming aid request for the staff.

The LLM decides which tool to call next (classify, search the docs, find volunteers,
create a case, or send to a human). Two things are never left to the LLM:
  - the rules for when a human must review a case (needs_review)
  - the audit log, which records every step so each decision can be explained later
"""
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select

from app import rag
from app.classifier import classify
from app.db import SessionLocal
from app.llm import get_client, mask_pii
from app.matching import top_volunteers
from app.models import AidRequest, AuditLog, Case, Volunteer

PROMPT_VERSION = "v1"
PROMPT = (Path(__file__).parent.parent / "prompts" / f"agent_{PROMPT_VERSION}.txt").read_text()

# The agent stops after this many LLM calls, so a confused model cannot loop forever
MAX_STEPS = 8

# Mistakes in these categories can hurt someone, so a person always checks them
SENSITIVE_CATEGORIES = ["medical", "legal", "mental_health"]
MIN_CONFIDENCE = 0.75

# What the LLM is told about each tool. The tools take no request id:
# the agent only ever works on the one request it was started for.
TOOLS = [
    {"type": "function", "function": {
        "name": "classify_request",
        "description": "Classify the request into a category with urgency and confidence. Call this first.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the program documents for rules such as eligibility, hours and what to bring.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "find_volunteers",
        "description": "Get the active volunteers most likely to accept this request, best first.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "create_case",
        "description": "Assign the request to a volunteer. Only for routine requests.",
        "parameters": {"type": "object", "properties": {"volunteer_id": {"type": "integer"}},
                       "required": ["volunteer_id"]},
    }},
    {"type": "function", "function": {
        "name": "send_to_review",
        "description": "Hand the request to a human reviewer instead of assigning it.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
    }},
]


def needs_review(request):
    # Returns the reason a human must look at this request, or None if it can be assigned automatically.
    # These rules live in code, not in the prompt, so the LLM cannot talk its way around them.
    if rag.is_crisis(request.text):
        return "possible crisis language, contact the person right away"
    if request.category is None:
        return "request has not been classified"
    if request.category in SENSITIVE_CATEGORIES:
        return f"sensitive category ({request.category})"
    if request.confidence is not None and request.confidence < MIN_CONFIDENCE:
        return f"low classifier confidence ({request.confidence:.2f})"
    if request.lat is None:
        return "city is missing or outside our service area"
    return None


# ---------- the tools ----------

def classify_request(session, request, args):
    result = classify(request.text)
    request.category = result.category
    request.urgency = result.urgency
    request.confidence = result.confidence
    return result.model_dump()


def search_docs(session, request, args):
    chunks = rag.retrieve(mask_pii(args["query"]), k=3)
    return {"results": [chunk.page_content for chunk, distance in chunks]}


def find_volunteers(session, request, args):
    if request.category is None:
        return {"error": "Call classify_request first."}
    return {"volunteers": top_volunteers(request)}


def create_case(session, request, args):
    reason = needs_review(request)
    if reason:
        return {"error": f"This request cannot be assigned automatically: {reason}. Call send_to_review."}

    volunteer = session.get(Volunteer, args["volunteer_id"])
    if volunteer is None or not volunteer.active or request.category not in volunteer.skills:
        return {"error": "That volunteer is not available for this category. Pick one from find_volunteers."}

    case = Case(request_id=request.id, volunteer_id=volunteer.id, status="open")
    session.add(case)
    request.status = "assigned"
    session.flush()  # gives the case its id
    return {"case_id": case.id, "volunteer": volunteer.name}


def send_to_review(session, request, args):
    request.status = "in_review"
    request.review_reason = args["reason"]
    return {"status": "in_review"}


TOOL_FUNCTIONS = {
    "classify_request": classify_request,
    "search_docs": search_docs,
    "find_volunteers": find_volunteers,
    "create_case": create_case,
    "send_to_review": send_to_review,
}


# ---------- audit log ----------

def write_audit(session, request, action, tool_input, tool_output, model=None):
    last_step = session.scalar(select(func.max(AuditLog.step)).where(AuditLog.request_id == request.id)) or 0
    session.add(AuditLog(
        request_id=request.id, step=last_step + 1, action=action, input=tool_input, output=tool_output,
        model=model, prompt_version=PROMPT_VERSION if model else None,
    ))
    session.commit()  # saved straight away, so the trail survives even if a later step crashes


def audit_trail(session, request_id):
    rows = session.scalars(select(AuditLog).where(AuditLog.request_id == request_id).order_by(AuditLog.step))
    return [{"step": row.step, "action": row.action, "input": row.input, "output": row.output,
             "model": row.model, "prompt_version": row.prompt_version, "at": row.created_at.isoformat()}
            for row in rows]


# ---------- the agent loop ----------

def run_agent(request_id, model_name="gpt-4o"):
    client, model = get_client(model_name)

    with SessionLocal() as session:
        request = session.get(AidRequest, request_id)
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"City: {request.city or 'not given'}\nRequest: {mask_pii(request.text)}"},
        ]
        summary = None

        for _ in range(MAX_STEPS):
            response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, temperature=0)
            message = response.choices[0].message

            # No tool call means the agent is finished and this is its summary
            if not message.tool_calls:
                summary = message.content
                break

            messages.append(message)
            for call in message.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                try:
                    result = TOOL_FUNCTIONS[call.function.name](session, request, args)
                except Exception as e:  # the agent sees the error and can react to it
                    result = {"error": str(e)}

                write_audit(session, request, call.function.name, args, result, model)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

        # Safety net: if the agent stopped without a decision, a human takes over
        if request.status not in ("assigned", "in_review"):
            reason = needs_review(request) or "the agent stopped without making a decision"
            result = send_to_review(session, request, {"reason": reason})
            write_audit(session, request, "safety_net", {"reason": reason}, result)

        request.summary = summary  # shown to the reviewer and to the assigned volunteer
        session.commit()
        case_id = session.scalar(select(func.max(Case.id)).where(Case.request_id == request.id))
        return {
            "request_id": request.id,
            "category": request.category,
            "urgency": request.urgency,
            "confidence": request.confidence,
            "status": request.status,
            "review_reason": request.review_reason,
            "case_id": case_id,
            "summary": summary,
            "steps": audit_trail(session, request.id),
        }


def run_in_background(request_id):
    # Used by the public request form: the person gets an answer straight away and the
    # agent works afterwards. If the agent fails (for example the LLM is down),
    # the request goes to a human instead of sitting unnoticed.
    try:
        run_agent(request_id)
    except Exception as e:
        with SessionLocal() as session:
            request = session.get(AidRequest, request_id)
            reason = "automatic triage failed, please handle by hand"
            result = send_to_review(session, request, {"reason": reason})
            write_audit(session, request, "safety_net", {"reason": reason, "error": str(e)[:300]}, result)


# ---------- human review ----------

def apply_review(request_id, approve, volunteer_id=None, category=None, note=None):
    """A staff member's decision on a request the agent sent to review.
    Returns the result, or raises ValueError if the decision cannot be applied."""
    with SessionLocal() as session:
        request = session.get(AidRequest, request_id)
        if request is None or request.status != "in_review":
            raise ValueError("This request is not waiting for review.")

        decision = {"approve": approve, "volunteer_id": volunteer_id, "category": category, "note": note}

        if category:  # the reviewer corrected the classifier
            request.category = category

        if not approve:
            request.status = "closed"
            result = {"status": "closed"}
        else:
            if volunteer_id is None:  # no volunteer chosen, so take the best match
                suggestions = top_volunteers(request)
                if not suggestions:
                    raise ValueError("No volunteer has this skill. Choose a volunteer_id or reject the request.")
                volunteer_id = suggestions[0]["volunteer_id"]

            volunteer = session.get(Volunteer, volunteer_id)
            if volunteer is None:
                raise ValueError("Unknown volunteer.")

            case = Case(request_id=request.id, volunteer_id=volunteer.id, status="open")
            session.add(case)
            request.status = "assigned"
            session.flush()
            result = {"status": "assigned", "case_id": case.id, "volunteer": volunteer.name}

        write_audit(session, request, "human_review", decision, result)
        return result


# ---------- volunteer responses ----------

def respond_to_case(case_id, action):
    """A volunteer accepts or declines a new case, or marks an accepted one as done.
    Raises ValueError if that is not possible for the case."""
    with SessionLocal() as session:
        case = session.get(Case, case_id)
        if case is None:
            raise ValueError("Unknown case.")
        request = session.get(AidRequest, case.request_id)
        volunteer = session.get(Volunteer, case.volunteer_id)

        if action == "accept" and case.status == "open":
            case.status = "accepted"

        elif action == "decline" and case.status == "open":
            # Staff pick someone else. The match model will not suggest this volunteer again for this request.
            case.status = "declined"
            request.status = "in_review"
            request.review_reason = f"{volunteer.name} declined the case"

        elif action == "done" and case.status == "accepted" and case.closed_at is None:
            # The case stays "accepted" so the match model keeps learning from it; closed_at marks it finished
            case.closed_at = datetime.now()
            request.status = "closed"

        else:
            raise ValueError(f"A case that is '{case.status}' cannot be changed with '{action}'.")

        result = {"case_id": case.id, "case_status": case.status, "request_status": request.status}
        write_audit(session, request, f"volunteer_{action}", {"volunteer_id": volunteer.id}, result)
        return result
