"""Creates the starting data for the project.

1. data/eval/labeled_requests.csv  - LLM-written aid requests with their true category
2. data/raw/intake_form_export.csv - messy web form export (source system 1)
3. data/raw/volunteers.xlsx        - the manual volunteer spreadsheet (source system 2)
4. data/raw/case_management.json   - past cases and volunteer outcomes (source system 3)
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import LengthFinishReasonError, OpenAI
from pydantic import BaseModel

from app.models import CATEGORIES

random.seed(42)
RAW = Path("data/raw")
EVAL = Path("data/eval")

CITIES = {
    "Austin": (30.27, -97.74),
    "Round Rock": (30.51, -97.68),
    "San Marcos": (29.88, -97.94),
    "Georgetown": (30.63, -97.68),
    "Pflugerville": (30.44, -97.62),
}
STYLES = [
    "short and urgent, like a text message",
    "long and emotional, explaining the whole situation",
    "written by a case worker on behalf of a client",
    "broken English with a few typos",
    "calm and formal, like an email",
]
FIRST = ["Maria", "James", "Aisha", "Wei", "Carlos", "Priya", "John", "Fatima", "David", "Lena",
         "Omar", "Sarah", "Raj", "Emily", "Kwame", "Sofia", "Daniel", "Mei", "Lucas", "Grace"]
LAST = ["Garcia", "Smith", "Khan", "Chen", "Lopez", "Patel", "Brown", "Ali", "Nguyen", "Johnson"]


class GeneratedRequest(BaseModel):
    text: str
    urgency: str  # low, medium, high


class Batch(BaseModel):
    requests: list[GeneratedRequest]


def generate_requests(per_style=20):
    load_dotenv()
    client = OpenAI()
    rows = []
    for category in CATEGORIES:
        for style in STYLES:
            prompt = (
                f"Write {per_style} different requests for help sent to a nonprofit in Central Texas. "
                f"The main need in every request is '{category}'. Style: {style}. "
                "Vary the people, situations and length. Some may mention a second smaller need, "
                "but the main need must be clear. Give each an urgency of low, medium or high."
            )
            batch = []
            for attempt in range(4):  # the model sometimes returns too few or rambles past the limit
                try:
                    response = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        response_format=Batch,
                        temperature=0.9,
                        max_tokens=8000,
                    )
                except LengthFinishReasonError:
                    continue
                batch += response.choices[0].message.parsed.requests
                if len(batch) >= per_style:
                    break
            for r in batch[:per_style]:
                rows.append({"text": r.text, "category": category, "urgency": r.urgency})
            print(f"{category:14} {style[:30]:30} total={len(rows)}")
    return pd.DataFrame(rows)


def clean_labeled(df):
    # the model often writes "Urgency: High" or "Low urgency." inside the text, which gives away the answer
    level = r"(low|medium|high)"
    for pattern in [rf"[^.!?\n]*\burgency\b[^.!?\n]*\b{level}\b[^.!?\n]*[.!?]?",
                    rf"[^.!?\n]*\b{level}\s+urgency\b[^.!?\n]*[.!?]?"]:
        df["text"] = df["text"].str.replace(pattern, "", case=False, regex=True)
    df["text"] = df["text"].str.strip()
    df["urgency"] = df["urgency"].str.lower()
    return df


def messy_date(d):
    return random.choice([d.strftime("%Y-%m-%d %H:%M"), d.strftime("%m/%d/%Y"), d.strftime("%d %b %Y")])


def messy_city(city):
    return random.choice([city, city.lower(), city.upper(), city + " ", " " + city + ", TX"])


def build_intake_form(labeled):
    start = datetime(2025, 1, 1)
    rows = []
    for i, r in labeled.iterrows():
        rows.append({
            "Submission ID": f"SUB-{1000 + i}",
            "Submitted On": messy_date(start + timedelta(minutes=random.randint(0, 600_000))),
            "Full Request": r["text"] if random.random() > 0.02 else "",
            "City": messy_city(random.choice(list(CITIES))) if random.random() > 0.05 else "",
            "How did you contact us": random.choice(["Web", "web form", "Phone", "phone call", "Walk-in", "walk in"]),
        })
    df = pd.DataFrame(rows)
    duplicates = df.sample(15, random_state=1)  # people often submit twice
    return pd.concat([df, duplicates], ignore_index=True)


def build_volunteers(n=80):
    skill_text = {"food": ["Food", "food bank"], "shelter": ["Shelter", "housing"],
                  "medical": ["Medical", "nurse"], "legal": ["Legal", "legal aid"],
                  "education": ["Education", "tutoring"], "mental_health": ["Mental Health", "counseling"]}
    availability = ["weekends only", "Weekdays", "weekday evenings", "any time", "Sat/Sun", "evenings"]
    rows, real_skills = [], []
    for i in range(n):
        name = f"{random.choice(FIRST)} {random.choice(LAST)}"
        skills = random.sample(CATEGORIES, random.randint(1, 3))
        real_skills.append(skills)
        email = f"{name.lower().replace(' ', '.')}{i}@example.org"
        rows.append({
            "Name": name,
            "Email": email.upper() if random.random() < 0.1 else email,
            "Can help with": ", ".join(random.choice(skill_text[s]) for s in skills),
            "Availability": random.choice(availability),
            "City": messy_city(random.choice(list(CITIES))),
            "Active?": random.choice(["Y", "yes", "Yes", "N", ""]),
        })
    return pd.DataFrame(rows), real_skills


def build_cases(intake, labeled, volunteers, real_skills):
    """Past cases. A volunteer is more likely to accept when the category matches
    their skills and they live in the same city - this is what the match model learns."""
    cases = []
    for _ in range(600):
        i = random.randrange(len(labeled))
        j = random.randrange(len(volunteers))
        v = volunteers.iloc[j]
        skill_match = labeled.loc[i, "category"] in real_skills[j]
        same_city = intake.loc[i, "City"].strip().lower().startswith(v["City"].strip().lower()[:5])
        p_accept = 0.15 + 0.5 * skill_match + 0.25 * same_city
        outcome = "accepted" if random.random() < p_accept else "declined"
        opened = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 400))
        cases.append({
            "case_ref": f"CASE-{len(cases) + 1:04d}",
            "submission_id": intake.loc[i, "Submission ID"],
            "volunteer_email": v["Email"],
            "outcome": random.choice([outcome, outcome.upper()]) if random.random() > 0.03 else None,
            "opened": opened.isoformat(),
            "closed": (opened + timedelta(days=random.randint(1, 30))).isoformat() if outcome == "accepted" else None,
        })
    return cases


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    labeled_path = EVAL / "labeled_requests.csv"
    if labeled_path.exists():  # LLM calls cost money, so only generate once
        labeled = pd.read_csv(labeled_path)
    else:
        labeled = generate_requests()
    labeled = clean_labeled(labeled)
    labeled.to_csv(labeled_path, index=False)

    intake = build_intake_form(labeled)
    volunteers, real_skills = build_volunteers()
    cases = build_cases(intake, labeled, volunteers, real_skills)

    intake.to_csv(RAW / "intake_form_export.csv", index=False)
    volunteers.to_excel(RAW / "volunteers.xlsx", index=False)
    (RAW / "case_management.json").write_text(json.dumps(cases, indent=2))

    print(f"labeled requests: {len(labeled)}")
    print(f"intake rows: {len(intake)}, volunteers: {len(volunteers)}, cases: {len(cases)}")
