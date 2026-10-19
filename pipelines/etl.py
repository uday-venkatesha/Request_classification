"""Cleans the 3 source system exports and loads them into Postgres.

Full reload: existing rows are wiped and everything is loaded again, so it is safe to re-run.
"""
import json

import pandas as pd
from sqlalchemy import text

from app.db import SessionLocal, engine
from app.models import CITY_COORDS, AidRequest, Base, Case, Volunteer

SOURCES = {"web": "web_form", "web form": "web_form", "phone": "phone", "phone call": "phone",
           "walk-in": "walk_in", "walk in": "walk_in"}
SKILLS = {"food": "food", "food bank": "food", "shelter": "shelter", "housing": "shelter",
          "medical": "medical", "nurse": "medical", "legal": "legal", "legal aid": "legal",
          "education": "education", "tutoring": "education",
          "mental health": "mental_health", "counseling": "mental_health"}
AVAILABILITY = {"weekends only": ["weekends"], "sat/sun": ["weekends"], "weekdays": ["weekdays"],
                "weekday evenings": ["weekdays", "evenings"], "evenings": ["evenings"],
                "any time": ["weekdays", "weekends", "evenings"]}


def clean_city(series):
    return series.fillna("").str.replace(", TX", "", case=False).str.strip().str.title().replace("", None)


def add_coords(df):
    df["lat"] = df["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    df["lon"] = df["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    return df


def clean_intake(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="Submission ID")
    df = df[df["Full Request"].fillna("").str.strip() != ""]
    df = pd.DataFrame({
        "submission_id": df["Submission ID"],
        "text": df["Full Request"].str.strip(),
        "source": df["How did you contact us"].str.lower().map(SOURCES),
        "city": clean_city(df["City"]),
        "created_at": pd.to_datetime(df["Submitted On"], format="mixed"),
    })
    return add_coords(df)


def clean_volunteers(path):
    df = pd.read_excel(path)
    df["email"] = df["Email"].str.strip().str.lower()
    df = df.drop_duplicates(subset="email")
    df = pd.DataFrame({
        "name": df["Name"].str.strip(),
        "email": df["email"],
        "skills": df["Can help with"].map(lambda s: sorted({SKILLS[x.strip().lower()] for x in s.split(",")})),
        "availability": df["Availability"].str.strip().str.lower().map(AVAILABILITY),
        "city": clean_city(df["City"]),
        "active": df["Active?"].fillna("").str.lower().isin(["y", "yes"]),
    })
    return add_coords(df)


def clean_cases(path):
    df = pd.DataFrame(json.loads(open(path).read()))
    cases = pd.DataFrame({
        "submission_id": df["submission_id"],
        "volunteer_email": df["volunteer_email"].str.lower(),
        "status": df["outcome"].str.lower().fillna("open"),  # no outcome yet means the case is still open
        "created_at": pd.to_datetime(df["opened"]),
        "closed_at": pd.to_datetime(df["closed"]),
    })
    cases.loc[cases["status"] == "open", "closed_at"] = None  # an open case can't have a closed date
    return cases


def records(df):
    """DataFrame rows as dicts, with NaN/NaT turned into None for the database."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def load(requests, volunteers, cases):
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE requests, volunteers, cases, audit_log RESTART IDENTITY"))

    with SessionLocal() as session:
        request_rows = [AidRequest(**r) for r in records(requests)]
        volunteer_rows = [Volunteer(**v) for v in records(volunteers)]
        session.add_all(request_rows + volunteer_rows)
        session.flush()  # fills in the new ids

        request_ids = {r.submission_id: r.id for r in request_rows}
        volunteer_ids = {v.email: v.id for v in volunteer_rows}
        cases["request_id"] = cases["submission_id"].map(request_ids)
        cases["volunteer_id"] = cases["volunteer_email"].map(volunteer_ids)
        cases = cases.dropna(subset=["request_id"])  # the request was dropped (empty text)

        columns = ["request_id", "volunteer_id", "status", "created_at", "closed_at"]
        session.add_all([Case(**c) for c in records(cases[columns])])
        session.commit()
    return len(cases)


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    requests = clean_intake("data/raw/intake_form_export.csv")
    volunteers = clean_volunteers("data/raw/volunteers.xlsx")
    cases = clean_cases("data/raw/case_management.json")
    n_cases = load(requests, volunteers, cases)
    print(f"loaded requests={len(requests)} volunteers={len(volunteers)} cases={n_cases}")
