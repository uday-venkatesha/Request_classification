"""Cleans the 3 source system exports and loads them into Postgres.

Safe to re-run: rows that are already in the database are skipped or updated, never duplicated.
"""
import json

import pandas as pd
from sqlalchemy import select

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
        "case_ref": df["case_ref"],
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
    """Adds new rows and updates changed ones. Nothing is ever deleted, so requests that
    came in through the API and the audit log are safe when this runs again."""
    with SessionLocal() as session:
        # Requests: the intake form never changes old submissions, so we only add new ones
        known_submissions = set(session.scalars(select(AidRequest.submission_id)))
        new_requests = requests[~requests["submission_id"].isin(known_submissions)]
        session.add_all([AidRequest(**row) for row in records(new_requests)])

        # Volunteers: staff edit the spreadsheet (skills, active or not), so existing rows are updated
        existing_volunteers = {v.email: v for v in session.scalars(select(Volunteer))}
        for row in records(volunteers):
            volunteer = existing_volunteers.get(row["email"])
            if volunteer is None:
                session.add(Volunteer(**row))
            else:
                for column, value in row.items():
                    setattr(volunteer, column, value)

        session.flush()  # gives the new rows their ids

        # Cases point at requests and volunteers by the old systems' keys,
        # so we look up our own ids before saving them
        request_ids = dict(session.execute(select(AidRequest.submission_id, AidRequest.id)).all())
        volunteer_ids = dict(session.execute(select(Volunteer.email, Volunteer.id)).all())
        existing_cases = {c.case_ref: c for c in session.scalars(select(Case).where(Case.case_ref.is_not(None)))}

        for row in records(cases):
            request_id = request_ids.get(row["submission_id"])
            if request_id is None:  # its request was dropped because the text was empty
                continue
            case = existing_cases.get(row["case_ref"])
            if case is None:
                session.add(Case(
                    case_ref=row["case_ref"], request_id=request_id,
                    volunteer_id=volunteer_ids.get(row["volunteer_email"]),
                    status=row["status"], created_at=row["created_at"], closed_at=row["closed_at"],
                ))
            else:  # an open case may have been accepted or declined since the last run
                case.status = row["status"]
                case.closed_at = row["closed_at"]

        session.commit()
        return session.query(AidRequest).count(), session.query(Volunteer).count(), session.query(Case).count()


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    requests = clean_intake("data/raw/intake_form_export.csv")
    volunteers = clean_volunteers("data/raw/volunteers.xlsx")
    cases = clean_cases("data/raw/case_management.json")
    n_requests, n_volunteers, n_cases = load(requests, volunteers, cases)
    print(f"in database: requests={n_requests} volunteers={n_volunteers} cases={n_cases}")
