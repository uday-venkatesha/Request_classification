"""Volunteer matching: ranks volunteers for a request by how likely they are to accept it.

A small logistic regression is trained on past cases (accepted or declined).
It is retrained when the drift check (evals/drift_check.py) sees that the data has changed.
"""
import io
import math

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func, select

from app.db import SessionLocal, engine
from app.models import Case, MatchModel, Volunteer

FEATURES = ["skill_match", "same_city", "distance_km", "skill_count", "availability_count",
            "past_acceptance_rate", "high_urgency"]

# Used when a request or volunteer has no known city. About the average distance between our cities.
UNKNOWN_DISTANCE_KM = 40

# A volunteer with no history yet is treated as average
DEFAULT_ACCEPTANCE_RATE = 0.5


def distance_km(lat1, lon1, lat2, lon2):
    # Haversine formula: straight-line distance between two points on the globe
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def make_features(request, volunteer, past_acceptance_rate):
    # One row of model inputs for a (request, volunteer) pair.
    # Training (pipelines/build_features.py) and live ranking both call this function,
    # so the model always sees features that were calculated in exactly the same way.
    if request.lat is None or volunteer.lat is None:
        distance = UNKNOWN_DISTANCE_KM
    else:
        distance = distance_km(request.lat, request.lon, volunteer.lat, volunteer.lon)

    return {
        "skill_match": int(request.category in volunteer.skills),
        "same_city": int(request.city is not None and request.city == volunteer.city),
        "distance_km": round(distance, 1),
        "skill_count": len(volunteer.skills),
        "availability_count": len(volunteer.availability),
        "past_acceptance_rate": past_acceptance_rate,
        "high_urgency": int(request.urgency == "high"),
    }


def train(features=None, save=True):
    """Trains a new model and saves it to the match_models table. Returns (model, auc)."""
    if features is None:
        features = pd.read_sql("SELECT * FROM match_features", engine)

    # Keep a quarter of the cases aside to check the model on data it has not seen
    x_train, x_test, y_train, y_test = train_test_split(
        features[FEATURES], features["accepted"], test_size=0.25, random_state=42, stratify=features["accepted"],
    )
    # Scaling first, because distance is in km while the other features are 0/1 or small counts
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(x_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    if save:
        saved = io.BytesIO()
        joblib.dump(model, saved)
        with SessionLocal() as session:
            session.add(MatchModel(auc=auc, trained_on_rows=len(features), data=saved.getvalue()))
            session.commit()
    return model, auc


# The model is kept in memory and only read from the database again when a newer one exists
loaded = {"id": None, "model": None}


def load_model():
    with SessionLocal() as session:
        latest_id = session.scalar(select(func.max(MatchModel.id)))
        if latest_id is None:
            raise RuntimeError("No match model has been trained yet. Run: python -m app.matching")
        if latest_id != loaded["id"]:
            row = session.get(MatchModel, latest_id)
            loaded["model"] = joblib.load(io.BytesIO(row.data))
            loaded["id"] = latest_id
    return loaded["model"]


def acceptance_rates(session):
    # Share of past cases each volunteer accepted
    rows = session.execute(
        select(Case.volunteer_id, func.count().filter(Case.status == "accepted"), func.count())
        .where(Case.status.in_(["accepted", "declined"]))
        .group_by(Case.volunteer_id)
    ).all()
    return {volunteer_id: accepted / total for volunteer_id, accepted, total in rows}


def top_volunteers(request, k=3):
    """The k active volunteers most likely to accept this request, best first.
    Only volunteers who have the needed skill are considered, so the list can be empty."""
    model = load_model()
    with SessionLocal() as session:
        rates = acceptance_rates(session)
        volunteers = session.scalars(select(Volunteer).where(Volunteer.active)).all()
        # Volunteers who already declined this request are not asked again
        declined = set(session.scalars(
            select(Case.volunteer_id).where(Case.request_id == request.id, Case.status == "declined")
        ))

    candidates = [v for v in volunteers if request.category in v.skills and v.id not in declined]
    if not candidates:
        return []

    rows = [make_features(request, v, rates.get(v.id, DEFAULT_ACCEPTANCE_RATE)) for v in candidates]
    scores = model.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]

    ranked = sorted(zip(candidates, scores, rows), key=lambda item: item[1], reverse=True)
    return [
        {"volunteer_id": v.id, "name": v.name, "city": v.city, "distance_km": row["distance_km"],
         "score": round(float(score), 2)}
        for v, score, row in ranked[:k]
    ]


if __name__ == "__main__":
    model, auc = train()
    print(f"trained match model, AUC on held-out cases: {auc:.2f}")

    # What the model learned: a positive number makes acceptance more likely
    weights = model.named_steps["logisticregression"].coef_[0]
    for name, weight in sorted(zip(FEATURES, weights), key=lambda item: -abs(item[1])):
        print(f"  {name:22} {weight:+.2f}")
