"""Builds the match_features table that the volunteer match model is trained on.

One row per past case: what the request and the volunteer looked like at the time,
and whether the volunteer accepted (1) or declined (0).

python -m pipelines.build_features
"""
from collections import defaultdict

import pandas as pd
from sqlalchemy import select

from app.db import SessionLocal, engine
from app.matching import DEFAULT_ACCEPTANCE_RATE, make_features
from app.models import AidRequest, Case, Volunteer


def build():
    with SessionLocal() as session:
        cases = session.execute(
            select(Case, AidRequest, Volunteer)
            .join(AidRequest, AidRequest.id == Case.request_id)
            .join(Volunteer, Volunteer.id == Case.volunteer_id)
            .where(Case.status.in_(["accepted", "declined"]), AidRequest.category.is_not(None))
            .order_by(Case.created_at, Case.id)
        ).all()

        # Cases are in date order. For each one, the volunteer's acceptance rate is worked out
        # from their earlier cases only. Using later cases would leak the answer into the features.
        accepted_before = defaultdict(int)
        cases_before = defaultdict(int)

        rows = []
        for case, request, volunteer in cases:
            if cases_before[volunteer.id] == 0:
                past_rate = DEFAULT_ACCEPTANCE_RATE
            else:
                past_rate = accepted_before[volunteer.id] / cases_before[volunteer.id]

            accepted = int(case.status == "accepted")
            row = make_features(request, volunteer, past_rate)
            row["case_id"] = case.id
            row["created_at"] = case.created_at
            row["accepted"] = accepted
            rows.append(row)

            cases_before[volunteer.id] += 1
            accepted_before[volunteer.id] += accepted

    features = pd.DataFrame(rows)
    features.to_sql("match_features", engine, if_exists="replace", index=False)
    return features


if __name__ == "__main__":
    features = build()
    print(f"match_features: {len(features)} rows, {features['accepted'].mean():.0%} accepted")
    print(features.drop(columns=["case_id", "created_at"]).describe().loc[["mean", "min", "max"]].round(2).to_string())
