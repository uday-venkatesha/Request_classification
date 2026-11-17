"""Classifies requests that have no category yet.

Requests imported from the old intake form were never categorised. The matching
model needs a category for every past request, so this fills them in once.

python -m pipelines.classify_backlog
python -m pipelines.classify_backlog --model gpt-4o
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from app.classifier import classify
from app.db import SessionLocal
from app.models import AidRequest


def classify_one(request_id, text, model_name):
    try:
        return request_id, classify(text, model_name)
    except Exception as e:  # leave it without a category, the next run will try again
        print(f"  request {request_id} failed: {str(e)[:80]}")
        return request_id, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3")
    args = parser.parse_args()

    with SessionLocal() as session:
        backlog = session.execute(select(AidRequest.id, AidRequest.text).where(AidRequest.category.is_(None))).all()
        print(f"{len(backlog)} requests to classify with {args.model}")

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = pool.map(lambda row: classify_one(row.id, row.text, args.model), backlog)

            done = 0
            for request_id, result in results:
                if result is None:
                    continue
                request = session.get(AidRequest, request_id)
                request.category = result.category
                request.urgency = result.urgency
                request.confidence = result.confidence
                done += 1
                if done % 100 == 0:
                    session.commit()  # save progress as we go
                    print(f"  {done} done")
        session.commit()
        print(f"classified {done} of {len(backlog)}")
