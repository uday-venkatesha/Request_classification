"""Checks whether the volunteer matching data has changed, and retrains the model if it has.

Past cases are split by date: the older ones are the reference, the most recent ones are
"current". Evidently compares the two. The model is retrained when
  - the data has drifted (most columns look different from before), or
  - the model no longer ranks recent cases well (AUC below MIN_AUC).

python -m evals.drift_check
python -m evals.drift_check --simulate-drift     # fakes a change to show the retraining path
"""
import argparse
from pathlib import Path

import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sklearn.metrics import roc_auc_score

from app.matching import FEATURES, load_model, train
from pipelines.build_features import build

REPORTS = Path("reports")
RECENT_SHARE = 0.3   # the newest 30% of cases are compared with the older 70%
MIN_AUC = 0.65       # below this the model is not much better than guessing (0.5)


def simulate_drift(current):
    # Pretends that volunteers moved further away and started declining most cases
    current = current.copy()
    current["same_city"] = 0
    current["distance_km"] = current["distance_km"] + 45
    current["past_acceptance_rate"] = current["past_acceptance_rate"] * 0.4
    accepted = current[current["accepted"] == 1].sample(frac=0.7, random_state=1).index
    current.loc[accepted, "accepted"] = 0
    return current


def check(simulate=False):
    features = build().sort_values("created_at")  # rebuilt first, so new cases are included
    split = int(len(features) * (1 - RECENT_SHARE))
    reference, current = features[:split], features[split:]
    if simulate:
        current = simulate_drift(current)

    # 1. Has the data changed?
    columns = FEATURES + ["accepted"]
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[columns], current_data=current[columns])
    REPORTS.mkdir(exist_ok=True)
    report_path = REPORTS / "match_drift.html"
    report.save_html(str(report_path))

    summary, per_column = [metric["result"] for metric in report.as_dict()["metrics"]]
    drifted_columns = [name for name, column in per_column["drift_by_columns"].items() if column["drift_detected"]]

    # 2. Is the model still good on recent cases?
    model = load_model()
    current_auc = roc_auc_score(current["accepted"], model.predict_proba(current[FEATURES])[:, 1])

    retrain = summary["dataset_drift"] or current_auc < MIN_AUC

    print(f"reference cases: {len(reference)}, current cases: {len(current)}")
    print(f"drifted columns: {drifted_columns or 'none'}")
    print(f"dataset drift:   {summary['dataset_drift']}")
    print(f"model AUC on current cases: {current_auc:.2f} (retrain below {MIN_AUC})")

    mlflow.set_experiment("match-model-monitoring")
    with mlflow.start_run(run_name="simulated-drift" if simulate else "drift-check"):
        mlflow.log_metrics({
            "share_of_drifted_columns": summary["share_of_drifted_columns"],
            "acceptance_rate_reference": reference["accepted"].mean(),
            "acceptance_rate_current": current["accepted"].mean(),
            "auc_on_current": current_auc,
            "retrained": int(retrain),
        })
        mlflow.log_artifact(str(report_path))

        if retrain:
            # Retrain on everything we have, including the cases that look different.
            # A model trained on simulated data is never saved, so it cannot reach the live service.
            _, new_auc = train(pd.concat([reference, current]), save=not simulate)
            mlflow.log_metric("auc_after_retraining", new_auc)
            print(f"retrained: AUC {new_auc:.2f}" + (" (simulation, model not saved)" if simulate else ", new model saved"))
        else:
            print("no retraining needed")
    return retrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-drift", action="store_true")
    args = parser.parse_args()
    check(args.simulate_drift)
