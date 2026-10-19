"""Measures how well the classifier does on the labeled requests.

python -m evals.eval_classifier                      # both models, all 600 requests
python -m evals.eval_classifier --limit 30           # quick check
python -m evals.eval_classifier --models llama3 --prompt v1
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from app.classifier import classify

RESULTS = Path("data/eval/results")


def predict(text, model_name, prompt_version):
    try:
        r = classify(text, model_name, prompt_version)
        return r.category, r.urgency, r.confidence
    except Exception as e:  # count a failed call as a wrong answer instead of stopping the run
        print("  error:", str(e)[:100])
        return "error", "error", 0.0


def evaluate(df, model_name, prompt_version):
    with ThreadPoolExecutor(max_workers=6) as pool:
        preds = list(pool.map(lambda t: predict(t, model_name, prompt_version), df["text"]))

    df = df.copy()
    df["pred_category"], df["pred_urgency"], df["confidence"] = zip(*preds)
    df["correct"] = df["pred_category"] == df["category"]

    print(f"\n===== {model_name} | prompt {prompt_version} | {len(df)} requests =====")
    print(f"category accuracy: {df['correct'].mean():.1%}")
    print(f"urgency accuracy:  {(df['pred_urgency'] == df['urgency']).mean():.1%}")
    print(f"failed calls:      {(df['pred_category'] == 'error').sum()}")
    print(f"avg confidence when right: {df.loc[df['correct'], 'confidence'].mean():.2f}, "
          f"when wrong: {df.loc[~df['correct'], 'confidence'].mean():.2f}")

    print("\naccuracy per category:")
    print(df.groupby("category")["correct"].mean().map("{:.0%}".format).to_string())

    print("\nconfusion matrix (rows = true, columns = predicted):")
    print(pd.crosstab(df["category"], df["pred_category"]).to_string())

    print("\nsome mistakes:")
    for _, r in df[~df["correct"]].head(5).iterrows():
        print(f"  true={r['category']:13} pred={r['pred_category']:13} {r['text'][:90]!r}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"{model_name}_{prompt_version}.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "llama3"])
    parser.add_argument("--prompt", default="v1")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    data = pd.read_csv("data/eval/labeled_requests.csv")
    if args.limit:  # a small sample with every category in it
        data = data.groupby("category").head(args.limit // 6)

    for model_name in args.models:
        evaluate(data, model_name, args.prompt)
