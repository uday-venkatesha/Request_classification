"""Measures how much the guardrails reduce made-up answers.

The same questions are answered twice:
  baseline   - plain prompt (prompts/rag_v0.txt), no distance or citation check
  guardrails - grounded prompt (prompts/rag_v1.txt) with all checks on

Some questions can be answered from the program documents and some cannot.
A good assistant answers the first kind correctly and refuses the second kind.

python -m evals.eval_rag
python -m evals.eval_rag --model llama3
"""
import argparse
from pathlib import Path

import pandas as pd
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference

from app import rag

RESULTS = Path("data/eval/results")

# RAGAS uses an LLM as the judge. A small model is enough for this and keeps the cost low.
judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))


def is_refusal(answer_text):
    return "have that information" in answer_text


def collect_answers(questions, model_name, prompt_version, guardrails):
    rows = []
    for _, q in questions.iterrows():
        result = rag.answer(q["question"], model_name, prompt_version, guardrails)
        chunks = rag.retrieve(q["question"])
        rows.append({
            "question": q["question"],
            "reference": q["reference"],
            "answerable": q["answerable"],
            "answer": result["answer"],
            "guardrail": result["guardrail"],
            "refused": is_refusal(result["answer"]),
            "contexts": [chunk.page_content for chunk, distance in chunks],
        })
    return pd.DataFrame(rows)


def ragas_score(rows, metric):
    # Turns our rows into the format RAGAS expects and returns the average score
    samples = []
    for _, r in rows.iterrows():
        samples.append({
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["reference"],
        })
    result = evaluate(EvaluationDataset.from_list(samples), metrics=[metric], llm=judge, show_progress=False)
    return result.to_pandas()[metric.name].mean()


def evaluate_config(name, questions, model_name, prompt_version, guardrails):
    answers = collect_answers(questions, model_name, prompt_version, guardrails)
    answerable = answers[answers["answerable"] == 1]
    unanswerable = answers[answers["answerable"] == 0]

    # Faithfulness: how much of each answer is actually supported by the retrieved chunks.
    # It is measured on every answer the assistant gave (refusals have nothing to check).
    given = answers[~answers["refused"]]

    scores = {
        # Questions the docs cannot answer, but the assistant answered anyway = made up
        "made_up_rate": 1 - unanswerable["refused"].mean(),
        # Questions the docs can answer, but the assistant refused = too strict
        "wrongly_refused_rate": answerable["refused"].mean(),
        "faithfulness": ragas_score(given, Faithfulness()),
    }
    answers.drop(columns="contexts").to_csv(RESULTS / f"rag_{name}_{model_name}.csv", index=False)
    return scores, answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    questions = pd.read_csv("data/eval/rag_questions.csv").fillna("")

    baseline, baseline_answers = evaluate_config("baseline", questions, args.model, "v0", guardrails=False)
    guarded, _ = evaluate_config("guardrails", questions, args.model, "v1", guardrails=True)

    # Context precision checks the search step: are the chunks that contain the answer ranked first?
    # Retrieval is the same in both runs, so it is measured once.
    answerable = baseline_answers[baseline_answers["answerable"] == 1]
    context_precision = ragas_score(answerable, LLMContextPrecisionWithReference())

    print(f"\n===== RAG evaluation | {args.model} | {len(questions)} questions =====")
    print(f"context precision (retrieval): {context_precision:.2f}\n")
    table = pd.DataFrame({"baseline": baseline, "guardrails": guarded})
    print(table.round(2).to_string())
