from pathlib import Path

from app.llm import get_client
from app.schemas import Classification

PROMPTS = Path(__file__).parent.parent / "prompts"


def load_prompt(version):
    return (PROMPTS / f"classifier_{version}.txt").read_text()


def classify(text, model_name="gpt-4o", prompt_version="v1"):
    client, model = get_client(model_name)
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": load_prompt(prompt_version)},
            {"role": "user", "content": text},
        ],
        response_format=Classification,
        temperature=0,
    )
    return response.choices[0].message.parsed


if __name__ == "__main__":
    samples = [
        "Our power got cut and there is nothing in the fridge for the kids.",
        "I need a lawyer, ICE took my brother and we don't know where he is.",
        "can someone help my daughter get ready for her GED test",
        "I keep having panic attacks at work and I can't sleep.",
        "Got kicked out tonight, nowhere to go, also I'm out of my blood pressure pills.",
    ]
    for model_name in ["gpt-4o", "llama3"]:
        print(f"\n== {model_name}")
        for text in samples:
            result = classify(text, model_name)
            print(f"{result.category:14} {result.urgency:6} {result.confidence:.2f}  {text[:55]}")
