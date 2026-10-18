import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Mesh API serves Llama 3 behind an OpenAI-compatible API, so one client library works for both
MODELS = {
    "gpt-4o": {"model": "gpt-4o", "base_url": None, "api_key": "OPENAI_API_KEY"},
    "llama3": {"model": "meta-llama/llama-3.3-70b-instruct-turbo", "base_url": "https://api.meshapi.ai/v1",
               "api_key": "MESH_API_KEY"},
}


def get_client(name):
    """Returns (client, model_id) for 'gpt-4o' or 'llama3'."""
    config = MODELS[name]
    client = OpenAI(api_key=os.environ[config["api_key"]], base_url=config["base_url"])
    return client, config["model"]


if __name__ == "__main__":
    for name in MODELS:
        client, model = get_client(name)
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in five words."}],
        )
        print(f"{name:7} {model:25} -> {reply.choices[0].message.content}")
