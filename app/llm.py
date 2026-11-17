import os
import re

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


def mask_pii(text):
    # People often type their phone number or email into a help form.
    # We replace them before the text is sent to OpenAI or Mesh API,
    # so personal data never leaves our system.
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", text)            # maria@gmail.com
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ssn]", text)                 # 123-45-6789
    text = re.sub(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[phone]", text)  # (512) 555-0100
    return text


if __name__ == "__main__":
    for name in MODELS:
        client, model = get_client(name)
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in five words."}],
        )
        print(f"{name:7} {model:25} -> {reply.choices[0].message.content}")
