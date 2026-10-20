import re
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.db import engine
from app.llm import get_client

DOCS = Path(__file__).parent.parent / "data" / "docs"
PROMPTS = Path(__file__).parent.parent / "prompts"
COLLECTION = "program_docs"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# in testing, real questions matched at 0.30-0.63 and off-topic ones at 0.62-0.93
MAX_DISTANCE = 0.70
FALLBACK = "I don't have that information. Please call our helpline at (512) 555-0100."
CRISIS = ("If you are thinking about suicide or are in danger, please call or text 988 (Suicide and Crisis "
          "Lifeline) or call 911 right now. During office hours you can also call us at (512) 555-0100.")
CRISIS_WORDS = ["suicide", "suicidal", "kill myself", "end my life", "want to die", "hurt myself", "self harm",
                "self-harm"]


def load_chunks():
    """One chunk per '##' section. The doc title and section name are put at the top of each
    chunk so a section like 'How often can I visit?' still says which program it belongs to."""
    splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")])
    chunks = []
    for path in sorted(DOCS.glob("*.md")):
        for chunk in splitter.split_text(path.read_text()):
            title = chunk.metadata.get("title", "")
            section = chunk.metadata.get("section", "Overview")
            chunk.page_content = f"{title} - {section}\n{chunk.page_content}"
            chunk.metadata["source"] = path.name
            chunks.append(chunk)
    return chunks


def ingest_docs():
    """Embeds all docs into pgvector. Deletes the old collection first, so it is safe to re-run."""
    chunks = load_chunks()
    PGVector.from_documents(
        chunks, embeddings, collection_name=COLLECTION, connection=engine,
        pre_delete_collection=True, use_jsonb=True,
    )
    return len(chunks)


def get_store():
    return PGVector(embeddings, collection_name=COLLECTION, connection=engine, use_jsonb=True)


def retrieve(question, k=4):
    """The k closest chunks as (doc, distance). Lower distance means a closer match."""
    return get_store().similarity_search_with_score(question, k=k)


def mask_pii(text):
    """Hide emails, SSNs and phone numbers so they are never sent to the LLM."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", text)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ssn]", text)
    return re.sub(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[phone]", text)


def answer(question, model_name="gpt-4o", prompt_version="v1", guardrails=True):
    """guardrails=False skips the distance and citation checks, used to measure their effect.
    PII masking and the crisis message always apply."""
    question = mask_pii(question)
    if any(word in question.lower() for word in CRISIS_WORDS):
        return {"answer": CRISIS, "sources": [], "guardrail": "crisis"}

    results = retrieve(question)
    if guardrails and results[0][1] > MAX_DISTANCE:
        return {"answer": FALLBACK, "sources": [], "guardrail": "off_topic"}

    context = "\n\n".join(f"[{i}] {doc.page_content}" for i, (doc, _) in enumerate(results, start=1))
    system = (PROMPTS / f"rag_{prompt_version}.txt").read_text().format(context=context)
    client, model = get_client(model_name)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        temperature=0,
    )
    text = response.choices[0].message.content
    sources = [{"n": i, "title": doc.page_content.splitlines()[0], "file": doc.metadata["source"],
                "distance": round(distance, 3)} for i, (doc, distance) in enumerate(results, start=1)]

    # an answer with no [n] citation is not grounded in our docs, so we do not show it
    if guardrails and "have that information" not in text and not re.search(r"\[\d+\]", text):
        return {"answer": FALLBACK, "sources": [], "guardrail": "no_citation"}
    return {"answer": text, "sources": sources, "guardrail": None}


if __name__ == "__main__":
    print(mask_pii("Call me at (512) 867-5309 or 512.555.1234, email maria.g@gmail.com, SSN 123-45-6789"))
    for question in [
        "How often can I go to the food pantry?",
        "I'm Maria, 737-555-2211, can my kids get tutoring?",
        "What is the capital of France?",
        "What's the best pizza in Austin?",
        "I lost my job and I want to end my life",
    ]:
        result = answer(question)
        print(f"\nQ: {question}\nA: {result['answer']}\n   guardrail: {result['guardrail']}")
