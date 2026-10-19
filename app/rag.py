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


def answer(question, model_name="gpt-4o", prompt_version="v1"):
    results = retrieve(question)
    context = "\n\n".join(f"[{i}] {doc.page_content}" for i, (doc, _) in enumerate(results, start=1))
    system = (PROMPTS / f"rag_{prompt_version}.txt").read_text().format(context=context)

    client, model = get_client(model_name)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        temperature=0,
    )
    sources = [{"n": i, "title": doc.page_content.splitlines()[0], "file": doc.metadata["source"],
                "distance": round(distance, 3)} for i, (doc, distance) in enumerate(results, start=1)]
    return {"answer": response.choices[0].message.content, "sources": sources}


if __name__ == "__main__":
    for question in [
        "How often can I go to the food pantry?",
        "My landlord changed the locks, is that allowed?",
        "Can I bring my dog to the shelter?",
        "How much do you pay for prescriptions?",
        "What is the capital of France?",
    ]:
        result = answer(question)
        print(f"\nQ: {question}\nA: {result['answer']}")
        for s in result["sources"][:2]:
            print(f"   [{s['n']}] {s['distance']}  {s['title']}")
