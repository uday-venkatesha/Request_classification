from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.db import engine

DOCS = Path(__file__).parent.parent / "data" / "docs"
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


if __name__ == "__main__":
    print("chunks stored:", ingest_docs())

    store = get_store()
    for question in ["How often can I go to the food pantry?", "Can I bring my dog to the shelter?",
                     "Do you help with criminal cases?"]:
        print(f"\nQ: {question}")
        for doc, distance in store.similarity_search_with_score(question, k=2):
            print(f"  {distance:.3f}  {doc.page_content.splitlines()[0]}  ({doc.metadata['source']})")
