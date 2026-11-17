"""RAG assistant: answers questions using only our program documents.

How it works:
1. Ingest (run once): split the docs into small chunks, turn each chunk into a vector
   (a list of numbers that captures its meaning) and save them in Postgres (pgvector).
2. Answer (every question): find the chunks closest in meaning to the question,
   give them to the LLM and tell it to answer only from them.

Guardrails stop the assistant from making things up or leaking personal data.
"""
import re
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.db import engine
from app.llm import get_client

DOCS_FOLDER = Path(__file__).parent.parent / "data" / "docs"
PROMPTS_FOLDER = Path(__file__).parent.parent / "prompts"

# All our chunks are stored under this name in the database
COLLECTION = "program_docs"

# The model that turns text into vectors. The same model must be used when saving
# the chunks and when searching, otherwise the numbers would not be comparable.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# If the closest chunk is further away than this, the question is not about our programs.
# Chosen by testing: real questions matched at 0.30-0.63, off-topic ones at 0.62-0.93.
MAX_DISTANCE = 0.70

FALLBACK_ANSWER = "I don't have that information. Please call our helpline at (512) 555-0100."

CRISIS_ANSWER = (
    "If you are thinking about suicide or are in danger, please call or text 988 "
    "(Suicide and Crisis Lifeline) or call 911 right now. "
    "During office hours you can also call us at (512) 555-0100."
)
CRISIS_WORDS = ["suicide", "suicidal", "kill myself", "end my life", "want to die",
                "hurt myself", "self harm", "self-harm"]


# ---------- Part 1: saving the documents ----------

def load_chunks():
    # We split each document at its "##" headings, so every section becomes one chunk.
    # Small chunks let the search return exactly the part that answers the question,
    # instead of a whole document.
    splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")])

    chunks = []
    for file in sorted(DOCS_FOLDER.glob("*.md")):
        for chunk in splitter.split_text(file.read_text()):
            title = chunk.metadata.get("title", "")
            section = chunk.metadata.get("section", "Overview")

            # A section called "How often can I visit?" doesn't say which program it is about,
            # so we add the document title on top: "Food Pantry Program - How often can I visit?"
            chunk.page_content = f"{title} - {section}\n{chunk.page_content}"

            # Remember the file name so the answer can say where it came from
            chunk.metadata["source"] = file.name
            chunks.append(chunk)
    return chunks


def ingest_docs():
    chunks = load_chunks()

    # pre_delete_collection=True removes the old chunks first,
    # so running this again after editing a document never creates duplicates
    PGVector.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION,
        connection=engine,
        pre_delete_collection=True,
        use_jsonb=True,
    )
    return len(chunks)


# ---------- Part 2: searching ----------

def retrieve(question, k=4):
    # Returns the k closest chunks as (chunk, distance) pairs.
    # Lower distance = closer in meaning to the question.
    store = PGVector(embeddings, collection_name=COLLECTION, connection=engine, use_jsonb=True)
    return store.similarity_search_with_score(question, k=k)


# ---------- Part 3: guardrails ----------

def mask_pii(text):
    # People often type their phone number or email into a help form.
    # We replace them before the question is sent to OpenAI or Mesh API,
    # so personal data never leaves our system.
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", text)            # maria@gmail.com
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ssn]", text)                 # 123-45-6789
    text = re.sub(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[phone]", text)  # (512) 555-0100
    return text


def is_crisis(question):
    # Someone in crisis must always get the 988 number, straight away.
    # We don't leave this to the search or the LLM, because either could get it wrong.
    question = question.lower()
    for word in CRISIS_WORDS:
        if word in question:
            return True
    return False


def has_citation(answer_text):
    # The prompt tells the LLM to cite sources like [1] or [2].
    # An answer with no citation didn't come from our documents, so it may be made up.
    if "have that information" in answer_text:  # the LLM's own "I don't know" reply is fine
        return True
    return re.search(r"\[\d+\]", answer_text) is not None


# ---------- Part 4: answering ----------

def build_prompt(chunks, prompt_version):
    # Number the chunks [1], [2], ... so the LLM can cite them
    context = ""
    for number, (chunk, distance) in enumerate(chunks, start=1):
        context += f"[{number}] {chunk.page_content}\n\n"

    prompt = (PROMPTS_FOLDER / f"rag_{prompt_version}.txt").read_text()
    return prompt.format(context=context)


def list_sources(chunks):
    # The chunks we used, so the person (and later the audit log) can see where the answer came from
    sources = []
    for number, (chunk, distance) in enumerate(chunks, start=1):
        sources.append({
            "n": number,
            "title": chunk.page_content.splitlines()[0],
            "file": chunk.metadata["source"],
            "distance": round(distance, 3),
        })
    return sources


def run_checks(question, guardrails=True):
    # Everything that happens before the LLM is called. Both answer() and answer_stream() use it,
    # so the guardrails are identical for normal and streamed answers.
    # Returns (masked question, chunks, early_result). early_result is set when a guardrail
    # has already decided the reply and the LLM must not be called.

    # Step 1: hide personal data
    question = mask_pii(question)

    # Step 2: crisis check comes first, before anything else can go wrong
    if is_crisis(question):
        return question, [], {"answer": CRISIS_ANSWER, "sources": [], "guardrail": "crisis"}

    # Step 3: find the closest chunks. If even the best one is far away,
    # the question isn't about our programs, so we refuse without calling the LLM (free and safe).
    chunks = retrieve(question)
    best_distance = chunks[0][1]
    if guardrails and best_distance > MAX_DISTANCE:
        return question, chunks, {"answer": FALLBACK_ANSWER, "sources": [], "guardrail": "off_topic"}

    return question, chunks, None


def build_messages(question, chunks, prompt_version):
    return [
        {"role": "system", "content": build_prompt(chunks, prompt_version)},
        {"role": "user", "content": question},
    ]


def answer(question, model_name="gpt-4o", prompt_version="v1", guardrails=True):
    # guardrails=False turns off the distance and citation checks.
    # We only do that in the evaluation (evals/eval_rag.py) to measure how much the guardrails help.
    # PII masking and the crisis check are never turned off.
    question, chunks, early_result = run_checks(question, guardrails)
    if early_result:
        return early_result

    # Step 4: ask the LLM to answer using only these chunks.
    # temperature=0 so the same question always gets the same answer.
    client, model = get_client(model_name)
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(question, chunks, prompt_version),
        temperature=0,
    )
    answer_text = response.choices[0].message.content

    # Step 5: don't show an answer that isn't backed by a source
    if guardrails and not has_citation(answer_text):
        return {"answer": FALLBACK_ANSWER, "sources": [], "guardrail": "no_citation"}

    return {"answer": answer_text, "sources": list_sources(chunks), "guardrail": None}


def answer_stream(question, model_name="gpt-4o", prompt_version="v1"):
    # Same as answer(), but gives the text back piece by piece so the person sees
    # the first words in under a second instead of waiting for the whole answer.
    # Yields {"token": "..."} for each piece and one final {"done": True, ...} event.
    question, chunks, early_result = run_checks(question)
    if early_result:
        yield {"token": early_result["answer"]}
        yield {"done": True, "sources": [], "guardrail": early_result["guardrail"]}
        return

    client, model = get_client(model_name)
    stream = client.chat.completions.create(
        model=model,
        messages=build_messages(question, chunks, prompt_version),
        temperature=0,
        stream=True,
    )
    answer_text = ""
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            token = event.choices[0].delta.content
            answer_text += token
            yield {"token": token}

    # The citation check needs the whole answer, so it can only run at the end.
    # If it fails, "replace_with" tells the page to swap the streamed text for the fallback.
    if not has_citation(answer_text):
        yield {"done": True, "sources": [], "guardrail": "no_citation", "replace_with": FALLBACK_ANSWER}
        return

    yield {"done": True, "sources": list_sources(chunks), "guardrail": None}


if __name__ == "__main__":
    print(mask_pii("Call me at (512) 867-5309 or 512.555.1234, email maria.g@gmail.com, SSN 123-45-6789"))

    test_questions = [
        "How often can I go to the food pantry?",
        "I'm Maria, 737-555-2211, can my kids get tutoring?",
        "What is the capital of France?",
        "What's the best pizza in Austin?",
        "I lost my job and I want to end my life",
    ]
    for question in test_questions:
        result = answer(question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"   guardrail: {result['guardrail']}")
