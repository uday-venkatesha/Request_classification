# AidRoute

AidRoute helps a nonprofit handle incoming requests for help. It classifies each request into one of six
service areas, routes it to the right team, matches a volunteer, answers questions from the program
documents, and sends anything sensitive or uncertain to a human reviewer. Every decision is logged.

The six service areas are food, shelter, medical, legal, education and mental health.

## Architecture

```mermaid
flowchart TB
    request([Aid request<br/>web form, phone, walk-in]) --> api
    question([Question about programs]) --> api
    reviewer([Human reviewer]) --> api

    subgraph api [FastAPI service]
        classifier[Classifier<br/>few-shot prompt, structured output]
        agent[Triage agent<br/>tool-calling loop, review rules, audit log]
        rag[RAG assistant<br/>guardrails, citations, streaming]
        matching[Volunteer match model<br/>logistic regression]
        agent --> classifier
        agent --> rag
        agent --> matching
    end

    classifier --> llm[GPT-4o / Llama 3.3]
    agent --> llm
    rag --> llm

    api --> db[(PostgreSQL + pgvector<br/>requests, volunteers, cases,<br/>audit log, document vectors, models)]
    sources[3 source systems<br/>CSV, Excel, JSON] --> etl[ETL pipeline<br/>pandas, SQLAlchemy] --> db
    docs[Program documents] --> ingest[Split, embed] --> db
    db --> metabase[Metabase dashboards]
    db --> drift[Drift check<br/>Evidently] -->|retrain| matching
    evals[Evaluations<br/>accuracy, RAGAS] --> mlflow[MLflow]
```

## What it does

| Part | How it works |
|---|---|
| **Classification** | A few-shot prompt and a Pydantic schema force the model to return a valid category, urgency, confidence and a one-sentence reason. GPT-4o and Llama 3.3 sit behind one client function, so the model is a configuration choice. |
| **Triage agent** | The LLM chooses the next tool: classify, search the documents, find volunteers, create a case, or send to review. The rules for when a human must decide are written in code, not in the prompt. |
| **Human review** | Medical, legal and mental health requests, low-confidence results, requests from outside the service area and crisis language always go to a reviewer, who approves, corrects or rejects. |
| **Audit log** | Each agent step and each human decision is stored with its input, output, model and prompt version. |
| **RAG assistant** | Program documents are split by section and stored in pgvector. Answers use only the retrieved sections and cite them. |
| **Guardrails** | Personal data is masked before any LLM call. Crisis language gets a fixed hotline reply. Off-topic questions are refused before the LLM is called. Answers without a citation are not shown. |
| **Volunteer matching** | A logistic regression trained on past cases ranks volunteers by how likely they are to accept. |
| **Monitoring** | Evidently compares recent cases with older ones. If the data has drifted or the model has got worse, it is retrained. |
| **Data pipeline** | Three source exports (web form CSV, volunteer spreadsheet, case system JSON) are cleaned, de-duplicated and loaded into one database. The load can be re-run safely. |

## Tech stack

Python 3.11, FastAPI, PostgreSQL with pgvector, SQLAlchemy, pandas, OpenAI API, Llama 3.3 through an
OpenAI-compatible API, LangChain (text splitting and vector store), scikit-learn, RAGAS, MLflow, Evidently,
Metabase, Docker, Kubernetes and GitHub Actions.

## Project layout

```
app/
  main.py         API endpoints
  classifier.py   few-shot classification
  agent.py        triage agent, review rules, audit log
  rag.py          document search, guardrails, answers
  matching.py     volunteer match model
  llm.py          model clients and PII masking
  models.py       database tables
  schemas.py      request and response shapes
pipelines/        data generation, ETL, backlog classification, feature building
evals/            classifier evaluation, RAG evaluation, drift check
prompts/          versioned prompt files
data/docs/        program documents used by the assistant
tests/            API and safety-rule tests (no real LLM calls)
k8s/              Kubernetes manifests
```

## Run it locally

You need Docker, Python 3.11 and two API keys: one for OpenAI and one for an OpenAI-compatible
provider that serves Llama 3.3 (this project uses Mesh API).

```bash
cp .env.example .env                 # then add your keys
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
docker compose up -d postgres
```

Load the data and train the match model. This is done once.

```bash
python -m pipelines.etl               # clean and load the three source files
python -m pipelines.classify_backlog  # give the imported requests a category
python -m pipelines.build_features    # build the training table for matching
python -m app.matching                # train the volunteer match model
python -m app.rag                     # embed the program documents
```

Start the API. There are two pages and the API documentation:

| Address | Who it is for |
|---|---|
| http://localhost:8000 | People who need help. They describe their problem and get a reference number. Classification, matching and assignment run in the background. |
| http://localhost:8000/volunteer | Volunteers. They see the cases assigned to them and accept, decline or finish them. |
| http://localhost:8000/docs | Developers and staff tools: review queue, audit trail, every endpoint. |

```bash
docker compose up -d --build api
```

Dashboards and experiment tracking are optional:

```bash
docker compose --profile tools up -d   # Metabase on :3000, MLflow on :5001
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /requests` | Classify a request and put it in a team queue |
| `GET /queues/{category}` | A team's waiting requests, most urgent first |
| `POST /intake` | Used by the public form: save the request, answer at once, run the agent in the background |
| `POST /triage` | Run the full agent on a request and wait for the result |
| `GET /review` | Requests waiting for a human decision, with suggested volunteers |
| `POST /review/{id}` | Approve, correct or reject |
| `GET /requests/{id}/audit` | Every step taken on a request |
| `GET /volunteers/{id}/cases` | A volunteer's new cases and cases in progress |
| `POST /cases/{id}/respond` | A volunteer accepts, declines or finishes a case |
| `POST /ask` | Ask the assistant a question. Streams by default, `?stream=false` returns JSON |
| `GET /health` | Health check |

```bash
curl -X POST localhost:8000/triage -H 'Content-Type: application/json' \
  -d '{"text": "We lost our income and need groceries this week.", "city": "Austin"}'

curl -N -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How often can I visit the food pantry?"}'
```

## Evaluation

The numbers below were measured on the generated data in this repository. Real requests are messier,
so expect lower scores on real data.

**Classifier**

| | GPT-4o | Llama 3.3 70B |
|---|---|---|
| Category accuracy (600 requests) | 99.0% | 99.3% |

Urgency is harder and more subjective. Prompt v2 was written to reduce the most dangerous mistake,
an urgent request marked as low (GPT-4o, 240 requests):

| | Prompt v1 | Prompt v2 |
|---|---|---|
| Urgency accuracy | 56.7% | 62.1% |
| Urgent requests marked low | 9.2% | 2.1% |

**RAG assistant** (34 questions: 22 the documents can answer, 12 they cannot)

| | Plain prompt | With guardrails |
|---|---|---|
| Unanswerable questions answered anyway | 100% | 0% |
| Answerable questions wrongly refused | 0% | 0% |
| Faithfulness (RAGAS) | 0.79 | 0.98 |

Context precision of the retrieval step is 0.97.

**Volunteer match model:** AUC 0.77 on held-out cases.

Run them yourself:

```bash
python -m evals.eval_classifier --limit 60
python -m evals.eval_rag
python -m evals.drift_check --simulate-drift
```

Each run is recorded in MLflow with its model, prompt version and scores.

## Tests

```bash
ruff check .
pytest -q
```

The tests replace the LLM with a fake, so they are free and repeatable. GitHub Actions runs them
against a real Postgres on every push and builds the Docker image.

## Kubernetes

The manifests in `k8s/` run two API replicas and Postgres. To try them on a local cluster with
[kind](https://kind.sigs.k8s.io):

```bash
kind create cluster --name aidroute
docker build -t aidroute:latest . && kind load docker-image aidroute:latest --name aidroute
kubectl create secret generic aidroute-secrets \
  --from-literal=POSTGRES_PASSWORD=aid \
  --from-literal=DATABASE_URL=postgresql+psycopg://aid:aid@postgres:5432/aidroute \
  --from-literal=OPENAI_API_KEY=... --from-literal=MESH_API_KEY=...
kubectl apply -f k8s/
kubectl port-forward svc/aidroute-api 8080:80     # then open http://localhost:8080/docs
```

## Deploy

The API runs on [Render](https://render.com) and the database on [Neon](https://neon.tech). Both have a
free tier.

1. Create a Neon project and copy its connection string.
2. Fill the database from your machine by running the five setup commands above with
   `DATABASE_URL` set to the Neon connection string.
3. In Render choose **New > Blueprint**, pick this repository, and enter `DATABASE_URL`,
   `OPENAI_API_KEY` and `MESH_API_KEY`. `render.yaml` describes the service.
4. For the weekly drift check, add `DATABASE_URL` as a repository secret in GitHub.

## Data

No real client data is in this repository. The requests, volunteers, cases and program documents are
generated or written for the project (`pipelines/generate_data.py`), so anyone can run it end to end.

## Known limits

- The volunteer portal has no login. A volunteer picks their name from a list. Real use needs sign-in
  before the portal is opened to the internet, because cases show contact details.
- Crisis detection is a keyword list. It catches clear phrases and misses indirect ones.
- PII masking covers phone numbers, emails and SSNs, not names or addresses.
- The model's own confidence score is weakly related to being right, which is why review rules do not
  depend on confidence alone.
- Tables are created at startup with `create_all`. A project with changing schemas would use migrations.
