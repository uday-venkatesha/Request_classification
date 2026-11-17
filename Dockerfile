FROM python:3.11-slim

WORKDIR /srv

# Dependencies are installed before the code is copied, so Docker reuses this
# layer and rebuilds stay fast when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY prompts prompts
COPY data/docs data/docs

# Don't run the service as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# Hosting platforms such as Render pass the port in $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
