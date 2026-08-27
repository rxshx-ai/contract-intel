# Single-instance container for the contract intelligence service.
#
# IMPORTANT: this app keeps analysis state in memory (api/main.py `_state`),
# writes SQLite to the working directory, and caches extractions on local disk.
# It must therefore run as ONE instance. Do not autoscale it, and do not put it
# behind a load balancer with more than one target -- each replica would answer
# from a different set of contracts. Fixing that is a real piece of work
# (externalise state to Postgres/S3), not a deployment flag.

FROM python:3.12-slim AS base

# pdfplumber needs no system libs; OCR would (tesseract/poppler) and is not
# installed here, so scanned PDFs degrade to the text layer and report it.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY web/ ./web/
COPY eval/ ./eval/
COPY contracts/ ./contracts/
COPY demo_cache/ ./demo_cache/

# Install the shipped extractions so the container serves the full demo with no
# API key. A key is only needed to analyse NEW documents or to run the agent.
RUN python eval/seed_cache.py

# Fail the build rather than ship a broken image.
RUN python -c "import sys; sys.path.insert(0,'.'); \
from datetime import date; from api import demo; \
b = demo.load(date(2026,8,27)); \
assert len(b) == 4, b; \
print(f'image check: {len(b)} contracts, {sum(len(x.claims) for x in b)} claims')"

EXPOSE 8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
