FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GHTRIAGE_REGISTRY_DIR=/app/models

COPY pyproject.toml ./
COPY src ./src
# libgomp1: OpenMP runtime the LightGBM wheel links against (not in the slim image).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ".[serve]"

# Run `make train` first so models/ exists, or mount it as a volume instead of copying.
COPY config ./config
COPY models ./models

# Hugging Face Spaces expects 7860; locally the default is 8000. Override with PORT.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

CMD ["sh", "-c", "uvicorn ghtriage.serve:app --host 0.0.0.0 --port ${PORT}"]
