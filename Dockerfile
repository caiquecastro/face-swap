FROM python:3.12-slim

WORKDIR /app

# System libraries required by opencv and insightface
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the locked production dependencies into a virtual environment
RUN pip install --no-cache-dir uv==0.10.8
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache
ENV PATH="/app/.venv/bin:$PATH"

COPY app.py .
COPY templates/ templates/
COPY static/ static/

RUN mkdir -p models static/generated

# Optionally download the model at build time.
# Usage: docker build --build-arg MODEL_URL=https://... .
ARG MODEL_URL
RUN if [ -n "$MODEL_URL" ]; then \
        echo "Downloading model from $MODEL_URL" && \
        curl -fL "$MODEL_URL" -o models/inswapper_128.onnx; \
    fi

# Keep application code read-only for the runtime user; grant access to data directories.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app models static/generated

EXPOSE 8000

USER app:app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
