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

# Install uv and use it to install Python dependencies system-wide
RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache .

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

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
