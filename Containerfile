FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./pyproject.toml ./
COPY ./LICENSE ./

COPY ./src/ ./src/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

RUN mkdir -p /.llama/checkpoints/
RUN chmod -R 777 /.llama/
CMD ["python3"]
