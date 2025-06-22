# Use slim base image
FROM python:3.13-slim


# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential && \
    apt-get clean

# Install Poetry
ENV POETRY_VERSION=2.1.3      
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Disable virtualenv creation in Poetry (we want system-wide install)
ENV POETRY_VIRTUALENVS_CREATE=false

# Set working directory
WORKDIR /app

# Copy Poetry files first for caching
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Optional: Set environment variables
ENV GOOGLE_GENAI_USE_VERTEXAI=1
ENV PATH="/home/myuser/.local/bin:$PATH"

RUN adduser --disabled-password --gecos "" myuser && \
    chown -R myuser:myuser /app

COPY . .

USER myuser
EXPOSE 8080

ENV PATH="/home/myuser/.local/bin:$PATH"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8080"]