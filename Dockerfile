FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY agentwatch /app/agentwatch

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000
CMD ["python", "-m", "agentwatch.api.main"]

