FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.7 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked

COPY . .

EXPOSE 8000
CMD ["uv","run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]