FROM python:3.10-slim

ENV PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /app

COPY mcp_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_server/ /app/mcp_server/
COPY .env /app/.env

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
