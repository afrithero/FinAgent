FROM dustynv/l4t-pytorch:r36.4.0

ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126/

RUN apt-get update && \
    apt-get install -y \
            wget \
            python3-dev

WORKDIR /app

COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ /app/agent/
COPY .env /app/.env
COPY data/ /app/data/

# CMD ["python3", "/app/agent/run_react_agent.py"]