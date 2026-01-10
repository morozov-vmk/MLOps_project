FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ build-essential && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    apt-get remove -y gcc g++ build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY . /app

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.predict"]
