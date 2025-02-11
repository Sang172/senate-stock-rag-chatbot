FROM python:3.12-slim

WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --only-binary=:all: lxml

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scrape.py process.py ./

CMD ["python", "scrape.py"]