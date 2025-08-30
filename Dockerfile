FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

RUN python -c "import nltk; nltk.download('punkt_tab')"

COPY . .

EXPOSE 5000

CMD ["python", "src/prod.py"]