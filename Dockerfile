FROM python:3.11-slim-bookworm AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt \
    && find /usr/local/lib/python3.11/site-packages/ -name '__pycache__' -type d -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages/ -name '*.pyc' -delete \
    && find /usr/local/lib/python3.11/site-packages/ -name '*.pyo' -delete \
    && find /usr/local/lib/python3.11/site-packages/ -name '*.dist-info' -type d -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages/ -name '*.egg-info' -type d -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages/ -name 'tests' -type d -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages/ -name 'test' -type d -exec rm -rf {} + \
    && apt-get purge -y --auto-remove build-essential

RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/local/nltk_data')"


FROM python:3.11-slim AS production
WORKDIR /app
COPY --from=base /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=base /usr/local/bin/ /usr/local/bin/
COPY --from=base /usr/local/nltk_data/ /usr/local/nltk_data/

COPY src/ ./src/
COPY model/model_wrapper.tar.gz ./model/model_wrapper.tar.gz

EXPOSE 5000

CMD ["python", "src/prod.py"]