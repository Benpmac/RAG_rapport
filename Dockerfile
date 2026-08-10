FROM apache/airflow:2.9.2-python3.12

USER root

# Outils système nécessaires pour pdfplumber (poppler) et compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copier requirements avant le code pour profiter du cache Docker
COPY requirements.txt /tmp/requirements.txt

# Installer torch CPU depuis l'index officiel, puis le reste
RUN pip install --no-cache-dir \
    torch==2.13.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    faiss-cpu==1.15.0 \
    FlagEmbedding==1.4.0 \
    rank-bm25==0.2.2 \
    pdfplumber==0.11.10 \
    numpy==2.5.1
