#!/usr/bin/env bash
# Démarre le stack RAG_3 ANSTAT-CI avec Docker Compose
# Usage : bash scripts/start.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Vérifications préalables
if [ ! -f ".env" ]; then
    echo "[start] .env absent — copie de .env.example"
    cp .env.example .env
    echo "[start] Éditez .env avant de relancer (AIRFLOW__CORE__FERNET_KEY manquante)."
    exit 1
fi

if [ ! -f "models/bge-m3/config.json" ]; then
    echo "[start] Modèle BGE-M3 absent. Lancez d'abord :"
    echo "        bash scripts/download_model.sh"
    exit 1
fi

if [ ! -f "docker-compose.yaml" ]; then
    echo "[start] docker-compose.yaml officiel Airflow absent. Téléchargez-le :"
    echo "        curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'"
    exit 1
fi

echo "[start] Construction de l'image..."
docker compose -f docker-compose.yaml -f docker-compose.override.yaml build

echo "[start] Initialisation de la base Airflow..."
docker compose -f docker-compose.yaml -f docker-compose.override.yaml run --rm airflow-init

echo "[start] Démarrage des services..."
docker compose -f docker-compose.yaml -f docker-compose.override.yaml up -d

echo "[start] Airflow disponible sur http://localhost:8080 (admin/admin)"
