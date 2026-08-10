#!/usr/bin/env bash
# Télécharge BGE-M3 depuis HuggingFace dans le répertoire models/
# Usage : bash scripts/download_model.sh [destination]
set -euo pipefail

DEST="${1:-./models/bge-m3}"
MODEL_ID="BAAI/bge-m3"

echo "[download_model] Destination : $DEST"
echo "[download_model] Modèle      : $MODEL_ID"

if [ -f "$DEST/config.json" ]; then
    echo "[download_model] Modèle déjà présent — rien à faire."
    exit 0
fi

mkdir -p "$DEST"

# Téléchargement via huggingface-hub (pip install huggingface-hub si absent)
python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$DEST",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print("[download_model] Téléchargement terminé.")
PYEOF
