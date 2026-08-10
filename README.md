# RAG_3 

Système de recherche et de question-réponse intelligent sur les rapports statistiques officiels de Côte d'Ivoire (ANSTAT-CI). Pose une question en français, le système cherche dans des dizaines de rapports PDF et répond avec les données précises et les sources.

---

## Architecture générale

```
┌─────────────────────────────────────────────────────────────┐
│                     PHASE 1 — PRÉPARATION                   │
│                   (automatisée par Airflow)                  │
│                                                             │
│  PDF déposé dans rag3_data/documents/                       │
│       │                                                     │
│       ▼                                                     │
│  chunk_pipeline.py — Chunking RAPTOR 4 niveaux              │
│  (leaf → section → chapter → document)                      │
│       │                                                     │
│       ▼                                                     │
│  embedding_v2.py — Encodage BGE-M3 + Index FAISS + BM25    │
│       │                                                     │
│       ▼                                                     │
│  rag3_data/embeddings_v2/ (faiss_index.bin, bm25_index.pkl) │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     PHASE 2 — REQUÊTE                       │
│                    (rag3_data/system.py)                    │
│                                                             │
│  Question utilisateur                                       │
│       │                                                     │
│       ├─► Cache 81K Q/R (sim > 0.88) ──► Réponse rapide    │
│       ├─► Cache conversationnel       ──► Réponse rapide    │
│       └─► RAG complet :                                     │
│           FAISS(50) + BM25(50)                              │
│           → RRF → Graph bonus                               │
│           → BGE Reranker → top 5 chunks                     │
│           → Qwen 2.5-7B (vLLM) ──────► Réponse sourcée     │
└─────────────────────────────────────────────────────────────┘
```

---

## Démarrage rapide (Docker)

```bash
git clone <repo>
bash scripts/download_model.sh
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'
bash scripts/start.sh
```

Airflow disponible sur http://localhost:8080 (admin / admin).

---

## Structure du projet

```
rag_3/
├── dags/
│   └── rag3_ingestion.py       # DAG Airflow — orchestration incrémentale SHA-256
├── rag3_data/                  # Données et scripts (monté dans Docker via RAG3_DIR)
│   ├── chunk_pipeline.py       # Pipeline de chunking RAPTOR (extraction + RAPTOR)
│   ├── embedding_v2.py         # Construction des index FAISS + BM25 avec BGE-M3
│   ├── system.py               # Moteur de requête RAG (cache + hybrid search + LLM)
│   ├── documents/              # PDFs sources à déposer ici (ignoré par git)
│   ├── chunks_output_v2/       # Chunks JSON générés automatiquement (ignoré par git)
│   └── embeddings_v2/          # Index FAISS, BM25 et vecteurs (ignoré par git)
├── models/
│   └── bge-m3/                 # Modèle BGE-M3 téléchargé par download_model.sh (ignoré par git)
├── scripts/
│   ├── download_model.sh       # Télécharge BGE-M3 depuis HuggingFace
│   └── start.sh                # Build Docker + lancement
├── documents_demo/             # PDFs exemples pour tester l'installation
├── docs/captures/              # Captures d'écran pour la documentation
├── Dockerfile                  # Étend apache/airflow:2.9.2 avec poppler + dépendances ML
├── docker-compose.override.yaml
├── requirements.txt            # Dépendances Python avec versions fixées
├── .env.example                # Variables d'environnement à copier en .env
└── .gitignore
```

---

## Prérequis

| Composant | Version | Usage |
|---|---|---|
| Docker + Compose | 24.0+ | Conteneurisation |
| Python | 3.12 | Langage principal |
| Apache Airflow | 2.9.2 | Orchestration du pipeline |
| BGE-M3 | BAAI/bge-m3 | Modèle d'embedding (1024 dim) |
| BGE Reranker | BAAI/bge-reranker-v2-m3 | Reranking cross-encoder |
| vLLM | — | Serveur LLM (Qwen 2.5-7B-Instruct-AWQ, optionnel) |
| FAISS | faiss-cpu 1.15.0 | Index vectoriel |
| BM25 | rank-bm25 0.2.2 | Index lexical |

---

## Configuration

Copiez `.env.example` en `.env` et adaptez :

```bash
cp .env.example .env
```

| Variable | Défaut | Description |
|---|---|---|
| `RAG3_DIR` | `~/rag_3/rag3_data` | Répertoire des données |
| `EMBEDDING_MODEL` | `~/models/bge-m3` | Chemin vers BGE-M3 |
| `BATCH_SIZE` | `16` | Taille de batch pour l'encodage |

Les secrets Airflow (URL vLLM, emails de notification) se configurent via **Admin > Variables** dans l'UI Airflow :

| Clé | Format | Description |
|---|---|---|
| `rag3_vllm_url` | URL | URL du serveur vLLM (laisser vide pour désactiver l'enrichissement LLM) |
| `rag3_email_dest` | JSON `["a@b.com"]` | Destinataires des notifications email |

---

## Usage

### Indexer de nouveaux PDFs

Dépose tes PDFs dans `rag3_data/documents/` :
```bash
cp mon_rapport.pdf rag3_data/documents/
```

Le DAG Airflow détecte automatiquement le nouveau fichier toutes les **30 minutes** et lance le pipeline complet.

---

## Pipeline Airflow détaillé

```
detecter_changements        ← SHA-256 de chaque PDF vs manifest
        │
y_a_t_il_du_travail         ← court-circuit si aucun changement
        │
supprimer_anciens_chunks    ← supprime les chunks des PDFs modifiés/supprimés
        │
chunker_pdfs                ← RaptorRAGPipeline.process_document()
        │
construire_index            ← embedding_v2.build() + maj manifest
```

**Notifications email** : un email est envoyé à la fin du chunking et de l'indexation, ainsi qu'en cas d'échec.

---

## Fichiers produits après indexation

```
rag3_data/embeddings_v2/
├── faiss_index.bin      # Index vectoriel FAISS (FlatIP < 10k chunks, IVFFlat sinon)
├── bm25_index.pkl       # Index lexical BM25Okapi
├── chunk_map.json       # Métadonnées de tous les chunks
├── graph_index.json     # Infos géographiques/temporelles (graph bonus)
├── embeddings.npz       # Vecteurs bruts (réutilisés en mode incrémental)
└── metadata.json        # Stats globales
```

---

## Modèles utilisés

| Modèle | Tâche | Dim |
|---|---|---|
| BAAI/bge-m3 | Embedding des chunks et des questions | 1024 |
| BAAI/bge-reranker-v2-m3 | Reranking cross-encoder (top 5) | — |
