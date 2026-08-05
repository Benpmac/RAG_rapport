# RAG_3 — Système de Question-Réponse ANSTAT-CI

Système de recherche et de question-réponse intelligent sur les rapports statistiques officiels de Côte d'Ivoire (ANSTAT-CI). Pose une question en français, le système cherche dans des dizaines de rapports PDF et répond avec les données précises et les sources.

---

## Architecture générale

```
┌─────────────────────────────────────────────────────────────┐
│                     PHASE 1 — PRÉPARATION                   │
│                   (automatisée par Airflow)                  │
│                                                             │
│  PDF déposé dans documents/                                 │
│       │                                                     │
│       ▼                                                     │
│  CHUNK.py.py — Chunking RAPTOR 4 niveaux                    │
│  (leaf → section → chapter → document)                      │
│       │                                                     │
│       ▼                                                     │
│  embedding_v2.py — Encodage BGE-M3 + Index FAISS + BM25    │
│       │                                                     │
│       ▼                                                     │
│  embeddings_v2/ (faiss_index.bin, bm25_index.pkl, ...)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     PHASE 2 — REQUÊTE                       │
│                    (system.py.py)                           │
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

## Structure du projet

```
rag_3/
├── CHUNK.py.py          # Pipeline de chunking RAPTOR (extraction + filtrage + RAPTOR)
├── embedding_v2.py      # Construction des index FAISS + BM25 avec BGE-M3
├── rag3_ingestion.py    # DAG Airflow — orchestration incrémentale SHA-256
├── system.py.py         # Moteur de requête RAG (cache 81K + hybrid search + LLM)
├── .gitignore
├── README.md
├── documents/           # PDFs sources à déposer ici
├── chunks_output_v2/    # Chunks JSON générés automatiquement
└── embeddings_v2/       # Index FAISS, BM25 et vecteurs générés automatiquement
```

---

## Prérequis

| Composant | Version | Usage |
|---|---|---|
| Ubuntu (WSL2) | 22.04+ | Environnement d'exécution |
| Python | 3.10+ | Langage principal |
| Apache Airflow | 2.8+ | Orchestration du pipeline |
| BGE-M3 | BAAI/bge-m3 | Modèle d'embedding (1024 dim) |
| BGE Reranker | BAAI/bge-reranker-v2-m3 | Reranking cross-encoder |
| vLLM | — | Serveur LLM (Qwen 2.5-7B-Instruct-AWQ) |
| FAISS | faiss-cpu | Index vectoriel |
| BM25 | rank-bm25 | Index lexical |

---

## Installation

### 1. Cloner le repo
```bash
git clone https://github.com/benpmac/rag_3.git
cd rag_3
```

### 2. Créer et activer l'environnement virtuel
```bash
python3 -m venv ~/airflow-env
source ~/airflow-env/bin/activate
```

### 3. Installer les dépendances
```bash
pip install apache-airflow faiss-cpu rank-bm25 FlagEmbedding \
            pymupdf pdfplumber opendataloader-pdf requests numpy
```

### 4. Télécharger les modèles
```bash
# BGE-M3 (~2.3 Go)
hf download BAAI/bge-m3 --local-dir ~/models/bge-m3

# BGE Reranker (~2.3 Go)
hf download BAAI/bge-reranker-v2-m3 --local-dir ~/models/bge-reranker-v2-m3
```

### 5. Configurer Airflow
```bash
export AIRFLOW_HOME=~/airflow
airflow db init

# Configurer le SMTP dans ~/airflow/airflow.cfg (section [smtp])
# pour recevoir les notifications par email

# Créer le lien symbolique pour le DAG
ln -s ~/rag_3/rag3_ingestion.py ~/airflow/dags/rag3_ingestion.py
```

### 6. Démarrer Airflow
```bash
source ~/airflow-env/bin/activate
airflow scheduler &
airflow webserver --port 8080 &
```

---

## Usage

### Indexer de nouveaux PDFs

Dépose tes PDFs dans le dossier `documents/` :
```bash
cp mon_rapport.pdf ~/rag_3/documents/
```

Le DAG Airflow détecte automatiquement le nouveau fichier toutes les **30 minutes** et lance le pipeline complet.

Pour déclencher manuellement :
```bash
airflow dags trigger rag3_anstat_ingestion
```

Ou via l'interface web : `http://localhost:8080`

### Interroger le système
```bash
source ~/airflow-env/bin/activate
cd ~/rag_3
python system.py.py
```

Exemples de questions :
```
> Quel est le taux de pauvreté en Côte d'Ivoire en 2021 ?
> Compare les conditions de vie entre milieu rural et urbain.
> Quels sont les indicateurs de santé de l'EDS 2021 ?
```

---

## Pipeline Airflow détaillé

```
detecter_changements        ← SHA-256 de chaque PDF vs manifest
        │
y_a_t_il_du_travail         ← court-circuit si aucun changement
        │
   ┌────┴────┐
   │         │
supprimer  chunker_pdfs     ← RaptorRAGPipeline.process_document()
   │         │
   └────┬────┘
        │
construire_index            ← embedding_v2.build() + maj manifest
```

**Notifications email** : un email est envoyé à la fin du chunking et de l'indexation, ainsi qu'en cas d'échec.

---

## Fichiers produits après indexation

```
embeddings_v2/
├── faiss_index.bin      # Index vectoriel FAISS IVFFlat (dim=1024)
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
| Qwen/Qwen2.5-7B-Instruct-AWQ | Génération de la réponse finale | — |

---

## Documents indexés

Rapports statistiques officiels de l'ANSTAT-CI :
- EHCVM 2021 — Conditions de vie des ménages
- EDS 2021 — Enquête Démographique et de Santé
- ENE 2019 — Enquête Nationale sur l'Emploi
- IPS Éducation 2022 / IPS Santé 2021
- RGPH — Recensement Général de la Population
- Et d'autres rapports ANSTAT-CI...

---

## Contact

Projet développé dans le cadre de l'ANSTAT-CI.
