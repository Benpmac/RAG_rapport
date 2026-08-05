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

## Usage

### Indexer de nouveaux PDFs

Dépose tes PDFs dans le dossier `documents/` :
```bash
cp mon_rapport.pdf ~/rag_3/documents/
```

Le DAG Airflow détecte automatiquement le nouveau fichier toutes les **30 minutes** et lance le pipeline complet.


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


