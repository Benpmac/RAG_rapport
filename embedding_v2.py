"""
embedding_v2.py — Construction des index vectoriels ANSTAT-CI
=============================================================

RÔLE
  Lit tous les chunks JSON produits par CHUNK.py.py, encode leur texte
  avec BGE-M3 et construit les index FAISS + BM25 utilisés par system.py.py.

INPUTS
  ~/rag_3/chunks_output_v2/*_chunks.json  ← produits par CHUNK.py.py
  ~/rag_3/embeddings_v2/embeddings.npz   ← vecteurs existants (mode incrémental)

OUTPUTS
  ~/rag_3/embeddings_v2/faiss_index.bin   ← index vectoriel FAISS IVFFlat (dim=1024)
  ~/rag_3/embeddings_v2/bm25_index.pkl    ← index lexical BM25Okapi
  ~/rag_3/embeddings_v2/chunk_map.json    ← métadonnées de tous les chunks
  ~/rag_3/embeddings_v2/graph_index.json  ← infos géo/temporelles pour le graph bonus
  ~/rag_3/embeddings_v2/embeddings.npz    ← vecteurs bruts sauvegardés pour réutilisation
  ~/rag_3/embeddings_v2/metadata.json     ← stats globales (nb chunks, modèle, dates)

FONCTIONS
  tokenize_fr(text)               → tokens français normalisés (sans accents, casse)
  extraire_localites(text)        → localités détectées (Abidjan, Korhogo, Nord…)
  extraire_series(doc_name)       → séries statistiques (EHCVM, EDS, ENE, RGPH…)
  charger_tous_les_chunks()       → (chunk_map, graph_meta) depuis tous les JSON
  charger_modele()                → (model, model_type) BGE-M3 ou SentenceTransformer
  encoder(model, type, texts)     → matrice numpy (N, 1024) normalisée L2
  charger_embeddings_existants()  → (ids, vecteurs) depuis embeddings.npz
  encoder_nouveaux_chunks(...)    → encode uniquement les chunks absents du .npz
  construire_faiss(ids, vecs)     → index FAISS IVFFlat entraîné et prêt
  construire_bm25(ids, chunk_map) → index BM25Okapi sur le corpus tokenisé
  sauvegarder(...)                → écrit tous les fichiers dans embeddings_v2/
  build(incremental)              → pipeline complet — point d'entrée principal

USAGE
  python embedding_v2.py          # build incrémental (défaut)
  python embedding_v2.py --full   # réencode tout depuis zéro
"""

import argparse
import json
import pickle
import re
import time
import unicodedata
from pathlib import Path

import faiss
import numpy as np

from rank_bm25 import BM25Okapi

try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_AVAILABLE = True
except ImportError:
    FLAG_AVAILABLE = False
    from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNKS_DIR       = Path.home() / "rag_3" / "chunks_output_v2"
EMBEDDINGS_DIR   = Path.home() / "rag_3" / "embeddings_v2"
EMBEDDING_MODEL  = str(Path.home() / "models" / "bge-m3")
BATCH_SIZE       = 16   # réduit pour les machines sans GPU

FAISS_INDEX_FILE = EMBEDDINGS_DIR / "faiss_index.bin"
BM25_INDEX_FILE  = EMBEDDINGS_DIR / "bm25_index.pkl"
CHUNK_MAP_FILE   = EMBEDDINGS_DIR / "chunk_map.json"
GRAPH_INDEX_FILE = EMBEDDINGS_DIR / "graph_index.json"
EMBEDDINGS_FILE  = EMBEDDINGS_DIR / "embeddings.npz"
METADATA_FILE    = EMBEDDINGS_DIR / "metadata.json"

# Référentiels pour graph_meta
LOCALITES_MOTS = {
    "abidjan": ["abidjan", "cocody", "yopougon", "abobo", "adjame", "marcory",
                "treichville", "port-bouet", "koumassi", "attecoube"],
    "yamoussoukro": ["yamoussoukro"],
    "nord":    ["korhogo", "ferkessedougou", "boundiali", "odienne", "nord"],
    "ouest":   ["man", "daloa", "duekoe", "guiglo", "ouest"],
    "est":     ["abengourou", "bondoukou", "est"],
    "centre":  ["bouake", "centre", "dimbokro"],
    "sud":     ["san-pedro", "sassandra", "sud"],
    "rural":   ["rural", "campagne", "zone rurale"],
    "urbain":  ["urbain", "ville", "zone urbaine"],
    "national":["national", "cote d'ivoire", "ivoirien"],
}

SERIES_MOTS = {
    "EHCVM": ["ehcvm"], "EDS": ["eds"], "ENV":  ["env"],
    "ENE":   ["ene"],   "RGPH":["rgph"], "ERI": ["eri-esi"],
    "MICS":  ["mics"],  "IPS": ["ips"],  "ICA": ["ica"],
    "BTP":   ["btp"],
}


# ── Utilitaires ────────────────────────────────────────────────────────────────

def tokenize_fr(text: str) -> list:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    tokens = re.findall(r"\b[\w]+\b", text)
    return [t for t in tokens if len(t) > 1]


def extraire_localites(text: str) -> list:
    t = text.lower()
    return [loc for loc, mots in LOCALITES_MOTS.items()
            if any(m in t for m in mots)]


def extraire_series(doc_name: str) -> list:
    n = doc_name.lower()
    return [serie for serie, mots in SERIES_MOTS.items()
            if any(m in n for m in mots)]


# ── Chargement des chunks ──────────────────────────────────────────────────────

def charger_tous_les_chunks() -> tuple[dict, dict]:
    """
    Lit tous les *_chunks.json dans chunks_output_v2/.
    Retourne (chunk_map, graph_meta).
    """
    chunk_map  = {}
    graph_meta = {}
    n_fichiers = 0

    for chunks_file in sorted(CHUNKS_DIR.glob("*_chunks.json")):
        with open(chunks_file, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        doc_meta = doc_data.get("metadata", {})
        n_fichiers += 1

        for chunk in doc_data.get("chunks", []):
            cid  = chunk["chunk_id"]
            meta = chunk.get("metadata", {})
            text = chunk.get("text", "")

            chunk_map[cid] = {
                "content":        text,
                "enriched_text":  text,
                "context_desc":   meta.get("context_description", ""),
                "document_name":  meta.get("document_name",  doc_meta.get("document_name",  "")),
                "document_title": meta.get("document_title", doc_meta.get("document_title", "")),
                "source_url":     meta.get("source_url",     doc_meta.get("document_url",   "")),
                "page_number":    meta.get("page_number", -1),
                "themes":         meta.get("themes", []),
                "annee":          meta.get("annee", doc_meta.get("annee")),
                "raptor_level":   meta.get("raptor_level", "leaf"),
                "is_table":       meta.get("is_table", False),
            }

            graph_meta[cid] = {
                "annee":        str(meta.get("annee", "")) if meta.get("annee") else "",
                "localites":    extraire_localites(text),
                "themes":       meta.get("themes", []),
                "series":       extraire_series(doc_meta.get("document_name", "")),
                "raptor_level": meta.get("raptor_level", "leaf"),
                "is_table":     meta.get("is_table", False),
            }

    print(f"   {n_fichiers} fichiers JSON lus → {len(chunk_map)} chunks")
    return chunk_map, graph_meta


# ── Encodage ───────────────────────────────────────────────────────────────────

def charger_modele():
    print(f"   Modèle : {EMBEDDING_MODEL}")
    if FLAG_AVAILABLE:
        model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=False)
        model_type = "bge-m3"
    else:
        model = SentenceTransformer(EMBEDDING_MODEL)
        model_type = "st"
    return model, model_type


def encoder(model, model_type: str, texts: list) -> np.ndarray:
    """Encode une liste de textes → matrice (N, 1024) normalisée L2."""
    if model_type == "bge-m3":
        out  = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        embs = out["dense_vecs"].astype(np.float32)
    else:
        embs = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-9)


def charger_embeddings_existants() -> tuple[list, np.ndarray]:
    """Charge embeddings.npz si disponible."""
    if not EMBEDDINGS_FILE.exists():
        return [], np.empty((0, 1024), dtype=np.float32)
    data    = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    ids     = list(data["chunk_ids"])
    vectors = data["embeddings"].astype(np.float32)
    print(f"   {len(ids)} embeddings existants chargés")
    return ids, vectors


def encoder_nouveaux_chunks(
    model, model_type: str,
    chunk_map: dict,
    existing_ids: list,
) -> tuple[list, np.ndarray]:
    """Encode uniquement les chunks absents de embeddings.npz."""
    existing_set = set(existing_ids)
    a_encoder = [
        (cid, chunk_map[cid]["enriched_text"])
        for cid in chunk_map
        if cid not in existing_set
    ]

    if not a_encoder:
        print("   Aucun nouveau chunk à encoder")
        return [], np.empty((0, 1024), dtype=np.float32)

    print(f"   Encodage de {len(a_encoder)} chunks...")
    ids   = [cid  for cid, _    in a_encoder]
    texts = [text for _,   text in a_encoder]

    t0   = time.time()
    vecs = encoder(model, model_type, texts)
    print(f"   Encodés en {time.time() - t0:.1f}s")
    return ids, vecs


# ── Construction des index ─────────────────────────────────────────────────────

def construire_faiss(chunk_ids: list, all_vecs: np.ndarray) -> faiss.Index:
    dim   = all_vecs.shape[1]
    n     = len(chunk_ids)
    nlist = min(512, max(4, n // 10))

    print(f"   FAISS IVFFlat : {n} vecteurs, {nlist} clusters, dim={dim}")
    quantizer = faiss.IndexFlatL2(dim)
    index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(all_vecs)
    index.add(all_vecs)
    index.nprobe = min(32, nlist)
    return index


def construire_bm25(chunk_ids: list, chunk_map: dict) -> BM25Okapi:
    print(f"   BM25 : {len(chunk_ids)} documents")
    corpus = [tokenize_fr(chunk_map[cid]["enriched_text"]) for cid in chunk_ids]
    return BM25Okapi(corpus)


# ── Sauvegarde ─────────────────────────────────────────────────────────────────

def sauvegarder(
    chunk_ids: list,
    all_vecs:  np.ndarray,
    chunk_map: dict,
    graph_meta: dict,
    faiss_index,
    bm25_index,
):
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        EMBEDDINGS_FILE,
        chunk_ids=np.array(chunk_ids, dtype=object),
        embeddings=all_vecs,
    )

    faiss.write_index(faiss_index, str(FAISS_INDEX_FILE))

    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25_index, f)

    with open(CHUNK_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False)

    graph_index = {"doc_relations": {}, "chunk_meta": graph_meta}
    with open(GRAPH_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(graph_index, f, ensure_ascii=False)

    tables_count = sum(1 for c in chunk_map.values() if c.get("is_table"))
    with_context = sum(1 for c in chunk_map.values() if c.get("context_desc"))
    metadata = {
        "total_chunks":    len(chunk_ids),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_stats": {
            "tables_count": tables_count,
            "with_context": with_context,
        },
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n   Sauvegardé dans {EMBEDDINGS_DIR}")
    print(f"   {len(chunk_ids)} chunks | {tables_count} tableaux | {with_context} avec contexte")


# ── Pipeline principal ─────────────────────────────────────────────────────────

def build(incremental: bool = True):
    t_start = time.time()

    print("\n" + "="*55)
    print("embedding_v2.py — Build index FAISS + BM25")
    print("="*55)

    # 1. Chargement des chunks
    print("\n[1/5] Chargement des chunks...")
    chunk_map, graph_meta = charger_tous_les_chunks()
    if not chunk_map:
        print("Aucun chunk trouvé dans", CHUNKS_DIR)
        return

    # 2. Chargement embeddings existants
    print("\n[2/5] Chargement embeddings existants...")
    existing_ids, existing_vecs = charger_embeddings_existants() if incremental else ([], np.empty((0, 1024), dtype=np.float32))

    # 3. Suppression des vecteurs orphelins (chunks supprimés)
    valid_existing = [(i, cid) for i, cid in enumerate(existing_ids) if cid in chunk_map]
    if len(valid_existing) < len(existing_ids):
        n_orphans = len(existing_ids) - len(valid_existing)
        print(f"   {n_orphans} vecteurs orphelins supprimés")
        keep_idx     = [i for i, _ in valid_existing]
        existing_ids = [cid for _, cid in valid_existing]
        existing_vecs = existing_vecs[keep_idx] if keep_idx else np.empty((0, 1024), dtype=np.float32)

    # 4. Encodage des nouveaux chunks
    print("\n[3/5] Chargement du modèle BGE-M3...")
    model, model_type = charger_modele()

    print("\n[4/5] Encodage des nouveaux chunks...")
    new_ids, new_vecs = encoder_nouveaux_chunks(model, model_type, chunk_map, existing_ids)

    # 5. Fusion
    all_ids  = existing_ids + new_ids
    all_vecs = (
        np.vstack([existing_vecs, new_vecs])
        if len(new_vecs) > 0
        else existing_vecs
    )

    # Réordonner selon chunk_map pour cohérence
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    ordered_ids  = list(chunk_map.keys())
    ordered_vecs = np.stack([all_vecs[id_to_idx[cid]] for cid in ordered_ids])

    # 5. Rebuild FAISS + BM25
    print("\n[5/5] Construction des index...")
    faiss_index = construire_faiss(ordered_ids, ordered_vecs)
    bm25_index  = construire_bm25(ordered_ids, chunk_map)

    sauvegarder(ordered_ids, ordered_vecs, chunk_map, graph_meta,
                faiss_index, bm25_index)

    print(f"\n✅ Terminé en {time.time() - t_start:.1f}s")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental", action="store_true", default=True,
                        help="Encode seulement les nouveaux chunks (défaut)")
    parser.add_argument("--full", action="store_true",
                        help="Réencode tout depuis zéro")
    args = parser.parse_args()

    build(incremental=not args.full)
