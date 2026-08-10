# ============================================================================
# system.py.py — MOTEUR DE REQUÊTE RAG ANSTAT-CI (V3)
# ============================================================================
#
# RÔLE
#   Point d'entrée pour interroger la base vectorielle construite par
#   embedding_v2.py. Reçoit une question en français et retourne une
#   réponse sourcée issue des rapports statistiques ANSTAT-CI.
#
# INPUTS
#   ~/rag_3/embeddings_v2/faiss_index.bin   ← index vectoriel
#   ~/rag_3/embeddings_v2/bm25_index.pkl    ← index BM25
#   ~/rag_3/embeddings_v2/chunk_map.json    ← métadonnées chunks
#   ~/rag_3/embeddings_v2/graph_index.json  ← graph bonus
#   ~/rag_3/embeddings_v2/cache_semantique/ ← cache 81 000 Q/R pré-indexées
#   ~/models/bge-m3/                        ← modèle d'embedding
#   ~/models/bge-reranker-v2-m3/            ← modèle de reranking
#   http://192.168.1.230:30179              ← vLLM (Qwen 2.5-7B) pour la génération
#
# OUTPUT
#   Réponse textuelle en français avec sources citées
#
# CLASSES
#   SemanticCache        cache conversationnel en mémoire (500 Q récentes)
#     .lookup(q_emb)     → résultat mis en cache si sim > 0.97
#     .store(q_emb, ...)  → enregistre une nouvelle Q/R
#     .stats()           → statistiques du cache
#   Cache81K             cache sémantique FAISS sur 81 000 Q/R pré-indexées
#     .load()            → charge cache_faiss.bin + cache_map.json
#     .lookup(q_emb)     → résultat instantané si sim > 0.88
#     .stats()           → statistiques du cache
#   ANSTATRagSystem      classe principale — point d'entrée utilisateur
#     .load()            → charge tous les index et modèles en mémoire
#     .ask(question)     → pipeline complet → réponse + sources
#     .clear_history()   → vide l'historique conversationnel
#     .clear_cache()     → vide le cache conversationnel
#     .stats()           → affiche les stats du système
#
# FONCTIONS
#   norm(text)                  → normalisation unicode
#   tokenize_fr(text)           → tokenisation française
#   extraire_contexte(question) → détecte année, localité, thème dans la question
#   load_all_indexes()          → charge FAISS, BM25, chunk_map, graph_index
#   load_models()               → charge BGE-M3 + BGE Reranker
#   encode_query(question, m)   → vecteur (1024,) de la question
#   search_faiss(q_emb, ...)    → top-K chunks par similarité vectorielle
#   search_bm25(question, ...)  → top-K chunks par score BM25
#   rrf_fusion(faiss, bm25)     → fusion par Reciprocal Rank Fusion
#   graph_bonus(cid, ctx, ...)  → score bonus géo/temporel
#   rerank_candidates(...)      → reclassement fin par BGE Reranker (cross-encoder)
#   hybrid_search(question, ...) → pipeline complet FAISS+BM25→RRF→graph→rerank
#   build_prompt(question, ...) → construit le prompt LLM avec contexte
#   call_vllm(messages, ...)    → appel HTTP vLLM → texte généré
#   check_vllm()                → vérifie la disponibilité du serveur vLLM
#   format_sources(chunks)      → formate les sources pour l'affichage
#   run_terminal(rag)           → boucle interactive en mode terminal
#   main()                      → point d'entrée CLI
#
# PIPELINE DE RÉPONSE
#   Question
#     → Cache 81K (sim > 0.88)     → réponse instantanée
#     → Cache conversationnel      → réponse rapide
#     → FAISS(50) + BM25(50)
#       → RRF → Graph bonus
#       → BGE Reranker → top 5
#       → Qwen 2.5-7B (vLLM)      → réponse sourcée
#
# USAGE
#   python system.py.py            # mode terminal interactif
# ============================================================================

import json, pickle, re, time, hashlib, unicodedata
from pathlib  import Path
from typing   import List, Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import faiss

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    FLAG_AVAILABLE = True
except ImportError:
    FLAG_AVAILABLE = False
    from sentence_transformers import SentenceTransformer, CrossEncoder

import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDINGS_DIR   = Path.home() / "rag_3" / "embeddings_v2"
FAISS_INDEX_FILE = EMBEDDINGS_DIR / "faiss_index.bin"
BM25_INDEX_FILE  = EMBEDDINGS_DIR / "bm25_index.pkl"
CHUNK_MAP_FILE   = EMBEDDINGS_DIR / "chunk_map.json"
GRAPH_INDEX_FILE = EMBEDDINGS_DIR / "graph_index.json"
METADATA_FILE    = EMBEDDINGS_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = str(Path.home() / "models" / "bge-m3")
RERANKER_MODEL_NAME  = str(Path.home() / "models" / "bge-reranker-v2-m3")

VLLM_URL   = "http://192.168.1.230:30179/v1/chat/completions"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Paramètres de recherche
TOP_K_FAISS   = 50
TOP_K_BM25    = 50
TOP_K_RRF     = 25
TOP_K_RERANK  = 5  
RRF_K         = 60
RRF_W_FAISS   = 0.6
RRF_W_BM25    = 0.4

# Cache conversationnel (questions récentes)
CACHE_ENABLED   = True
CACHE_THRESHOLD = 0.97   # ← CORRIGÉ (était 0.92) : seuil plus strict pour éviter
                          #   les faux hits entre questions similaires (ex: "rural"
                          #   vs "rural et urbain")
CACHE_MAX_SIZE  = 500

# ── Cache sémantique 81 000 Q/R ──────────────────────────────────────────────
CACHE_81K_DIR        = Path.home() / "rag_3" / "embeddings_v2" / "cache_semantique"
CACHE_81K_FAISS_FILE = CACHE_81K_DIR / "cache_faiss.bin"
CACHE_81K_MAP_FILE   = CACHE_81K_DIR / "cache_map.json"
CACHE_81K_THRESHOLD  = 0.88   # seuil similarité pour considérer une Q connue

# Historique conversationnel
MAX_HISTORY = 6

# ============================================================================
# RÉFÉRENTIELS (identiques à embedding_v2.py)
# ============================================================================

ANNEES_CONNUES = [
    1985, 1986, 1993, 1994, 1995, 1998, 1999, 2002,
    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
    2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025,
]

LOCALITES_CI = {
    "abidjan":      ["abidjan", "plateau", "cocody", "yopougon", "abobo",
                     "adjame", "marcory", "treichville", "port-bouet",
                     "koumassi", "attecoube", "anyama"],
    "yamoussoukro": ["yamoussoukro"],
    "nord":         ["korhogo", "ferkessedougou", "boundiali", "odienne",
                     "tengrela", "nord", "hambol", "poro", "tchologo"],
    "ouest":        ["man", "daloa", "duekoe", "guiglo", "ouest",
                     "tonkpi", "haut-sassandra", "cavally", "guemon"],
    "est":          ["abengourou", "bondoukou", "est",
                     "indenie-djuablin", "gontougo", "bounkani"],
    "centre":       ["bouake", "centre", "dimbokro", "gbeke", "belier"],
    "sud":          ["san-pedro", "sassandra", "grand-lahou", "dabou",
                     "sud", "bas-sassandra", "grands-ponts"],
    "rural":        ["rural", "campagne", "village", "zone rurale"],
    "urbain":       ["urbain", "ville", "metropole", "zone urbaine"],
    "national":     ["cote d'ivoire", "national", "pays", "ivoirien"],
}

THEMES_GRAPH = {
    "pauvrete":            ["pauvrete", "pauvre", "seuil de pauvrete", "incidence",
                            "quintile", "inegalite", "gini"],
    "emploi":              ["emploi", "chomage", "travail", "actif", "salaire",
                            "sous-emploi", "main-d'oeuvre"],
    "sante":               ["sante", "hopital", "medecin", "vaccination",
                            "paludisme", "vih", "sida", "maladie", "mortalite"],
    "education":           ["education", "scolarisation", "ecole", "eleve",
                            "alphabetisation", "baccalaureat"],
    "demographie":         ["population", "recensement", "natalite", "fecondite",
                            "migration", "rgph"],
    "economie":            ["pib", "croissance", "inflation", "prix",
                            "budget", "investissement", "exportation"],
    "conditions_vie":      ["conditions de vie", "menage", "depense",
                            "consommation", "niveau de vie", "ehcvm"],
    "tic":                 ["tic", "internet", "telephone", "numerique",
                            "smartphone", "connectivite"],
    "genre":               ["femme", "genre", "egalite", "discrimination",
                            "autonomisation"],
    "alimentation":        ["alimentation", "securite alimentaire", "nutrition",
                            "malnutrition", "famine"],
    "habitat":             ["logement", "eau potable", "electricite",
                            "assainissement", "habitat"],
    "chiffre_affaires":    ["chiffre d'affaires", "ica", "indice ca", "ventes"],
    "production_industrie":["production industrielle", "activite industrielle",
                            "indice de production"],
    "btp":                 ["btp", "batiment", "construction", "permis de construire"],
    "prix_production":     ["prix de production", "ipp", "prix producteur"],
}

SERIES_DOCS = {
    "EHCVM": ["ehcvm", "conditions de vie"],
    "EDS":   ["eds", "demographique et de sante"],
    "ENV":   ["env", "niveau de vie des menages"],
    "ENE":   ["ene", "enquete nationale emploi"],
    "EMSI":  ["emsi", "mesure societe information"],
    "RGPH":  ["rgph", "recensement population"],
    "ERI":   ["eri-esi", "emploi secteur informel"],
    "MICS":  ["mics", "indicateurs multiples"],
    "IPS":   ["ips", "indicateurs prestations services"],
    "ICA":   ["ica", "indice chiffre affaires"],
    "BTP":   ["btp", "batiment", "indicateur avance"],
}

# ============================================================================
# PROMPT SYSTÈME ANSTAT-CI
# ============================================================================

SYSTEM_PROMPT = """Tu es ANSTAT-IA, l'assistant statistique officiel de l'Agence Nationale de la Statistique et de la Démographie de Côte d'Ivoire (ANSTAT-CI).

Ton rôle est de répondre aux questions sur les statistiques officielles de la Côte d'Ivoire en te basant UNIQUEMENT sur les documents fournis dans le contexte.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir des documents fournis — ne invente jamais de chiffres
2. Si l'information n'est pas dans le contexte, dis clairement : "Je n'ai pas cette information dans les documents disponibles."
3. Cite toujours tes sources : nom du document, année, page si disponible
4. Pour les chiffres, sois précis et mentionne l'année de référence
5. Si plusieurs documents donnent des chiffres différents, signale la différence
6. Réponds en français sauf si la question est en anglais
7. Sois factuel, concis et structuré

FORMAT DE RÉPONSE :
- Commence par la réponse directe
- Appuie avec les chiffres précis des documents
- Termine par les sources entre crochets [Source : Titre du document, Année, p.XX]"""


# ============================================================================
# UTILITAIRES
# ============================================================================

def norm(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def tokenize_fr(text: str) -> List[str]:
    text   = norm(text)
    tokens = re.findall(r'\b[\w]+\b', text)
    return [t for t in tokens if len(t) > 1]


def extraire_contexte(question: str) -> Dict[str, Any]:
    """Extrait années, localités, thèmes, séries depuis la question."""
    q = norm(question)
    ctx = {"annees": [], "localites": [], "themes": [],
           "series": [], "raptor_hint": "leaf"}

    # Années
    annees = [int(a) for a in re.findall(r'\b(19\d{2}|20\d{2})\b', question)]
    ctx["annees"] = [a for a in annees if a in ANNEES_CONNUES]
    if re.search(r'\b(recent|dernier|actuel|derniere)\b', q):
        ctx["annees"] += sorted(ANNEES_CONNUES)[-3:]

    # Localités
    for loc, mots in LOCALITES_CI.items():
        if any(norm(m) in q for m in mots):
            ctx["localites"].append(loc)

    # Thèmes
    for theme, mots in THEMES_GRAPH.items():
        if any(norm(m) in q for m in mots):
            ctx["themes"].append(theme)

    # Séries
    for serie, mots in SERIES_DOCS.items():
        if any(norm(m) in q for m in mots):
            ctx["series"].append(serie)

    # Niveau RAPTOR suggéré
    if re.search(r'\b(resume|synthese|global|ensemble|rapport|contenu)\b', q):
        ctx["raptor_hint"] = "section"
    elif re.search(r'\b(chapitre|section)\b', q):
        ctx["raptor_hint"] = "chapter"

    return ctx


# ============================================================================
# CHARGEMENT DES INDEX
# ============================================================================

def load_all_indexes() -> Dict[str, Any]:
    """
    Charge tous les index au démarrage.
    Retourne un dict avec faiss_index, bm25, chunk_map, graph_index,
    chunk_ids, chunk_texts, metadata.
    """
    print("\n" + "="*60)
    print("ANSTAT-IA — Chargement du système RAG V2")
    print("="*60)

    # Vérification dossier
    if not EMBEDDINGS_DIR.exists():
        raise FileNotFoundError(
            f"❌ Dossier embeddings_v2 introuvable : {EMBEDDINGS_DIR}\n"
            "   Lance d'abord embedding_v2.py"
        )

    t0 = time.time()

    # FAISS
    print("📦 Chargement FAISS index...")
    faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
    print(f"   ✅ {faiss_index.ntotal} vecteurs dim={faiss_index.d}")

    # BM25
    print("📝 Chargement BM25 index...")
    with open(BM25_INDEX_FILE, "rb") as f:
        bm25 = pickle.load(f)
    print(f"   ✅ {bm25.corpus_size} documents")

    # Chunk map
    print("🗂️  Chargement chunk map...")
    with open(CHUNK_MAP_FILE, "r", encoding="utf-8") as f:
        chunk_map = json.load(f)
    print(f"   ✅ {len(chunk_map)} chunks")

    # Graph index
    print("🗺️  Chargement graphe JSON...")
    with open(GRAPH_INDEX_FILE, "r", encoding="utf-8") as f:
        graph_index = json.load(f)
    print(f"   ✅ {len(graph_index.get('doc_relations', {}))} documents dans le graphe")

    # Metadata
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Liste ordonnée des chunk_ids (même ordre que FAISS)
    chunk_ids   = list(chunk_map.keys())
    chunk_texts = [chunk_map[cid].get("enriched_text",
                   chunk_map[cid].get("content", "")) for cid in chunk_ids]

    print(f"\n⏱️  Index chargés en {time.time()-t0:.1f}s")
    print(f"📊 Stats : {metadata.get('total_chunks',0)} chunks | "
          f"{metadata.get('chunk_stats',{}).get('tables_count',0)} tableaux | "
          f"{metadata.get('chunk_stats',{}).get('with_context',0)} avec contexte LLM")

    return {
        "faiss_index": faiss_index,
        "bm25":        bm25,
        "chunk_map":   chunk_map,
        "graph_index": graph_index,
        "chunk_ids":   chunk_ids,
        "chunk_texts": chunk_texts,
        "metadata":    metadata,
    }


# ============================================================================
# MODÈLES EMBEDDING + RERANKER
# ============================================================================

def load_models() -> Dict[str, Any]:
    """Charge BGE-M3 et BGE Reranker."""
    models = {}

    print("\n🤖 Chargement BGE-M3 (embedding)...")
    if FLAG_AVAILABLE:
        models["embedder"]      = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True)
        models["embedder_type"] = "bge-m3"
        print("   ✅ BGE-M3 via FlagEmbedding")
    else:
        models["embedder"]      = SentenceTransformer(EMBEDDING_MODEL_NAME)
        models["embedder_type"] = "st"
        print("   ✅ BGE-M3 via SentenceTransformers")

    print("🔁 Chargement BGE Reranker...")
    try:
        if FLAG_AVAILABLE:
            models["reranker"]      = FlagReranker(RERANKER_MODEL_NAME, use_fp16=True)
            models["reranker_type"] = "flag"
            print("   ✅ BGE Reranker via FlagEmbedding")
        else:
            models["reranker"]      = CrossEncoder(RERANKER_MODEL_NAME)
            models["reranker_type"] = "cross"
            print("   ✅ CrossEncoder via SentenceTransformers")
    except Exception as e:
        print(f"   ⚠️  Reranker non disponible : {e}")
        models["reranker"]      = None
        models["reranker_type"] = None

    return models


def encode_query(question: str, models: Dict) -> np.ndarray:
    """Encode la question en vecteur dense."""
    if models["embedder_type"] == "bge-m3":
        out  = models["embedder"].encode(
            [question], batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        emb  = out["dense_vecs"].astype(np.float32)
        norm_val = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norm_val, 1e-9)
    else:
        return models["embedder"].encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)


# ============================================================================
# CACHE SÉMANTIQUE
# ============================================================================

class SemanticCache:
    """
    Cache sémantique basé sur similarité cosine.
    Si une question similaire a déjà été posée (sim > threshold),
    retourne directement la réponse en cache.

    CORRECTIONS :
    - deque(maxlen) pour éviter les pop(0) O(n)
    - matrix maintenue incrémentalement (pas de vstack à chaque lookup)
    - vérification des mots-clés critiques (localités/thèmes/années)
      pour éviter les faux hits entre questions similaires
      ex : "rural" vs "rural et urbain" → sim > 0.92 mais contexte différent
    """
    def __init__(self, threshold: float = CACHE_THRESHOLD,
                 max_size: int = CACHE_MAX_SIZE):
        self.threshold = threshold
        self.max_size  = max_size
        from collections import deque
        self.questions : deque        = deque(maxlen=max_size)
        self.embeddings: deque        = deque(maxlen=max_size)
        self.responses : deque        = deque(maxlen=max_size)
        self._matrix  : Optional[np.ndarray] = None  # matrice pré-calculée
        self.hits   = 0
        self.misses = 0

    def _rebuild_matrix(self):
        """Reconstruit la matrice d'embeddings (appelée seulement si nécessaire)."""
        if self.embeddings:
            self._matrix = np.vstack(list(self.embeddings))
        else:
            self._matrix = None

    def lookup(self, q_emb: np.ndarray, question: str = "") -> Optional[Dict]:
        """
        Cherche une réponse similaire dans le cache.
        Vérifie en plus que les mots-clés critiques (localités, thèmes, années)
        sont identiques pour éviter les faux hits sémantiques.
        """
        if not self.embeddings or self._matrix is None:
            self.misses += 1
            return None

        scores = (self._matrix @ q_emb.T).flatten()
        best   = int(np.argmax(scores))

        if scores[best] >= self.threshold:
            # ── Vérification des mots-clés critiques ──────────────────────
            # Une similarité élevée ne suffit pas : "rural" et "rural et urbain"
            # peuvent dépasser 0.92 alors que les localités diffèrent.
            if question:
                q_ctx    = extraire_contexte(question)
                cached_q = list(self.questions)[best]
                c_ctx    = extraire_contexte(cached_q)

                localites_diff = set(q_ctx["localites"]) != set(c_ctx["localites"])
                themes_diff    = set(q_ctx["themes"])    != set(c_ctx["themes"])
                annees_diff    = set(q_ctx["annees"])    != set(c_ctx["annees"])
                series_diff    = set(q_ctx["series"])    != set(c_ctx["series"])

                if localites_diff or themes_diff or annees_diff or series_diff:
                    # Contexte différent → on ne sert pas le cache
                    self.misses += 1
                    return None
            # ──────────────────────────────────────────────────────────────
            self.hits += 1
            return list(self.responses)[best]

        self.misses += 1
        return None

    def store(self, q_emb: np.ndarray, question: str, response: Dict):
        """Stocke une nouvelle réponse dans le cache."""
        self.questions.append(question)
        self.embeddings.append(q_emb.flatten())
        self.responses.append(response)
        # Reconstruire la matrice après chaque ajout
        self._rebuild_matrix()

    def stats(self) -> str:
        total = self.hits + self.misses
        rate  = self.hits / total * 100 if total > 0 else 0
        return f"Cache conversationnel : {self.hits} hits / {total} requêtes ({rate:.1f}%)"


# ============================================================================
# CACHE SÉMANTIQUE 81 000 Q/R
# ============================================================================

class Cache81K:
    """
    Cache sémantique basé sur les 81 000 Q/R du dataset fine-tuning.
    Chargé depuis l'index FAISS pré-calculé par build_cache_semantique.py.

    Pipeline :
      Question → BGE-M3 → vecteur → FAISS search sur 81K Q/R
      Si similarité > CACHE_81K_THRESHOLD → réponse directe
      Sinon → RAG normal
    """

    def __init__(self):
        self.index    = None
        self.cache_map: Dict[str, Dict] = {}
        self.loaded   = False
        self.hits     = 0
        self.misses   = 0

    def load(self):
        """Charge l'index FAISS et la map Q/R depuis le disque."""
        if not CACHE_81K_FAISS_FILE.exists():
            print(f"   ⚠️  Cache 81K introuvable : {CACHE_81K_FAISS_FILE}")
            print(f"   ℹ️  Lance build_cache_semantique.py pour le créer")
            return

        if not CACHE_81K_MAP_FILE.exists():
            print(f"   ⚠️  Cache map introuvable : {CACHE_81K_MAP_FILE}")
            return

        try:
            print("📚 Chargement cache sémantique 81K Q/R...")
            t0          = time.time()
            self.index  = faiss.read_index(str(CACHE_81K_FAISS_FILE))

            with open(CACHE_81K_MAP_FILE, "r", encoding="utf-8") as f:
                self.cache_map = json.load(f)

            # ── CORRIGÉ : vérification de cohérence index ↔ map ──────────
            n_faiss = self.index.ntotal
            n_map   = len(self.cache_map)
            if n_faiss != n_map:
                print(f"   ⚠️  Désynchronisation : FAISS={n_faiss} vecteurs vs map={n_map} entrées")
                print(f"   ℹ️  Relance build_cache_semantique.py pour reconstruire le cache")
                # On continue quand même mais en mode dégradé
            # ──────────────────────────────────────────────────────────────

            self.loaded = True
            elapsed     = time.time() - t0
            print(f"   ✅ {self.index.ntotal:,} Q/R indexées chargées en {elapsed:.1f}s")
            print(f"   🎯 Seuil similarité : {CACHE_81K_THRESHOLD}")

        except Exception as e:
            print(f"   ❌ Erreur chargement cache 81K : {e}")
            self.loaded = False

    def lookup(self, q_emb: np.ndarray, question: str = "") -> Optional[Dict]:
        """
        Cherche si la question est similaire à une Q/R du dataset.
        Retourne la réponse si sim > CACHE_81K_THRESHOLD, sinon None.

        CORRECTIONS :
        - Vérification des mots-clés critiques (localités/thèmes/années/séries)
          pour éviter les faux hits : "rural" vs "rural et urbain" → refusé
        - Exception loggée au lieu d'être avalée silencieusement
        - Assertion de cohérence index/map au chargement (voir load())
        """
        if not self.loaded or self.index is None:
            self.misses += 1
            return None

        try:
            scores, indices = self.index.search(q_emb, 1)
            best_score = float(scores[0][0])
            best_idx   = int(indices[0][0])

            if best_score >= CACHE_81K_THRESHOLD and best_idx >= 0:
                entry = self.cache_map.get(str(best_idx))
                if entry:
                    # ── Vérification des mots-clés critiques ──────────────
                    # Même raisonnement que SemanticCache.lookup() :
                    # une forte similarité ne suffit pas si le contexte diffère.
                    if question:
                        q_ctx = extraire_contexte(question)
                        c_ctx = extraire_contexte(entry.get("question", ""))

                        localites_diff = set(q_ctx["localites"]) != set(c_ctx["localites"])
                        themes_diff    = set(q_ctx["themes"])    != set(c_ctx["themes"])
                        annees_diff    = set(q_ctx["annees"])    != set(c_ctx["annees"])
                        series_diff    = set(q_ctx["series"])    != set(c_ctx["series"])

                        if localites_diff or themes_diff or annees_diff or series_diff:
                            self.misses += 1
                            return None
                    # ──────────────────────────────────────────────────────
                    self.hits += 1
                    return {
                        "answer"  : entry["answer"],
                        "sources" : f"\n📚 SOURCE : {entry['source']}" if entry.get("source") else "",
                        "chunks"  : [],
                        "cached"  : True,
                        "cache_type": "81K",
                        "similarity": round(best_score, 4),
                        "matched_question": entry["question"],
                        "timing"  : {},   # ← CORRIGÉ : clé présente pour éviter KeyError
                    }

            self.misses += 1
            return None

        except Exception as e:
            # ← CORRIGÉ : exception loggée (était silencieuse)
            import logging
            logging.warning(f"[Cache81K.lookup] Erreur FAISS : {e}")
            self.misses += 1
            return None

    def stats(self) -> str:
        total = self.hits + self.misses
        rate  = self.hits / total * 100 if total > 0 else 0
        n     = self.index.ntotal if self.loaded and self.index else 0
        return f"Cache 81K : {n:,} Q/R | {self.hits} hits / {total} requêtes ({rate:.1f}%)"


# ============================================================================
# RECHERCHE HYBRIDE
# ============================================================================

def search_faiss(q_emb: np.ndarray, indexes: Dict,
                 top_k: int = TOP_K_FAISS) -> List[Tuple[str, float]]:
    """Recherche vectorielle FAISS."""
    scores_arr, indices_arr = indexes["faiss_index"].search(
        q_emb, min(top_k, len(indexes["chunk_ids"]))
    )
    results = []
    for score, idx in zip(scores_arr[0], indices_arr[0]):
        if 0 <= idx < len(indexes["chunk_ids"]):
            results.append((indexes["chunk_ids"][idx], float(score)))
    return results


def search_bm25(question: str, indexes: Dict,
                top_k: int = TOP_K_BM25) -> List[Tuple[str, float]]:
    """Recherche lexicale BM25."""
    if not indexes["bm25"]:
        return []
    tokens  = tokenize_fr(question)
    scores  = indexes["bm25"].get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(indexes["chunk_ids"][i], float(scores[i]))
            for i in top_idx if scores[i] > 0]


def rrf_fusion(faiss_res: List[Tuple[str, float]],
               bm25_res:  List[Tuple[str, float]],
               k: int = RRF_K) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    for rank, (cid, _) in enumerate(faiss_res):
        scores[cid] = scores.get(cid, 0.0) + RRF_W_FAISS / (k + rank + 1)
    for rank, (cid, _) in enumerate(bm25_res):
        scores[cid] = scores.get(cid, 0.0) + RRF_W_BM25 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def graph_bonus(cid: str, contexte: Dict, graph_index: Dict) -> float:
    """Calcule un bonus selon le graphe et le contexte de la question."""
    meta   = graph_index["chunk_meta"].get(cid, {})
    bonus  = 0.0

    annees    = set(str(a) for a in contexte.get("annees", []))
    localites = set(contexte.get("localites", []))
    themes    = set(contexte.get("themes", []))
    series    = set(contexte.get("series", []))

    # Année (priorité maximale pour ANSTAT)
    chunk_annee = str(meta.get("annee", ""))
    if annees and chunk_annee:
        bonus += 3.0 if chunk_annee in annees else -0.3

    # Localité
    bonus += len(localites & set(meta.get("localites", []))) * 2.0

    # Thème
    bonus += len(themes & set(meta.get("themes", []))) * 1.5

    # Série documentaire
    bonus += len(series & set(meta.get("series", []))) * 2.0

    # Niveau RAPTOR
    if meta.get("raptor_level") == contexte.get("raptor_hint", "leaf"):
        bonus += 0.5

    # Tableau statistique (données chiffrées précieuses)
    if meta.get("is_table"):
        bonus += 0.4

    return bonus


def rerank_candidates(question: str, candidates: List[Dict],
                      models: Dict, top_k: int = TOP_K_RERANK) -> List[Dict]:
    """BGE Reranker sur le texte enrichi (contexte LLM + contenu)."""
    if not models.get("reranker") or len(candidates) <= top_k:
        return candidates[:top_k]

    if models["reranker_type"] == "flag":
        pairs  = [[question, c["enriched_text"][:2000]] for c in candidates]
        scores = models["reranker"].compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
    else:
        pairs  = [(question, c["enriched_text"][:1500]) for c in candidates]
        scores = models["reranker"].predict(pairs)

    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def hybrid_search(question: str, models: Dict, indexes: Dict,
                  verbose: bool = False) -> List[Dict]:
    """
    Pipeline complet :
    FAISS + BM25 → RRF → Graphe JSON bonus → BGE Reranker → top K
    """
    t0 = time.time()

    # Contexte de la question
    contexte = extraire_contexte(question)

    # Embedding de la question
    q_emb = encode_query(question, models)

    # FAISS + BM25
    faiss_res = search_faiss(q_emb, indexes, TOP_K_FAISS)
    bm25_res  = search_bm25(question, indexes, TOP_K_BM25)

    # RRF fusion
    rrf_res = rrf_fusion(faiss_res, bm25_res)[:TOP_K_RRF]

    # Bonus graphe JSON
    scored = []
    for cid, rrf_score in rrf_res:
        bonus    = graph_bonus(cid, contexte, indexes["graph_index"])
        combined = rrf_score + bonus * 0.1
        scored.append((cid, combined, rrf_score, bonus))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Construire les candidats
    candidates = []
    chunk_map  = indexes["chunk_map"]
    graph_meta = indexes["graph_index"]["chunk_meta"]

    for cid, combined, rrf_score, bonus in scored:
        cm = chunk_map.get(cid, {})
        gm = graph_meta.get(cid, {})
        candidates.append({
            "chunk_id":       cid,
            "content":        cm.get("content", ""),
            "enriched_text":  cm.get("enriched_text", cm.get("content", "")),
            "context_desc":   cm.get("context_desc", ""),
            "combined_score": combined,
            "rrf_score":      rrf_score,
            "graph_bonus":    bonus,
            "rerank_score":   None,
            "document_name":  cm.get("document_name", ""),
            "document_title": cm.get("document_title", ""),
            "source_url":     cm.get("source_url", ""),
            "page_number":    cm.get("page_number", -1),
            "themes":         cm.get("themes", []),
            "annee":          gm.get("annee", cm.get("annee")),
            "localites":      gm.get("localites", []),
            "raptor_level":   cm.get("raptor_level", "leaf"),
            "is_table":       cm.get("is_table", False),
        })

    # BGE Reranker
    results = rerank_candidates(question, candidates, models, TOP_K_RERANK)

    if verbose:
        elapsed = time.time() - t0
        print(f"\n   🔍 Contexte : années={contexte['annees']} | "
              f"localités={contexte['localites']} | thèmes={contexte['themes'][:3]}")
        print(f"   📐 FAISS:{len(faiss_res)} | BM25:{len(bm25_res)} | "
              f"RRF:{len(rrf_res)} | Rerank→{len(results)} | ⏱️ {elapsed:.2f}s")

    return results


# ============================================================================
# CONSTRUCTION DU PROMPT
# ============================================================================

def build_prompt(question: str, chunks: List[Dict],
                 history: List[Dict] = None) -> List[Dict]:
    """
    Construit le prompt pour Qwen avec :
    - Historique conversationnel
    - Contexte (chunks récupérés)
    - Question
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Historique (limité à MAX_HISTORY tours)
    if history:
        for turn in history[-MAX_HISTORY:]:
            messages.append({"role": "user",      "content": turn["question"]})
            messages.append({"role": "assistant",  "content": turn["answer"]})

    # Contexte documentaire
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        titre   = chunk.get("document_title") or chunk.get("document_name", "N/A")
        annee   = chunk.get("annee", "")
        page    = chunk.get("page_number", "")
        url     = chunk.get("source_url", "")
        niveau  = chunk.get("raptor_level", "leaf")
        tableau = " [TABLEAU STATISTIQUE]" if chunk.get("is_table") else ""

        source_line = f"[Source {i} : {titre}"
        if annee:
            source_line += f", {annee}"
        if page and page != -1:
            source_line += f", p.{page}"
        if url:
            source_line += f" — {url}"
        source_line += f"]{tableau}"

        context_parts.append(
            f"{source_line}\n"
            f"Niveau : {niveau}\n"
            f"{chunk['content']}"
        )

    context_str = "\n\n" + "─"*50 + "\n\n".join(context_parts)

    # Message utilisateur
    user_msg = (
        f"DOCUMENTS DE RÉFÉRENCE :\n{context_str}\n\n"
        f"{'─'*50}\n\n"
        f"QUESTION : {question}"
    )
    messages.append({"role": "user", "content": user_msg})

    return messages


# ============================================================================
# APPEL VLLM
# ============================================================================

def call_vllm(messages: List[Dict],
              max_tokens: int = 800,
              temperature: float = 0.1) -> str:
    """Appelle le vLLM Qwen 2.5 via /v1/chat/completions."""
    try:
        resp = requests.post(
            VLLM_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model":       VLLM_MODEL,
                "messages":    messages,
                "max_tokens":  max_tokens,
                "temperature": temperature,
                "stop":        ["<|im_end|>", "</s>"],
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ Erreur vLLM {resp.status_code} : {resp.text[:200]}"
    except requests.exceptions.Timeout:
        return "❌ Timeout : le modèle met trop de temps à répondre."
    except requests.exceptions.ConnectionError:
        return "❌ Impossible de se connecter au vLLM. Vérifie que le serveur tourne."
    except Exception as e:
        return f"❌ Erreur inattendue : {str(e)}"


def check_vllm() -> bool:
    """Vérifie que le vLLM est accessible."""
    try:
        resp = requests.get(
            VLLM_URL.replace("/v1/chat/completions", "/health"),
            timeout=5
        )
        return resp.status_code == 200
    except Exception:
        # Essayer directement avec une question simple
        try:
            resp = requests.post(
                VLLM_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model":      VLLM_MODEL,
                    "messages":   [{"role": "user", "content": "Bonjour"}],
                    "max_tokens": 10,
                },
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False


# ============================================================================
# FORMATAGE DES SOURCES
# ============================================================================

def format_sources(chunks: List[Dict]) -> str:
    """Formate les sources pour affichage."""
    lines = ["\n📚 SOURCES CONSULTÉES :"]
    seen  = set()
    for i, c in enumerate(chunks, 1):
        titre = c.get("document_title") or c.get("document_name", "N/A")
        annee = c.get("annee", "")
        page  = c.get("page_number", "")
        url   = c.get("source_url", "")
        score = c.get("rerank_score")
        level = c.get("raptor_level", "leaf")

        key   = f"{titre}_{annee}"
        if key in seen:
            continue
        seen.add(key)

        line  = f"  [{i}] {titre}"
        if annee:
            line += f" ({annee})"
        if page and page != -1:
            line += f" — p.{page}"
        if score is not None:
            line += f" | score: {score:.3f}"
        line += f" | niveau: {level}"
        if c.get("is_table"):
            line += " 📊"
        lines.append(line)
        if url:
            lines.append(f"       🔗 {url}")

    return "\n".join(lines)


# ============================================================================
# MOTEUR RAG PRINCIPAL
# ============================================================================

class ANSTATRagSystem:
    """
    Système RAG complet ANSTAT-CI.
    Gère : indexation, cache, historique, génération.
    """

    def __init__(self):
        self.indexes    = None
        self.models     = None
        self.cache      = SemanticCache()      # cache conversationnel
        self.cache_81k  = Cache81K()           # cache 81 000 Q/R
        self.history    : List[Dict] = []
        self.vllm_ok    = False
        self._ready     = False

    def load(self):
        """Charge tous les composants."""
        self.indexes = load_all_indexes()
        self.models  = load_models()

        # Chargement cache sémantique 81K Q/R
        self.cache_81k.load()

        # Vérification vLLM
        print("\n🌐 Vérification vLLM...")
        self.vllm_ok = check_vllm()
        if self.vllm_ok:
            print(f"   ✅ vLLM accessible : {VLLM_URL}")
        else:
            print(f"   ⚠️  vLLM non accessible ({VLLM_URL})")
            print(f"   ℹ️  Le système fonctionnera en mode retrieval seul")

        self._ready = True
        print("\n✅ ANSTAT-IA prêt !\n")

    def ask(self, question: str,
            top_k: int = TOP_K_RERANK,
            verbose: bool = False,
            use_cache: bool = CACHE_ENABLED) -> Dict[str, Any]:
        """
        Répond à une question.
        Retourne : answer, sources, chunks, cached, timing
        """
        if not self._ready:
            return {"answer": "❌ Système non chargé. Lance d'abord .load()",
                    "sources": "", "chunks": [], "cached": False}

        t_total = time.time()
        question = question.strip()

        # ── Étape 1 : Cache sémantique 81K Q/R ──────────────────────────
        q_emb = encode_query(question, self.models)
        if use_cache and self.cache_81k.loaded:
            cached_81k = self.cache_81k.lookup(q_emb, question)   # ← CORRIGÉ : passe question
            if cached_81k:
                if verbose:
                    print(f"   ⚡ Cache 81K hit ! sim={cached_81k['similarity']} | "
                          f"Q matchée : {cached_81k['matched_question'][:60]}...")
                cached_81k["timing"] = {"total": round(time.time() - t_total, 3)}
                # Ajouter aussi au cache conversationnel
                self.cache.store(q_emb, question, cached_81k)
                return cached_81k

        # ── Étape 2 : Cache conversationnel (questions récentes) ─────────
        if use_cache:
            cached = self.cache.lookup(q_emb, question)            # ← CORRIGÉ : passe question
            if cached:
                if verbose:
                    print(f"   ⚡ Cache conversationnel hit ! {self.cache.stats()}")
                return {**cached, "cached": True,
                        "timing": {"total": time.time() - t_total}}

        # ── Recherche hybride ────────────────────────────────────────────
        t_search = time.time()
        chunks   = hybrid_search(question, self.models, self.indexes,
                                 verbose=verbose)
        t_search = time.time() - t_search

        if not chunks:
            return {
                "answer":  "Je n'ai trouvé aucun passage pertinent dans les documents.",
                "sources": "", "chunks": [], "cached": False,
                "timing":  {"search": t_search, "total": time.time() - t_total}
            }

        # ── Génération LLM ───────────────────────────────────────────────
        t_gen = time.time()
        if self.vllm_ok:
            messages = build_prompt(question, chunks, self.history)
            answer   = call_vllm(messages)
        else:
            # Fallback sans LLM : réponse extractive
            answer = self._extractive_answer(question, chunks)
        t_gen = time.time() - t_gen

        # Sources formatées
        sources = format_sources(chunks)

        # ── Historique ───────────────────────────────────────────────────
        self.history.append({
            "question": question,
            "answer":   answer,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-MAX_HISTORY * 2:]

        result = {
            "answer":  answer,
            "sources": sources,
            "chunks":  chunks,
            "cached":  False,
            "timing":  {
                "search": round(t_search, 2),
                "gen":    round(t_gen, 2),
                "total":  round(time.time() - t_total, 2),
            }
        }

        # Stocker dans le cache
        if use_cache:
            self.cache.store(q_emb, question, result)

        return result

    def _extractive_answer(self, question: str, chunks: List[Dict]) -> str:
        """Réponse extractive si vLLM indisponible."""
        lines = [f"⚠️  Mode retrieval seul (vLLM indisponible)\n"]
        lines.append(f"Question : {question}\n")
        lines.append("Passages pertinents trouvés :\n")
        for i, c in enumerate(chunks[:3], 1):
            titre  = c.get("document_title", "N/A")
            annee  = c.get("annee", "")
            extrait = c["content"][:400]
            lines.append(f"\n[{i}] {titre} {f'({annee})' if annee else ''}")
            lines.append(f"    {extrait}...")
        return "\n".join(lines)

    def clear_history(self):
        """Efface l'historique conversationnel."""
        self.history = []
        print("   🗑️  Historique effacé")

    def clear_cache(self):
        """Vide le cache conversationnel (le cache 81K reste intact)."""
        self.cache = SemanticCache()
        print("   🗑️  Cache conversationnel vidé (cache 81K conservé)")

    def stats(self):
        """Affiche les statistiques du système."""
        meta = self.indexes["metadata"] if self.indexes else {}
        print(f"\n{'='*55}")
        print("STATISTIQUES ANSTAT-IA V3")
        print(f"{'='*55}")
        print(f"   Chunks indexés    : {meta.get('total_chunks', '?')}")
        print(f"   Tableaux          : {meta.get('chunk_stats',{}).get('tables_count','?')}")
        print(f"   Contexte LLM      : {meta.get('chunk_stats',{}).get('with_context','?')}")
        print(f"   Modèle embedding  : {meta.get('embedding_model','?')}")
        print(f"   Modèle reranker   : {meta.get('reranker_model','?')}")
        print(f"   vLLM              : {'✅' if self.vllm_ok else '❌'}")
        print(f"   Historique        : {len(self.history)} tours")
        print(f"   {self.cache_81k.stats()}")
        print(f"   {self.cache.stats()}")
        print(f"{'='*55}\n")


# ============================================================================
# INTERFACE TERMINAL
# ============================================================================

def print_banner():
    print("\n" + "="*60)
    print("  ANSTAT-IA — Assistant Statistique Officiel CI")
    print("  Système RAG V3 — BGE-M3 + BM25 + RRF + Cache 81K + Qwen 2.5")
    print("="*60)
    print("  Commandes spéciales :")
    print("   /stats    → statistiques du système")
    print("   /history  → voir l'historique")
    print("   /clear    → effacer l'historique")
    print("   /cache    → vider le cache conversationnel")
    print("   /verbose  → activer/désactiver les logs")
    print("   /quit     → quitter")
    print("="*60 + "\n")


def run_terminal(rag: ANSTATRagSystem):
    """Interface terminal interactive."""
    print_banner()
    verbose = False

    while True:
        try:
            question = input("❓ Votre question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Au revoir !")
            break

        if not question:
            continue

        # Commandes spéciales
        if question.lower() == "/quit":
            print("👋 Au revoir !")
            break
        elif question.lower() == "/stats":
            rag.stats()
            continue
        elif question.lower() == "/history":
            if rag.history:
                print(f"\n📜 Historique ({len(rag.history)} tours) :")
                for i, h in enumerate(rag.history, 1):
                    print(f"  [{i}] Q: {h['question'][:80]}")
                    print(f"       R: {h['answer'][:100]}...")
            else:
                print("   Historique vide.")
            continue
        elif question.lower() == "/clear":
            rag.clear_history()
            continue
        elif question.lower() == "/cache":
            rag.clear_cache()
            continue
        elif question.lower() == "/verbose":
            verbose = not verbose
            print(f"   Verbose : {'✅ activé' if verbose else '❌ désactivé'}")
            continue

        # Traitement de la question
        print(f"\n⏳ Recherche en cours...")
        result = rag.ask(question, verbose=verbose)

        # Affichage
        print("\n" + "="*60)
        if result.get("cached"):
            cache_type = result.get("cache_type", "")
            sim        = result.get("similarity", "")
            if cache_type == "81K":
                print(f"⚡ RÉPONSE (Cache 81K — similarité {sim}) :")
            else:
                print("⚡ RÉPONSE (depuis le cache conversationnel) :")
        else:
            t = result.get("timing", {})
            print(f"💬 RÉPONSE (recherche:{t.get('search','?')}s | "
                  f"génération:{t.get('gen','?')}s | "
                  f"total:{t.get('total','?')}s) :")
        print("="*60)
        print(result["answer"])
        print(result["sources"])
        print()


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    rag = ANSTATRagSystem()
    rag.load()

    # Questions de test au démarrage
    print("\n🧪 Test rapide du système...")
    test_questions = [
        "Quel est le taux de pauvreté en Côte d'Ivoire en 2021 ?",
        "Indice du chiffre d'affaires dans l'industrie",
    ]
    for q in test_questions:
        print(f"\n   Test : {q}")
        r = rag.ask(q, verbose=True)
        print(f"   ✅ Réponse ({r['timing']['total']}s) : {r['answer'][:150]}...")

    print("\n✅ Tests terminés — lancement de l'interface interactive\n")

    # Interface terminal
    run_terminal(rag)


if __name__ == "__main__":
    main()