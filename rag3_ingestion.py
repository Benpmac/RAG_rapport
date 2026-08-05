"""
rag3_ingestion.py — DAG Airflow ingestion incrémentale RAG_3 ANSTAT-CI
=======================================================================

RÔLE
  Orchestre la mise à jour automatique de la base vectorielle du RAG_3.
  Détecte les PDFs nouveaux, modifiés (via SHA-256) et supprimés dans
  ~/rag_3/documents/, re-chunke et réindexe uniquement ce qui a changé.

INPUTS
  ~/rag_3/documents/*.pdf    ← PDFs à indexer (dépôt manuel)
  ~/rag_3/manifest.json      ← hashes SHA-256 des PDFs déjà indexés
  ~/rag_3/CHUNK.py.py        ← pipeline de chunking RAPTOR (importé dynamiquement)
  ~/rag_3/embedding_v2.py    ← pipeline d'indexation FAISS+BM25 (importé dynamiquement)

OUTPUTS
  ~/rag_3/chunks_output_v2/*_chunks.json  ← chunks JSON par PDF (via CHUNK.py.py)
  ~/rag_3/embeddings_v2/faiss_index.bin   ← index vectoriel mis à jour
  ~/rag_3/embeddings_v2/bm25_index.pkl    ← index BM25 mis à jour
  ~/rag_3/embeddings_v2/chunk_map.json    ← métadonnées chunks mises à jour
  ~/rag_3/embeddings_v2/graph_index.json  ← graph bonus mis à jour
  ~/rag_3/embeddings_v2/embeddings.npz    ← vecteurs mis à jour
  ~/rag_3/manifest.json                   ← hashes SHA-256 mis à jour

TÂCHES AIRFLOW (TaskFlow API)
  detecter_changements()      → compare PDFs du disque vs manifest SHA-256
                                retourne {nouveaux, modifies, supprimes}
  y_a_t_il_du_travail()       → short_circuit : stoppe si aucun changement
  supprimer_anciens_chunks()  → supprime les *_chunks.json des PDFs retirés/modifiés
  chunker_pdfs()              → appelle RaptorRAGPipeline.process_document() par PDF
  construire_index()          → appelle embedding_v2.build() + met à jour le manifest

FONCTIONS UTILITAIRES
  _sha256(path)               → hash SHA-256 d'un fichier
  _charger_manifest()         → charge manifest.json (avec migration ancien format)
  _sauvegarder_manifest(m)    → écrit manifest.json

PIPELINE
  detecter_changements
          │
  y_a_t_il_du_travail   ← court-circuit si rien à faire
          │
     ┌────┴────┐
     │         │
  supprimer  chunker_pdfs
     │         │
     └────┬────┘
          │
  construire_index      ← embedding_v2.build() + maj manifest
"""

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.utils.email import send_email

# ── Configuration ──────────────────────────────────────────────────────────────
RAG3_DIR       = Path.home() / "rag_3"
DOCUMENTS_DIR  = RAG3_DIR / "documents"
CHUNKS_DIR     = RAG3_DIR / "chunks_output_v2"
MANIFEST_PATH  = RAG3_DIR / "manifest.json"

CHUNK_SCRIPT   = RAG3_DIR / "CHUNK.py.py"
EMBED_SCRIPT   = RAG3_DIR / "embedding_v2.py"

# LLM pour l'enrichissement contextuel RAPTOR (optionnel)
VLLM_URL       = "http://192.168.1.230:30179/v1/chat/completions"
USE_LLM        = True   # Mettre False si vLLM inaccessible

# Notifications email
EMAIL_DEST     = [
    "pierremariecdricbenson@gmail.com",
    "j.migone@stat.plan.gouv.ci",
]


# ── Callbacks email ────────────────────────────────────────────────────────────

def _envoyer_email_succes(context):
    task_id  = context["task_instance"].task_id
    run_id   = context["run_id"]
    duration = context["task_instance"].duration
    dur_str  = f"{duration:.0f}s" if duration else "?"
    send_email(
        to      = EMAIL_DEST,
        subject = f"[RAG3 ✅] {task_id} terminé",
        html_content = f"""
        <h3>✅ Tâche réussie — RAG3 ANSTAT</h3>
        <table>
          <tr><td><b>Tâche</b></td><td>{task_id}</td></tr>
          <tr><td><b>Run</b></td><td>{run_id}</td></tr>
          <tr><td><b>Durée</b></td><td>{dur_str}</td></tr>
          <tr><td><b>Heure</b></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
        </table>
        <p>Logs : <a href="http://localhost:8080">Interface Airflow</a></p>
        """,
    )


def _envoyer_email_echec(context):
    task_id   = context["task_instance"].task_id
    run_id    = context["run_id"]
    exception = context.get("exception", "inconnue")
    send_email(
        to      = EMAIL_DEST,
        subject = f"[RAG3 ❌] {task_id} a échoué",
        html_content = f"""
        <h3>❌ Échec — RAG3 ANSTAT</h3>
        <table>
          <tr><td><b>Tâche</b></td><td>{task_id}</td></tr>
          <tr><td><b>Run</b></td><td>{run_id}</td></tr>
          <tr><td><b>Erreur</b></td><td>{exception}</td></tr>
          <tr><td><b>Heure</b></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
        </table>
        <p>Logs : <a href="http://localhost:8080">Interface Airflow</a></p>
        """,
    )


# ── Utilitaires manifest ───────────────────────────────────────────────────────

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloc in iter(lambda: f.read(65536), b""):
            h.update(bloc)
    return h.hexdigest()


def _charger_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # Migration : ancien format liste → dict avec hashes
    if isinstance(data, list):
        print("Migration manifest : conversion ancien format en cours...")
        nouveau = {}
        for nom in data:
            chemin = DOCUMENTS_DIR / nom
            if chemin.exists():
                nouveau[nom] = {
                    "hash":       _sha256(str(chemin)),
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                nouveau[nom] = {"hash": "inconnu", "indexed_at": "inconnu"}
        _sauvegarder_manifest(nouveau)
        return nouveau
    return data


def _sauvegarder_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# ── DAG ───────────────────────────────────────────────────────────────────────

@dag(
    dag_id="rag3_anstat_ingestion",
    start_date=datetime(2026, 8, 1),
    schedule="*/30 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["rag3", "anstat", "faiss", "bm25", "raptor"],
    default_args={
        "on_failure_callback": _envoyer_email_echec,
    },
)
def rag3_ingestion():

    @task
    def detecter_changements() -> dict:
        """
        Compare les PDFs dans ~/rag_3/documents/ avec le manifest SHA-256.
        Retourne {nouveaux, modifies, supprimes} — listes de noms de fichiers.
        """
        if not DOCUMENTS_DIR.exists():
            raise AirflowSkipException(f"Dossier absent : {DOCUMENTS_DIR}")

        manifest = _charger_manifest()

        pdfs_disque = {
            f for f in os.listdir(DOCUMENTS_DIR)
            if f.lower().endswith(".pdf")
            and not f.endswith(".Identifier")
        }

        nouveaux, modifies, supprimes = [], [], []

        for nom in pdfs_disque:
            chemin     = str(DOCUMENTS_DIR / nom)
            hash_actuel = _sha256(chemin)
            if nom not in manifest:
                nouveaux.append(nom)
            elif manifest[nom]["hash"] != hash_actuel:
                modifies.append(nom)

        for nom in manifest:
            if nom not in pdfs_disque:
                supprimes.append(nom)

        print(f"Détection : {len(pdfs_disque)} PDF(s) sur disque")
        print(f"  Nouveaux  : {len(nouveaux)}")
        print(f"  Modifiés  : {len(modifies)}")
        print(f"  Supprimés : {len(supprimes)}")

        return {
            "nouveaux":  nouveaux,
            "modifies":  modifies,
            "supprimes": supprimes,
        }

    @task.short_circuit
    def y_a_t_il_du_travail(changements: dict) -> bool:
        """Court-circuite le pipeline si aucun changement détecté."""
        total = (
            len(changements["nouveaux"])
            + len(changements["modifies"])
            + len(changements["supprimes"])
        )
        if total == 0:
            print("Aucun changement détecté, rien à faire.")
            return False
        return True

    @task
    def supprimer_anciens_chunks(changements: dict) -> None:
        """
        Supprime les *_chunks.json des PDFs retirés ou modifiés.
        embedding_v2.py détectera les vecteurs orphelins et les retirera
        lors de la reconstruction des index.
        """
        a_supprimer = changements["supprimes"] + changements["modifies"]
        if not a_supprimer:
            print("Aucun chunk à supprimer.")
            return

        supprime_count = 0
        for nom in a_supprimer:
            stem       = Path(nom).stem
            chunk_file = CHUNKS_DIR / f"{stem}_chunks.json"
            if chunk_file.exists():
                chunk_file.unlink()
                print(f"  Supprimé : {chunk_file.name}")
                supprime_count += 1

        # Retirer les PDFs supprimés du manifest (les modifiés y restent
        # jusqu'à ce que l'indexation réussisse)
        manifest = _charger_manifest()
        for nom in changements["supprimes"]:
            manifest.pop(nom, None)
        _sauvegarder_manifest(manifest)

        print(f"{supprime_count} fichier(s) chunk supprimé(s). "
              f"Manifest mis à jour : {len(changements['supprimes'])} retrait(s).")

    @task(execution_timeout=None, on_success_callback=_envoyer_email_succes)
    def chunker_pdfs(changements: dict) -> list:
        """
        Appelle RaptorRAGPipeline.process_document() sur chaque PDF
        nouveau ou modifié. La config est adaptée pour WSL2 (chemins,
        camelot désactivé, parallélisme minimal).
        Retourne la liste des PDFs traités avec succès.
        """
        pdfs_a_chunker = changements["nouveaux"] + changements["modifies"]
        if not pdfs_a_chunker:
            print("Aucun PDF à chunker.")
            return []

        if not CHUNK_SCRIPT.exists():
            raise FileNotFoundError(
                f"Script CHUNK.py.py introuvable : {CHUNK_SCRIPT}\n"
                "Vérifiez que le disque Windows est monté dans WSL2 (/mnt/c/)."
            )

        # Import dynamique — le nom de fichier contient des points
        spec     = importlib.util.spec_from_file_location("chunk_pipeline", str(CHUNK_SCRIPT))
        chunk_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chunk_mod)

        PipelineConfig    = chunk_mod.PipelineConfig
        RaptorRAGPipeline = chunk_mod.RaptorRAGPipeline

        config = PipelineConfig(
            documents_dir   = DOCUMENTS_DIR,
            output_dir      = CHUNKS_DIR,
            use_llm_context = USE_LLM,
            vllm_url        = VLLM_URL,
            use_camelot     = False,  # dépendances complexes non installées
            use_pdfplumber  = True,
            max_workers     = 1,      # séquentiel pour préserver la RAM
        )

        pipeline = RaptorRAGPipeline(config)
        traites  = []

        for nom in pdfs_a_chunker:
            pdf_path = DOCUMENTS_DIR / nom
            if not pdf_path.exists():
                print(f"  Introuvable (ignoré) : {nom}")
                continue

            print(f"\n── Chunking : {nom} ──")
            result = pipeline.process_document(pdf_path)

            if result.get("success"):
                traites.append(nom)
                print(f"  OK — {result['chunks_count']} chunks "
                      f"en {result.get('processing_time', 0):.1f}s")
            else:
                print(f"  ÉCHEC : {result.get('error', 'inconnu')}")

        print(f"\n{len(traites)}/{len(pdfs_a_chunker)} PDF(s) chunkés avec succès.")
        return traites

    @task(execution_timeout=None, on_success_callback=_envoyer_email_succes)
    def construire_index(pdfs_traites: list, changements: dict) -> str:
        """
        1. Appelle embedding_v2.build(incremental=True) :
           - retire les vecteurs orphelins (PDFs supprimés)
           - encode seulement les nouveaux chunks avec BGE-M3
           - reconstruit FAISS IVFFlat + BM25Okapi
        2. Met à jour le manifest SHA-256 des PDFs indexés avec succès.
        """
        if not EMBED_SCRIPT.exists():
            raise FileNotFoundError(f"embedding_v2.py introuvable : {EMBED_SCRIPT}")

        # Import dynamique
        spec    = importlib.util.spec_from_file_location("embedding_v2", str(EMBED_SCRIPT))
        emb_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(emb_mod)

        emb_mod.build(incremental=True)

        # Mise à jour du manifest APRÈS indexation réussie
        manifest = _charger_manifest()
        for nom in pdfs_traites:
            chemin = DOCUMENTS_DIR / nom
            if chemin.exists():
                manifest[nom] = {
                    "hash":       _sha256(str(chemin)),
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }
        _sauvegarder_manifest(manifest)

        msg = (
            f"{len(pdfs_traites)} PDF(s) indexés. "
            f"Manifest : {len(manifest)} PDF(s) suivis."
        )
        print(msg)
        return msg

    # ── Câblage des tâches ────────────────────────────────────────────────────
    #
    #  detecter_changements
    #         │ (XComArg)
    #  y_a_t_il_du_travail    ← short-circuit si aucun changement
    #         │
    #    ┌────┴────┐
    #    │         │
    # supprimer  chunker_pdfs
    #    │         │ (XComArg pdfs_traites)
    #    └────┬────┘
    #         │
    #  construire_index

    changements  = detecter_changements()
    go           = y_a_t_il_du_travail(changements)
    suppression  = supprimer_anciens_chunks(changements)
    pdfs_traites = chunker_pdfs(changements)
    indexation   = construire_index(pdfs_traites, changements)

    go >> [suppression, pdfs_traites]
    suppression >> indexation


rag3_ingestion()
