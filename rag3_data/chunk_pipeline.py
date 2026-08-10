# ============================================================================
# CHUNK.py.py — PIPELINE DE CHUNKING RAPTOR ANSTAT-CI
# ============================================================================
#
# RÔLE
#   Transforme un PDF brut en fichier JSON de chunks structurés,
#   prêts à être encodés par embedding_v2.py.
#
# INPUT
#   Un fichier PDF (rapport statistique ANSTAT-CI)
#
# OUTPUT
#   ~/rag_3/chunks_output_v2/<nom>_chunks.json
#   Format : { "metadata": {...}, "chunks": [ {chunk_id, text, metadata}, ... ] }
#   Niveaux RAPTOR : leaf (L0) · section (L1) · chapter (L2) · document (L3)
#
# CLASSES
#   PipelineConfig            configuration du pipeline (chemins, LLM, chunking)
#     documents_dir           dossier source des PDFs
#     output_dir              dossier de sortie des JSON
#     chunk_size_l0           taille cible des chunks feuilles (~1400 car)
#     use_llm_context         active l'enrichissement contextuel via LLM
#     vllm_url / vllm_model   endpoint et modèle vLLM pour les résumés RAPTOR
#     use_camelot             active l'extraction de tableaux avec bordures
#     use_pdfplumber          active l'extraction de tableaux sans bordures
#   RaptorRAGPipeline         pipeline principal
#     .process_document(pdf)  → traite un PDF → retourne {success, chunks_count, ...}
#     .process_all()          → traite tous les PDFs du dossier documents_dir
#     .extract_pages(pdf)     → extrait les pages (ODL → PyMuPDF fallback)
#     .filter_pages(pages)    → filtre les pages bruit (C0→C10)
#     .extract_tables(pdf)    → extrait les tableaux (pdfplumber + camelot)
#     .log(msg)               → log horodaté dans fichier + stdout
#
# FONCTIONS UTILITAIRES (extraction)
#   extract_with_opendataloader(pdf)     → pages via ODL (meilleure qualité)
#   extract_text_pymupdf_fallback(pdf)   → pages via PyMuPDF (fallback)
#   extract_tables_pdfplumber(pdf, ...)  → tableaux sans bordures
#   extract_tables_camelot(pdf, ...)     → tableaux avec bordures
#   merge_table_sources(plumber, camelot)→ fusion des deux sources tableaux
#
# FONCTIONS UTILITAIRES (filtrage)
#   clean_raw_text(text)                 → nettoyage basique (unicode, espaces)
#   remove_header_footer_lines(text)     → supprime en-têtes/pieds répétitifs
#   detect_repeated_headers_footers(...) → détecte les lignes répétées entre pages
#   is_structural_noise_page(text)       → détecte pages de bruit (TOC, pages vides)
#   is_questionnaire_chunk(text)         → détecte les grilles de questionnaire
#   is_true_junk_chunk(text)             → détecte les chunks inutilisables
#   deduplicate_chunks(chunks, seuil)    → supprime les doublons (Jaccard)
#
# FONCTIONS UTILITAIRES (RAPTOR)
#   raptor_chunk_level0(pages, ...)      → découpe en chunks feuilles (L0)
#   raptor_build_level1(leaves, ...)     → résumés de sections (L1)
#   raptor_build_level2(sections, ...)   → résumés de chapitres (L2)
#   raptor_build_level3(chapters, ...)   → résumé document complet (L3)
#   _summarize_group(text, ...)          → appel LLM pour résumer un groupe
#
# FONCTIONS UTILITAIRES (contexte & assemblage)
#   generate_contextual_description(...) → enrichissement LLM du chunk
#   assemble_chunk(text, ...)            → construit le dict chunk final
#   detect_themes(text, ...)             → détecte les thèmes statistiques
#   lookup_document_info(pdf_path)       → métadonnées du document (catalogue)
#
# USAGE
#   Importé dynamiquement par rag3_ingestion.py via importlib
#   Ou directement : python CHUNK.py.py
# ============================================================================

import os, json, time, hashlib, traceback, re, unicodedata, tempfile, shutil
import multiprocessing, threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import Counter, defaultdict

import fitz          # PyMuPDF
import pandas as pd
import numpy as np

# ── OpenDataLoader ─────────────────────────────────────────────────────────
try:
    import opendataloader_pdf
    OPENDATALOADER_AVAILABLE = True
    print("✅ OpenDataLoader actif")
except ImportError:
    OPENDATALOADER_AVAILABLE = False
    print("⚠️  OpenDataLoader non installé — pip install opendataloader-pdf")

# ── pdfplumber (tableaux) ───────────────────────────────────────────────────
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    print("✅ pdfplumber actif")
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️  pdfplumber non installé — pip install pdfplumber")

# ── Camelot (tableaux avec bordures) ───────────────────────────────────────
try:
    import camelot
    CAMELOT_AVAILABLE = True
    print("✅ Camelot actif")
except ImportError:
    CAMELOT_AVAILABLE = False
    print("⚠️  Camelot non installé — pip install camelot-py[cv]")

# ── LLM local pour contexte RAPTOR (Ollama ou vLLM) ────────────────────────
try:
    import requests
    REQUESTS_AVAILABLE = True
    print("requests actif")
except ImportError:
    REQUESTS_AVAILABLE = False
    print("requests non installé")

# ============================================================================
# CATALOGUE DOCUMENTS ANSTAT-CI
# ============================================================================

DOCUMENT_CATALOG = {
    # ── Documents originaux ─────────────────────────────────────────────────
    "file_projet1719943795": {
        "titre": "Enquête sur l'Accès et l'Utilisation des TIC par les Ménages en Côte d'Ivoire (DATAFID)",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719943795.pdf",
        "themes": ["tic", "ménages", "numérique"],
        "annee": None,
    },
    "sigi2022": {
        "titre": "Rapport SIGI 2022",
        "url": "https://www.anstat.ci/assets/projet/SIGI2022.pdf",
        "themes": ["genre", "inégalités", "social"],
        "annee": 2022,
    },
    "ips_education2022": {
        "titre": "Enquête sur les Indicateurs des Prestations de Services de l'Éducation EIPSE 2022",
        "url": "https://www.anstat.ci/assets/projet/IPS_EDUCATION2022.pdf",
        "themes": ["education", "services", "indicateurs"],
        "annee": 2022,
    },
    "eds2021": {
        "titre": "Enquête Démographique et de Santé 2021",
        "url": "https://www.anstat.ci/assets/projet/EDS2021.pdf",
        "themes": ["demographie", "sante", "ménages"],
        "annee": 2021,
    },
    "ips_sante2021": {
        "titre": "Enquête sur les Indicateurs des Prestations de Service de Santé EIPS 2021",
        "url": "https://www.anstat.ci/assets/projet/IPS_SANTE2021.pdf",
        "themes": ["sante", "services", "indicateurs"],
        "annee": 2021,
    },
    "ehcvm2021": {
        "titre": "Enquête Harmonisée sur les Conditions de Vie des Ménages (EHCVM 2021)",
        "url": "https://www.anstat.ci/assets/projet/ehcvm2021.pdf",
        "themes": ["conditions_de_vie", "ménages", "pauvrete"],
        "annee": 2021,
    },
    "ene_2019": {
        "titre": "Enquête Nationale sur l'Emploi (ENE) 2019",
        "url": "https://www.anstat.ci/assets/projet/ENE_2019.pdf",
        "themes": ["emploi", "chomage", "travail"],
        "annee": 2019,
    },
    "emsi_2019": {
        "titre": "Mesure sur la Société de l'Information (EMSI) 2019",
        "url": "https://www.anstat.ci/assets/projet/EMSI_2019.pdf",
        "themes": ["tic", "information", "numérique"],
        "annee": 2019,
    },
    "file_projet1719985883": {
        "titre": "Enquête Harmonisée sur les Conditions de Vie des Ménages (EHCVM 2018)",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719985883.pdf",
        "themes": ["conditions_de_vie", "ménages", "pauvrete"],
        "annee": 2018,
    },
    "eri-esi_2017": {
        "titre": "Enquête Régionale Intégrée sur l'Emploi et le Secteur Informel (ERI-ESI 2017-2018)",
        "url": "https://www.anstat.ci/assets/projet/ERI-ESI_2017.pdf",
        "themes": ["emploi", "secteur_informel", "travail"],
        "annee": 2017,
    },
    "emsi_2017": {
        "titre": "Enquête de Mesure de la Société de l'Information (EMSI) 2017",
        "url": "https://www.anstat.ci/assets/projet/EMSI_2017.pdf",
        "themes": ["tic", "information", "numérique"],
        "annee": 2017,
    },
    "file_projet1719989361": {
        "titre": "Enquête à Indicateurs Multiples (MICS) 2016",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719989361.pdf",
        "themes": ["demographie", "sante", "enfants", "education"],
        "annee": 2016,
    },
    "file_projet1719988541": {
        "titre": "Enquête Nationale sur la Situation de l'Emploi et le Secteur Informel (ENSESI) 2016",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719988541.pdf",
        "themes": ["emploi", "secteur_informel", "travail"],
        "annee": 2016,
    },
    "file_projet1719990286": {
        "titre": "Suivi de la Saison Agricole et de la Vulnérabilité Alimentaire (SAVA) 2016",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719990286.pdf",
        "themes": ["alimentation", "agriculture", "vulnerabilite"],
        "annee": 2016,
    },
    "ebcs2016": {
        "titre": "Enquête de Base Cantines Scolaires (EBCS) 2016",
        "url": "https://www.anstat.ci/assets/projet/EBCS2016.pdf",
        "themes": ["education", "alimentation", "cantines"],
        "annee": 2016,
    },
    "file_projet1719985733": {
        "titre": "Enquête sur le Niveau de Vie des Ménages (ENV) 2015",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719985733.pdf",
        "themes": ["conditions_de_vie", "pauvrete", "ménages"],
        "annee": 2015,
    },
    "rgph_2014": {
        "titre": "Recensement Général de la Population et de l'Habitat (RGPH) 2014",
        "url": "https://www.anstat.ci/assets/projet/RGPH_2014.pdf",
        "themes": ["demographie", "population", "habitat"],
        "annee": 2014,
    },
    "file_projet1719987881": {
        "titre": "Enquête de Couverture Vaccinale (ECV) 2013",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719987881.pdf",
        "themes": ["sante", "vaccination", "enfants"],
        "annee": 2013,
    },
    "papc2013": {
        "titre": "Étude de Référence pour la Phase Additionnelle du Projet d'Assistance Post-Conflit (PAPC) 2013",
        "url": "https://www.anstat.ci/assets/projet/PAPC2013.pdf",
        "themes": ["conditions_de_vie", "post_conflit", "social"],
        "annee": 2013,
    },
    "ipc_ouest&nord-ouest2013": {
        "titre": "Enquête sur l'Insécurité Alimentaire Aiguë et la Vulnérabilité à l'Ouest et au Nord-Ouest 2013",
        "url": "https://www.anstat.ci/assets/projet/IPC_Ouest&Nord-ouest2013.pdf",
        "themes": ["alimentation", "securite_alimentaire", "vulnerabilite"],
        "annee": 2013,
    },
    "file_projet1719989090": {
        "titre": "Enquête Nationale sur la Situation de l'Emploi et le Travail des Enfants (ENSETE) 2013",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719989090.pdf",
        "themes": ["emploi", "enfants", "travail"],
        "annee": 2013,
    },
    "file_projet1719988120": {
        "titre": "Enquête sur les Envois de Fonds des Travailleurs Migrants (EEFTM) 2012",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719988120.pdf",
        "themes": ["migration", "transferts_fonds", "emploi"],
        "annee": 2012,
    },
    "esasu2012": {
        "titre": "Évaluation de la Sécurité Alimentaire en Situation d'Urgence 2012",
        "url": "https://www.anstat.ci/assets/projet/ESASU2012.pdf",
        "themes": ["alimentation", "securite_alimentaire", "urgence"],
        "annee": 2012,
    },
    "esao2012": {
        "titre": "Évaluation de la Sécurité Alimentaire à l'Ouest de la Côte d'Ivoire 2012",
        "url": "https://www.anstat.ci/assets/projet/ESAO2012.pdf",
        "themes": ["alimentation", "securite_alimentaire", "region"],
        "annee": 2012,
    },
    "file_projet1719990011": {
        "titre": "Enquête Post Distribution Monitoring (PDM) 2012",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719990011.pdf",
        "themes": ["alimentation", "aide_humanitaire", "distribution"],
        "annee": 2012,
    },
    "eds2011-2012": {
        "titre": "Enquête Démographique et de Santé 2011-2012",
        "url": "https://www.anstat.ci/assets/projet/EDS2011-2012.pdf",
        "themes": ["demographie", "sante", "ménages"],
        "annee": 2012,
    },
    "file_projet1719990636": {
        "titre": "Enquête Post Distribution Monitoring (PDM) 2011",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719990636.pdf",
        "themes": ["alimentation", "aide_humanitaire", "distribution"],
        "annee": 2011,
    },
    "fsms_2010": {
        "titre": "Système de Suivi de la Sécurité Alimentaire 2010",
        "url": "https://www.anstat.ci/assets/projet/FSMS_2010.pdf",
        "themes": ["alimentation", "securite_alimentaire", "suivi"],
        "annee": 2010,
    },
    "easa2009": {
        "titre": "Évaluation Approfondie de la Sécurité Alimentaire des Ménages Ruraux en Côte d'Ivoire 2009",
        "url": "https://www.anstat.ci/assets/projet/EASA2009.pdf",
        "themes": ["alimentation", "securite_alimentaire", "ménages", "rural"],
        "annee": 2009,
    },
    "env2008": {
        "titre": "Enquête sur le Niveau de Vie des Ménages 2008",
        "url": "https://www.anstat.ci/assets/projet/ENV2008.pdf",
        "themes": ["conditions_de_vie", "pauvrete", "ménages"],
        "annee": 2008,
    },
    "file_projet1719986438": {
        "titre": "Enquête Qualité sur l'Évolution du Niveau de Vie en Côte d'Ivoire 2002",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719986438.pdf",
        "themes": ["conditions_de_vie", "pauvrete", "ménages"],
        "annee": 2002,
    },
    "rgph_98tome1": {
        "titre": "Recensement Général de la Population et de l'Habitat (RGPH) 1998",
        "url": "https://www.anstat.ci/assets/projet/RGPH_98TOME1.pdf",
        "themes": ["demographie", "population", "habitat"],
        "annee": 1998,
    },
    "eds1998": {
        "titre": "Enquête Démographique et de Santé 1998-1999",
        "url": "https://www.anstat.ci/assets/projet/EDS1998.pdf",
        "themes": ["demographie", "sante", "ménages"],
        "annee": 1998,
    },
    "file_projet1719987012": {
        "titre": "Enquête Qualité sur l'Évolution du Niveau de Vie en Côte d'Ivoire 1998",
        "url": "https://www.anstat.ci/assets/projet/File_projet1719987012.pdf",
        "themes": ["conditions_de_vie", "pauvrete", "ménages"],
        "annee": 1998,
    },
    "env1995": {
        "titre": "Enquête Qualité sur l'Évolution du Niveau de Vie en Côte d'Ivoire 1995",
        "url": "https://www.anstat.ci/assets/projet/ENV1995.pdf",
        "themes": ["conditions_de_vie", "pauvrete", "ménages"],
        "annee": 1995,
    },
    "eds1994": {
        "titre": "Enquête Démographique et de Santé 1994",
        "url": "https://www.anstat.ci/assets/projet/EDS1994.pdf",
        "themes": ["demographie", "sante", "ménages"],
        "annee": 1994,
    },
    "file_projet1720002062": {
        "titre": "Profil de Pauvreté en Côte d'Ivoire 1993 et 1995",
        "url": "https://www.anstat.ci/assets/projet/File_projet1720002062.pdf",
        "themes": ["pauvrete", "conditions_de_vie", "social"],
        "annee": 1995,
    },
    "epam1986": {
        "titre": "Enquête Permanente auprès des Ménages 1986",
        "url": "https://www.anstat.ci/assets/projet/EPAM1986.pdf",
        "themes": ["conditions_de_vie", "ménages", "pauvrete"],
        "annee": 1986,
    },
    "epam1985": {
        "titre": "Enquête Permanente auprès des Ménages 1985",
        "url": "https://www.anstat.ci/assets/projet/EPAM1985.pdf",
        "themes": ["conditions_de_vie", "ménages", "pauvrete"],
        "annee": 1985,
    },
    # ── NOUVEAUX DOCUMENTS — Indices et indicateurs ─────────────────────────
    "file_val_indicateur1772864957": {
        "titre": "Indice du Chiffre d'Affaires dans l'Industrie",
        "url": "https://www.anstat.ci/assets/publications/files/File_val_indicateur1772864957.pdf",
        "themes": ["industrie", "chiffre_affaires", "indicateurs", "economie"],
        "annee": 2025,
    },
    "file_val_indicateur1772507808": {
        "titre": "Indice du Chiffre d'Affaires dans le Commerce",
        "url": "https://www.anstat.ci/assets/publications/files/File_val_indicateur1772507808.pdf",
        "themes": ["commerce", "chiffre_affaires", "indicateurs", "economie"],
        "annee": 2025,
    },
    "file_val_indicateur1772123415": {
        "titre": "Indice de Chiffre d'Affaires dans les Services",
        "url": "https://www.anstat.ci/assets/publications/files/File_val_indicateur1772123415.pdf",
        "themes": ["services", "chiffre_affaires", "indicateurs", "economie"],
        "annee": 2025,
    },
    "file_val_indicateur1772121364": {
        "titre": "Indice des Prix de Production dans l'Industrie",
        "url": "https://www.anstat.ci/assets/publications/files/File_val_indicateur1772121364.pdf",
        "themes": ["industrie", "prix_production", "indicateurs", "economie"],
        "annee": 2025,
    },
    "file_val_indicateur1771950566": {
        "titre": "Indice de Production dans l'Industrie",
        "url": "https://www.anstat.ci/assets/publications/files/File_val_indicateur1771950566.pdf",
        "themes": ["industrie", "production", "indicateurs", "economie"],
        "annee": 2025,
    },
    "file_val_indicateur1771927909": {
        "titre": "Indicateur Avancé des BTP",
        "url": "https://www.anstat.ci/assets/publications/files/File_val_indicateur1771927909.pdf",
        "themes": ["btp", "construction", "indicateurs", "economie"],
        "annee": 2025,
    },
    "file_val_indicateur1762328869": {
        "titre": "Chiffres Clés de la Côte d'Ivoire",
        "url": "https://www.anstat.ci/assets/publications/files/File_val_indicateur1762328869.pdf",
        "themes": ["economie", "indicateurs", "statistiques", "synthese"],
        "annee": 2025,
    },
}

# ── Mapping nom fichier original → nom safe (OpenDataLoader) ───────────────
PDF_NAME_MAPPING = {
    "Demographic and health survey 2021(RAPPORT EDS 2021).pdf":                                          "eds2021.pdf",
    "Enquête de base cantines scolaires (RAPPORT EBCS 2016).pdf":                                        "ebcs2016.pdf",
    "Enquête de couverture vaccinale ( RAPPORT ECV 2013).pdf":                                           "ecv2013.pdf",
    "Enquête de Mesure de la Société de l'Information ( RAPPORT EMSI) 2017.pdf":                         "emsi2017.pdf",
    "Enquête démographique et de santé 1994 ( RAPPORT EDS 1994).pdf":                                    "eds1994.pdf",
    "Enquête démographique et de santé 1998-1999 ( RAPPORT EDS 1998 ).pdf":                              "eds1998.pdf",
    "Enquête démographique et de santé 2011-2012.pdf":                                                   "eds2011.pdf",
    "Enquête harmonisée sur les conditions de vie des ménages ( RAPPORT EHCVM 2018).pdf":                "ehcvm2018.pdf",
    "Enquête harmonisée sur les conditions de vie des ménages (EHCVM 2021).pdf":                         "ehcvm2021.pdf",
    "Enquête nationale sur l'emploi (ENE) 2019 (RAPPORT ENE 2019).pdf":                                  "ene2019.pdf",
    "Enquête nationale sur la situation de l'emploi et le secteur informel ( RAPPORT ENSESI 2016).pdf":  "ensesi2016.pdf",
    "ENQUÊTE NATIONALE SUR LA SITUATION DE L'EMPLOI ET LE TRAVAIL DES ENFANTS ( RAPPORT ENSETE 2013).pdf": "ensete2013.pdf",
    "Enquête permanente auprès des ménages 1985 (RAPPORT EPAM 1985).pdf":                                "epam1985.pdf",
    "Enquête permanente auprès des ménages 1986 ( RAPPORT EPAM 1986).pdf":                               "epam1986.pdf",
    "ENQUÊTE POST DISTRIBUTION MONITORING ( RAPPORT PDM) 2011.pdf":                                      "pdm2011.pdf",
    "ENQUÊTE POST DISTRIBUTION MONITORING (PDM) 2012.pdf":                                               "pdm2012.pdf",
    "Enquête qualité sur l'évolution du niveau de vie en Côte d'Ivoire 1995(RAPPORT ENV 1995).pdf":      "env1995.pdf",
    "Enquête qualité sur l'évolution du niveau de vie en Côte d'Ivoire 1998 (RAPPORT ENV 1998).pdf":     "env1998.pdf",
    "Enquête qualité sur l'évolution du niveau de vie en Côte d'Ivoire 2002 (RAPPORT ENV 2002).pdf":     "env2002.pdf",
    "Enquête régionale intégrée sur l'emploi et le secteur informel ( RAPPORT ERI-ESI, 2017-2018).pdf":  "eri_esi2017.pdf",
    "Enquête sur le niveau de vie des ménages ( RAPPORT ENV 2015).pdf":                                  "env2015.pdf",
    "Enquête sur le niveau de vie des ménages 2008 ( RAPPORT ENV 2008).pdf":                             "env2008.pdf",
    "Enquête sur les envois de fonds des travailleurs migrants ( RAPPORT EEFTM 2012).pdf":               "eeftm2012.pdf",
    "Enquête sur les indicateurs des prestations de service de santé EIPS 2021(RAPPORT IPS SANTE 2021).pdf": "ips_sante2021.pdf",
    "Enquête sur les indicateurs des prestations de services de l'éducation EIPSE 2022(RAPPORT IPS EDUCATION 2022).pdf": "ips_education2022.pdf",
    "ENQUÊTE SUR L'ACCÈS ET L'UTILISATION DES TIC PAR LES MÉNAGES EN COTE D'IVOIRE (DATAFID).pdf":      "datafid.pdf",
    "Enquête sur l'Insécurité Alimentaire Aiguë et la Vulnérabilité à l'Ouest et au Nord-Ouest 2013 (RAPPORT IPC 2013).pdf": "ipc2013.pdf",
    "Enquête à indicateurs multiples ( RAPPORT MICS 2016).pdf":                                          "mics2016.pdf",
    "Etude de référence pour la phase additionnelle du projet d'assistance post-conflit ( RAPPORT PAPC 2013).pdf": "papc2013.pdf",
    "Evaluation approfondie de la sécurité alimentaire des ménages ruraux en Côte d'Ivoire 2009 (RAPPORT EASA 2009).pdf": "easa2009.pdf",
    "Evaluation de la sécurité alimentaire en situation d'urgence 2012(RAPPORT ESASU 2012).pdf":         "esasu2012.pdf",
    "Evaluation de la sécurité alimentaire à l'Ouest de la Côte d'Ivoire 2012(RAPPORT ESAO 2012).pdf":  "esao2012.pdf",
    "Mesure sur la société de l'information (RAPPORT EMSI 2019).pdf":                                    "emsi2019.pdf",
    "PROFIL DE PAUVRETÉ EN COTE D'IVOIRE 1993 ET 1995 (RAPPORT PPC 1993 - 1995).pdf":                   "ppc1993.pdf",
    "RAPPORT SIGI 2022.pdf":                                                                              "sigi2022.pdf",
    "Recensement général de la population et de l'habitat (RGPH) 1998.pdf":                              "rgph1998.pdf",
    "Recensement général de la population et de l'habitat (RAPPORT RGPH 2014).pdf":                      "rgph2014.pdf",
    "Suivi de la saison agricole et de la vulnérabilité alimentaire ( RAPPORT SAVA 2016).pdf":           "sava2016.pdf",
    "Système de suivi de la sécurité alimentaire 2010 ( RAPPORT FSMS 2010).pdf":                         "fsms2010.pdf",
    # Nouveaux indicateurs
    "Indice du Chiffre d'Affaires dans l'Industrie.pdf": "ica_industrie.pdf",
    "Indice du Chiffre d'Affaires dans le Commerce.pdf": "ica_commerce.pdf",
    "Indice du Chiffre d'Affaires dans les Services.pdf": "ica_services.pdf",
    "Indice de la Productivité dans l'Industrie.pdf": "ipp_industrie.pdf",
    "Indice de l'Activité dans l'Industrie.pdf": "ip_industrie.pdf",
    "Indice de l'Activité dans le Bâtiment et l'Installation.pdf": "ia_btp.pdf",
    "Chiffres Clés pour la Côte d'Ivoire.pdf": "chiffres_cles_ci.pdf",
}

# ============================================================================
# MOTS-CLÉS THÉMATIQUES ENRICHIS
# ============================================================================

THEME_KEYWORDS = {
    "alimentation": [
        "alimentation", "nourriture", "sécurité alimentaire", "famine", "nutrition",
        "faim", "calorie", "ration", "consommation alimentaire", "insécurité alimentaire",
        "agriculture", "récolte", "céréale", "vivrier", "cantine", "distribut",
    ],
    "demographie": [
        "population", "recensement", "natalité", "mortalité", "fécondité",
        "naissance", "décès", "migration", "démographique", "ménage", "habitants",
        "croissance démographique", "répartition", "âge", "sexe", "genre",
    ],
    "pauvrete": [
        "pauvreté", "pauvre", "indigent", "vulnérable", "inégalité", "revenu",
        "quintile", "seuil de pauvreté", "dépense", "consommation", "niveau de vie",
        "profil de pauvreté", "incidence", "déficit", "exclusion",
    ],
    "emploi": [
        "emploi", "chômage", "travail", "actif", "inactif", "secteur informel",
        "salarié", "indépendant", "profession", "main-d'œuvre", "marché du travail",
        "taux d'activité", "sous-emploi", "salaire", "rémunération", "travail des enfants",
    ],
    "education": [
        "éducation", "scolarisation", "école", "alphabétisation", "élève",
        "enseignant", "classe", "primaire", "secondaire", "université", "formation",
        "taux de scolarisation", "abandon scolaire", "cantine scolaire", "diplôme",
    ],
    "sante": [
        "santé", "maladie", "mortalité infantile", "vaccination", "maternité",
        "hôpital", "médecin", "couverture sanitaire", "paludisme", "VIH", "SIDA",
        "accouchement", "nutrition", "malnutrition", "prévalence", "morbidité",
    ],
    "tic": [
        "TIC", "internet", "téléphone", "mobile", "numérique", "informatique",
        "connectivité", "accès internet", "technologie", "communication", "réseau",
        "société de l'information", "smartphone", "ordinateur", "données",
    ],
    "conditions_de_vie": [
        "niveau de vie", "conditions de vie", "logement", "habitat", "eau potable",
        "électricité", "assainissement", "bien-être", "dépenses ménages",
        "accès aux services", "infrastructure", "cadre de vie",
    ],
    "social": [
        "social", "genre", "inégalité", "discrimination", "femme", "enfant",
        "protection sociale", "cohésion", "communauté", "ménage", "famille",
    ],
    # Nouveaux thèmes pour indicateurs économiques
    "economie": [
        "PIB", "croissance", "économie", "économique", "produit intérieur brut",
        "investissement", "exportation", "importation", "balance commerciale",
        "secteur", "activité économique", "conjoncture", "cycle économique",
    ],
    "industrie": [
        "industrie", "industriel", "production industrielle", "manufacture",
        "usine", "fabrication", "transformation", "secteur secondaire",
        "indice de production", "capacité de production",
    ],
    "commerce": [
        "commerce", "commercial", "vente", "chiffre d'affaires", "distribution",
        "marché", "achat", "échange", "transaction", "retail", "gros",
    ],
    "services": [
        "services", "prestation", "service", "tertiaire", "secteur tertiaire",
        "télécommunication", "banque", "assurance", "transport", "tourisme",
    ],
    "btp": [
        "BTP", "construction", "bâtiment", "travaux publics", "logement",
        "infrastructure", "immobilier", "permis de construire", "chantier",
    ],
    "indicateurs": [
        "indice", "indicateur", "taux", "ratio", "pourcentage", "statistique",
        "mesure", "variation", "évolution", "tendance", "glissement annuel",
        "base 100", "pondération", "ICA", "IPI", "IPP",
    ],
}

# ============================================================================
# ███ BLOC 1 — NETTOYAGE TEXTE (COUCHES C0, C4) ███
# ============================================================================

ENCODING_GARBAGE_PATTERNS = [
    (re.compile(r'Ã©'), 'é'), (re.compile(r'Ã¨'), 'è'),
    (re.compile(r'Ã '), 'à'), (re.compile(r'Ã®'), 'î'),
    (re.compile(r'Ã´'), 'ô'), (re.compile(r'Ã»'), 'û'),
    (re.compile(r'Ã§'), 'ç'), (re.compile(r'â€™'), "'"),
    (re.compile(r'â€œ'), '"'), (re.compile(r'â€\x9d'), '"'),
    (re.compile(r'â€"'), '–'), (re.compile(r'â€¦'), '…'),
    (re.compile(r'Â«'), '«'), (re.compile(r'Â»'), '»'),
    (re.compile(r'Â '), ' '), (re.compile(r'\ufffd'), ''),
    (re.compile(r'\x00'), ''),
]

RE_OCR_ISOLATED_CHARS = re.compile(
    r'(?<!\w)([|\\/<>~^`°±§¶†‡•])(?!\w)'
)
RE_GARBAGE_SEQUENCES = re.compile(r'[^\x00-\x7F\xc0-\xff\s]{4,}')
RE_PAGE_NUMBER_ONLY  = re.compile(r'^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$')
RE_HEADER_FOOTER_INST = re.compile(
    r'^.{0,80}(anstat|côte\s+d.ivoire|ministère|république|rapport\s+final|'
    r'enquête\s+nationale|recensement\s+général|direction\s+générale).{0,80}$',
    re.IGNORECASE
)
RE_DECORATIVE_SEP = re.compile(r'^[\s\-–_=*#~]{3,}\s*$')
WATERMARK_PATTERNS = [
    re.compile(r'\b(draft|brouillon|confidentiel|specimen|sample|usage\s+interne)\b',
               re.IGNORECASE),
]


def clean_raw_text(text: str) -> str:
    if not text:
        return ""
    for pattern, replacement in ENCODING_GARBAGE_PATTERNS:
        text = pattern.sub(replacement, text)
    text = unicodedata.normalize("NFC", text)
    text = RE_GARBAGE_SEQUENCES.sub(' ', text)
    text = RE_OCR_ISOLATED_CHARS.sub(' ', text)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = re.sub(r'[ \t]+', ' ', line).strip()
        if RE_DECORATIVE_SEP.match(line):
            continue
        cleaned.append(line)
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    for pat in WATERMARK_PATTERNS:
        text = pat.sub('', text)
    return text.strip()


def remove_header_footer_lines(text: str) -> str:
    lines   = text.split('\n')
    cleaned = []
    for line in lines:
        s = line.strip()
        if RE_PAGE_NUMBER_ONLY.match(s):
            continue
        if len(s) < 120 and RE_HEADER_FOOTER_INST.match(s):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def detect_repeated_headers_footers(pages_data: List[Dict],
                                    threshold: float = 0.55) -> set:
    if len(pages_data) < 3:
        return set()
    counter     = Counter()
    total_pages = len(pages_data)
    for page in pages_data:
        lines = [l.strip() for l in page["text"].split('\n')
                 if l.strip() and len(l.strip()) > 5]
        for line in lines[:3]:
            if len(line) < 150:
                counter[line] += 1
        for line in lines[-3:]:
            if len(line) < 150:
                counter[line] += 1
    return {l for l, cnt in counter.items()
            if cnt / total_pages >= threshold and len(l) > 5}


def remove_repeated_lines(text: str, repeated: set) -> str:
    if not repeated:
        return text
    return '\n'.join(l for l in text.split('\n') if l.strip() not in repeated)

# ============================================================================
# ███ BLOC 2 — FILTRES STRUCTURELS (C1, C2, C5, C6, C8) ███
# IMPORTANT : C3 et C7 CORRIGÉS — préservent les données chiffrées statistiques
# ============================================================================

STRUCTURAL_SECTION_KEYWORDS = [
    "table des matières", "tables des matières", "table des matieres",
    "sommaire", "table of contents", "contents",
    "liste des tableaux", "liste des figures", "liste des graphiques",
    "liste des cartes", "liste des encadrés",
    "liste des abréviations", "liste des sigles", "liste des acronymes",
    "abréviations et sigles", "sigles et abréviations",
    "annexe", "annexes", "appendice", "appendix",
    "bibliographie", "références bibliographiques",
    "remerciements", "avant-propos", "preface", "préface",
    "dédicace", "avertissement",
    "droits d'auteur", "copyright", "mentions légales",
    "tous droits réservés",
    "questionnaire", "guide d'entretien", "protocole de collecte",
    "nomenclature", "classification", "codification",
    "glossaire", "lexique",
    "index alphabétique",
]

RE_TOC_LINE     = re.compile(
    r'^(?:[\d]+(?:\.\d+)*\s+)?.{3,100}[\.\s·•\-]{3,}\d{1,4}\s*$',
    re.MULTILINE
)
RE_TOC_DOT_LINE = re.compile(r'^[\.\s\-·•]{5,}\s*\d{1,4}\s*$', re.MULTILINE)

RE_FIGURE_CAPTION = re.compile(
    r'^(?:figure|tableau|graphique|carte|encadré|photo|image|schéma|diagramme)\s*'
    r'\d*\s*[:\-–]?\s*.{0,120}$', re.IGNORECASE
)
RE_FOOTNOTE    = re.compile(r'^[\d\*†‡§]\s+.{5,150}$')
RE_SOURCE_NOTE = re.compile(
    r'^(?:source|note|notes?|n\.b\.?|avertissement)\s*[:\-–]?\s*.{0,200}$',
    re.IGNORECASE
)
RE_INTERNAL_REF = re.compile(
    r'(?:voir|cf\.?|se\s+référer\s+à|consulter|référence\s*:?|source\s*:)\s*'
    r'(?:tableau|figure|graphique|carte|encadré|annexe|chapitre|section|page)\s*[\w\d\.]+',
    re.IGNORECASE
)

EXECUTIVE_SUMMARY_PATTERNS = [
    re.compile(r'^résumé\s*(exécutif|analytique|synthétique)?', re.IGNORECASE),
    re.compile(r'^summary',             re.IGNORECASE),
    re.compile(r'^synthèse',            re.IGNORECASE),
    re.compile(r'^points\s+clés',       re.IGNORECASE),
    re.compile(r'^faits\s+saillants',   re.IGNORECASE),
    re.compile(r'^messages\s+clés',     re.IGNORECASE),
    re.compile(r'^principales\s+conclusions',    re.IGNORECASE),
    re.compile(r'^principales\s+constatations',  re.IGNORECASE),
]

QUESTIONNAIRE_PATTERNS = [
    re.compile(r'^\s*Q\d{1,3}[a-z]?\s*[:\.\-]', re.IGNORECASE),
    re.compile(r'^\s*[A-Z]{2,5}\d{1,3}\s*[:\.\-]'),
    re.compile(r'facteur\s+d.expansion',     re.IGNORECASE),
    re.compile(r'pondération\s+finale',      re.IGNORECASE),
    re.compile(r'poids\s+de\s+sondage',      re.IGNORECASE),
    re.compile(r'strate\s+d.échantillonnage', re.IGNORECASE),
    re.compile(r'code\s+géographique\s*:',   re.IGNORECASE),
    re.compile(r'nomenclature\s+des\s+',     re.IGNORECASE),
    re.compile(r'variable\s+\w+\s*:\s*\d+',  re.IGNORECASE),
    re.compile(r'modalité\s*\d+\s*[:\-]',    re.IGNORECASE),
]


def is_structural_noise_page(text: str) -> Tuple[bool, str]:
    if not text:
        return True, "page vide"
    text_lower  = text.lower().strip()
    first_lines = "\n".join(text_lower.split("\n")[:5])
    for kw in STRUCTURAL_SECTION_KEYWORDS:
        if kw in first_lines:
            return True, f"section structurelle: '{kw}'"
    lines     = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True, "page sans contenu"
    toc_count = sum(1 for l in lines
                    if RE_TOC_LINE.match(l) or RE_TOC_DOT_LINE.match(l))
    if toc_count / len(lines) >= 0.35:
        return True, f"page TOC ({toc_count/len(lines):.0%})"
    if len(text_lower) < 300:
        alpha = [c for c in text if c.isalpha()]
        if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.72:
            return True, "page de garde (majuscules > 72%)"
    return False, ""


def is_questionnaire_chunk(text: str) -> Tuple[bool, str]:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False, ""
    match_count = sum(
        1 for line in lines
        if any(pat.search(line) for pat in QUESTIONNAIRE_PATTERNS)
    )
    ratio = match_count / len(lines)
    if ratio >= 0.45:
        return True, f"questionnaire ({ratio:.0%})"
    return False, ""


def is_executive_summary_chunk(text: str) -> bool:
    first = "\n".join(text.strip().split('\n')[:4]).lower()
    return any(pat.search(first) for pat in EXECUTIVE_SUMMARY_PATTERNS)


def clean_chunk_intra(text: str) -> str:
    """C5 — Supprime légendes, notes de bas de page, renvois internes."""
    lines   = text.split('\n')
    cleaned = []
    for line in lines:
        s = line.strip()
        if RE_FIGURE_CAPTION.match(s) and len(s) < 130:
            continue
        if RE_FOOTNOTE.match(s) and len(s) < 160:
            continue
        if RE_SOURCE_NOTE.match(s) and len(s) < 210:
            continue
        line_c = RE_INTERNAL_REF.sub('', line)
        if line_c.strip():
            cleaned.append(line_c)
    return '\n'.join(cleaned).strip()


# ── C3/C7 CORRIGÉS : tableaux et données chiffrées PRÉSERVÉS ───────────────

def is_statistical_table_text(text: str) -> bool:
    """
    Détecte si le texte est issu d'un tableau statistique.
    Ces chunks sont PRÉCIEUX et doivent être CONSERVÉS, pas supprimés.
    Critères : présence de données numériques + headers ou labels de colonnes.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False
    # Présence de séparateurs de tableau |
    pipe_lines = sum(1 for l in lines if '|' in l)
    if pipe_lines / len(lines) > 0.3:
        return True
    # Beaucoup de nombres mais aussi des labels textuels → tableau stats
    num_lines  = sum(1 for l in lines if re.search(r'\d+[,\.]\d+', l))
    text_lines = sum(1 for l in lines if sum(1 for c in l if c.isalpha()) > 5)
    if num_lines > 2 and text_lines > 1:
        return True
    return False


def is_true_junk_chunk(text: str) -> Tuple[bool, str]:
    """
    C3 + C7 CORRIGÉS :
    Supprime UNIQUEMENT les vrais chunks sans valeur :
    - Listes de chiffres purs sans aucun label (ex : pages de codebook)
    - Chunks trop courts pour avoir du sens
    - Chunks avec quasi aucune lettre

    CONSERVE :
    - Tableaux statistiques avec chiffres + labels
    - Données chiffrées avec contexte textuel
    """
    if not text or not text.strip():
        return True, "vide"
    words = text.split()
    if len(words) < 6:
        return True, f"trop court ({len(words)} mots)"

    # Si c'est un tableau statistique → toujours garder
    if is_statistical_table_text(text):
        return False, ""

    # Quasi aucune lettre (page de codes purs, séquences binaires, etc.)
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.15:
        return True, f"quasi aucune lettre ({alpha_ratio:.0%})"

    # Vocabulaire vraiment pauvre ET pas de chiffres utiles
    unique_words = {w.lower() for w in words if len(w) > 2}
    if len(unique_words) < 4 and alpha_ratio < 0.25:
        return True, "vocabulaire nul sans données"

    return False, ""

# ============================================================================
# ███ BLOC 3 — DÉDUPLICATION SÉMANTIQUE (C9) ███
# ============================================================================

def build_shingle_set(text: str, n: int = 5) -> set:
    words = re.sub(r'\s+', ' ', text.lower()).split()
    if len(words) < n:
        return {text.lower()}
    return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def deduplicate_chunks(chunks: List[Dict],
                       threshold: float = 0.80) -> Tuple[List[Dict], int]:
    unique, seen_shingles, removed = [], [], 0
    for chunk in chunks:
        shingles = build_shingle_set(chunk.get("text", ""))
        if any(jaccard_similarity(shingles, s) >= threshold
               for s in seen_shingles):
            removed += 1
        else:
            unique.append(chunk)
            seen_shingles.append(shingles)
    return unique, removed

# ============================================================================
# ███ BLOC 4 — EXTRACTION TABLEAUX (pdfplumber + Camelot) ███
# ============================================================================

def table_df_to_text(df: pd.DataFrame, table_title: str = "",
                     doc_info: Dict = None) -> str:
    """
    Convertit un DataFrame de tableau en texte riche et lisible par le LLM.
    Format inspiré du tableau de l'image envoyée (orange header, données chiffrées).

    Résultat :
    [TABLEAU: Titre — Document, Page N]
    Colonne1 | Colonne2 | Colonne3
    Valeur1  | Valeur2  | Valeur3
    ...
    Résumé: Ce tableau présente X lignes de données sur Y.
    """
    if df is None or df.empty:
        return ""

    # Nettoyer le DataFrame
    df = df.fillna("").astype(str)
    df = df.replace(r'^\s*$', '', regex=True)

    lines = []

    # En-tête du tableau
    doc_titre = doc_info.get("titre", "") if doc_info else ""
    if table_title and doc_titre:
        lines.append(f"[TABLEAU: {table_title} — {doc_titre}]")
    elif table_title:
        lines.append(f"[TABLEAU: {table_title}]")
    elif doc_titre:
        lines.append(f"[TABLEAU: {doc_titre}]")
    else:
        lines.append("[TABLEAU: Données statistiques]")

    # En-têtes de colonnes
    headers = [str(col).strip() for col in df.columns if str(col).strip()]
    if headers:
        lines.append(" | ".join(headers))
        lines.append("-" * min(len(" | ".join(headers)), 80))

    # Lignes de données
    for _, row in df.iterrows():
        cells = [str(v).strip() for v in row.values]
        # Filtrer les lignes complètement vides
        if any(c for c in cells):
            lines.append(" | ".join(cells))

    # Résumé automatique
    n_rows = len(df)
    n_cols = len(df.columns)
    lines.append(f"[Ce tableau contient {n_rows} lignes et {n_cols} colonnes de données.]")
    lines.append("[FIN TABLEAU]")

    return "\n".join(lines)


def extract_tables_pdfplumber(pdf_path: Path,
                              doc_info: Dict = None) -> Dict[int, List[str]]:
    """
    Extraction tableaux via pdfplumber.
    Retourne dict {page_number: [texte_tableau_1, texte_tableau_2, ...]}
    Idéal pour tableaux sans bordures visibles (stream mode).
    """
    result: Dict[int, List[str]] = defaultdict(list)
    if not PDFPLUMBER_AVAILABLE:
        return result
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if not tables:
                    continue
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    # Première ligne = headers
                    headers = [str(h).strip() if h else f"Col{i}"
                               for i, h in enumerate(table[0])]
                    rows    = table[1:]
                    try:
                        df = pd.DataFrame(rows, columns=headers)
                        text = table_df_to_text(df, doc_info=doc_info)
                        if text and len(text) > 30:
                            result[page_num].append(text)
                    except Exception:
                        pass
    except Exception as e:
        pass
    return result


def extract_tables_camelot(pdf_path: Path,
                           doc_info: Dict = None) -> Dict[int, List[str]]:
    """
    Extraction tableaux via Camelot.
    Essaie lattice (bordures) puis stream (sans bordures).
    Les WARNING Java/Ghostscript sont redirigés (inoffensifs).
    Retourne dict {page_number: [texte_tableau, ...]}
    """
    import sys, io, os
    result: Dict[int, List[str]] = defaultdict(list)
    if not CAMELOT_AVAILABLE:
        return result
    # Supprimer les warnings Java/Ghostscript de Camelot (inoffensifs)
    _old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        # Méthode 1 : lattice (tableaux avec lignes de bordure)
        try:
            tables = camelot.read_pdf(str(pdf_path), pages='all',
                                      flavor='lattice', suppress_stdout=True)
            for table in tables:
                if table.parsing_report.get('accuracy', 0) < 50:
                    continue
                df   = table.df
                page = table.parsing_report.get('page', 1)
                text = table_df_to_text(df, doc_info=doc_info)
                if text and len(text) > 30:
                    result[page].append(text)
        except Exception:
            pass

        # Méthode 2 : stream (tableaux sans bordures)
        if not result:
            try:
                tables = camelot.read_pdf(str(pdf_path), pages='all',
                                          flavor='stream', suppress_stdout=True)
                for table in tables:
                    if table.parsing_report.get('accuracy', 0) < 40:
                        continue
                    df   = table.df
                    page = table.parsing_report.get('page', 1)
                    text = table_df_to_text(df, doc_info=doc_info)
                    if text and len(text) > 30:
                        result[page].append(text)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        sys.stderr = _old_stderr  # Restaurer stderr
    return result


def merge_table_sources(pdfplumber_tables: Dict[int, List[str]],
                        camelot_tables:    Dict[int, List[str]]) -> Dict[int, List[str]]:
    """
    Fusionne les tableaux de pdfplumber et Camelot.
    Préfère Camelot (plus précis) quand les deux ont des résultats.
    Déduplique par contenu similaire.
    """
    merged: Dict[int, List[str]] = defaultdict(list)
    all_pages = set(pdfplumber_tables.keys()) | set(camelot_tables.keys())

    for page in all_pages:
        camelot_res   = camelot_tables.get(page, [])
        pdfplumber_res = pdfplumber_tables.get(page, [])

        if camelot_res:
            # Camelot prioritaire
            merged[page].extend(camelot_res)
            # Ajouter pdfplumber s'il trouve des tables différentes
            for pt in pdfplumber_res:
                if not any(jaccard_similarity(
                    build_shingle_set(pt), build_shingle_set(ct)
                ) > 0.6 for ct in camelot_res):
                    merged[page].append(pt)
        else:
            merged[page].extend(pdfplumber_res)

    return merged

# ============================================================================
# ███ BLOC 5 — EXTRACTION TEXTE (OpenDataLoader + PyMuPDF fallback) ███
# ============================================================================

def _safe_pdf_path(pdf_path: Path, tmp_dir: str) -> Path:
    """Copie le PDF avec un nom safe pour OpenDataLoader."""
    original = pdf_path.name
    if original in PDF_NAME_MAPPING:
        safe_name = PDF_NAME_MAPPING[original]
    else:
        name = unicodedata.normalize("NFD", pdf_path.stem)
        name = "".join(c for c in name if unicodedata.category(c) != "Mn")
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")[:80]
        safe_name = f"{name}.pdf"
    safe_path = Path(tmp_dir) / safe_name
    shutil.copy2(str(pdf_path), str(safe_path))
    return safe_path


def _table_to_text_odl(block: Dict) -> str:
    try:
        rows = block.get("rows", block.get("data", []))
        if not rows:
            return block.get("text", "")
        lines = []
        for row in rows:
            if isinstance(row, list):
                lines.append(" | ".join(str(c) for c in row if c))
            elif isinstance(row, dict):
                lines.append(" | ".join(str(v) for v in row.values() if v))
        return "\n".join(lines)
    except Exception:
        return block.get("text", "")


def _extract_block_text(block: Dict) -> str:
    block_type = block.get("type", "")
    if block_type == "table":
        return _table_to_text_odl(block)
    text = block.get("content", block.get("text", ""))
    if not isinstance(text, str):
        text = str(text) if text else ""
    kids = block.get("kids", [])
    if kids:
        kid_texts = [_extract_block_text(k) for k in kids
                     if isinstance(k, dict)]
        kid_texts = [kt for kt in kid_texts if kt.strip()]
        if kid_texts:
            kids_str = "\n".join(kid_texts)
            text = (text + "\n" + kids_str).strip() if text.strip() else kids_str
    return text.strip()


def _parse_odl_json(odl_data: Any) -> List[Dict]:
    pages_acc: Dict[int, List[str]] = {}
    if isinstance(odl_data, dict):
        blocks = odl_data.get("kids", [])
    elif isinstance(odl_data, list):
        blocks = odl_data
    else:
        return []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        raw_page = block.get("page number",
                   block.get("page_number",
                   block.get("page", 1)))
        try:
            page_num = int(raw_page)
        except Exception:
            page_num = 1
        text = _extract_block_text(block)
        if text.strip():
            pages_acc.setdefault(page_num, []).append(text)
    result = []
    for page_num in sorted(pages_acc.keys()):
        combined = "\n\n".join(t for t in pages_acc[page_num] if t.strip())
        if combined.strip():
            result.append({
                "page_number":       page_num,
                "text":              combined,
                "char_count":        len(combined),
                "word_count":        len(combined.split()),
                "extraction_method": "opendataloader_json",
            })
    return result


def _markdown_table_to_text(md_text: str) -> str:
    lines, result = md_text.split('\n'), []
    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|'):
            if re.match(r'^\s*\|[\s\-:|]+\|\s*$', line):
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')
                     if c.strip()]
            if cells:
                result.append(' | '.join(cells))
        else:
            result.append(line)
    return '\n'.join(result)


def _parse_odl_markdown(md_content: str) -> List[Dict]:
    pages_data  = []
    sep         = re.compile(r'\n---\n|\f|\n={3,}\n|\n\f\n')
    page_blocks = sep.split(md_content)
    for i, page_text in enumerate(page_blocks, 1):
        page_text = page_text.strip()
        if not page_text:
            continue
        page_text = re.sub(r'^#{1,6}\s+', '', page_text, flags=re.MULTILINE)
        page_text = _markdown_table_to_text(page_text)
        if page_text.strip():
            pages_data.append({
                "page_number":       i,
                "text":              page_text,
                "char_count":        len(page_text),
                "word_count":        len(page_text.split()),
                "extraction_method": "opendataloader_markdown",
            })
    return pages_data


def extract_with_opendataloader(pdf_path: Path) -> Tuple[List[Dict], str, bool]:
    if not OPENDATALOADER_AVAILABLE:
        return [], "unavailable", False
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            safe_pdf = _safe_pdf_path(pdf_path, tmp_dir)
            out_path = tmp_path / "output"
            out_path.mkdir(exist_ok=True)
            opendataloader_pdf.convert(
                input_path=str(safe_pdf),
                output_dir=str(out_path),
                format="markdown,json"
            )
            json_files = list(out_path.glob("**/*.json"))
            md_files   = list(out_path.glob("**/*.md"))
            if json_files:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    odl_data = json.load(f)
                pages_data = _parse_odl_json(odl_data)
                if pages_data:
                    return pages_data, "opendataloader_json", True
            if md_files:
                with open(md_files[0], "r", encoding="utf-8") as f:
                    md_content = f.read()
                pages_data = _parse_odl_markdown(md_content)
                if pages_data:
                    return pages_data, "opendataloader_markdown", True
            return [], "opendataloader_empty", False
    except Exception as e:
        return [], f"opendataloader_error:{str(e)[:60]}", False


def is_image_based_pdf(doc: fitz.Document, sample: int = 3) -> bool:
    pages_to_check = min(sample, len(doc))
    total = sum(len(doc[i].get_text("text").strip()) for i in range(pages_to_check))
    return (total / pages_to_check if pages_to_check > 0 else 0) < 50


def extract_text_pymupdf_fallback(pdf_path: Path) -> Tuple[List[Dict], str]:
    try:
        doc          = fitz.open(pdf_path)
        pdf_is_image = is_image_based_pdf(doc)
        method       = "ocr_pymupdf" if pdf_is_image else "pymupdf_fallback"
        pages_data   = []
        for page_num, page in enumerate(doc, start=1):
            if pdf_is_image:
                try:
                    tp   = page.get_textpage_ocr(flags=0, language="fra+eng", dpi=300)
                    text = page.get_text(textpage=tp).strip()
                except Exception:
                    text = ""
                if not text:
                    try:
                        import pytesseract
                        from PIL import Image
                        import io
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = page.get_pixmap(matrix=mat)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = pytesseract.image_to_string(img, lang="fra+eng").strip()
                    except Exception:
                        text = ""
            else:
                text = page.get_text("text").strip()
            if not text:
                continue
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            pages_data.append({
                "page_number":       page_num,
                "text":              text,
                "char_count":        len(text),
                "word_count":        len(text.split()),
                "extraction_method": method,
            })
        doc.close()
        return pages_data, method
    except Exception as e:
        return [], f"pymupdf_error:{str(e)[:60]}"

# ============================================================================
# ███ BLOC 6 — CONTEXTUAL RETRIEVAL (enrichissement LLM avant indexation) ███
# ============================================================================

CONTEXTUAL_PROMPT = """Tu es un assistant spécialisé dans les documents statistiques de l'ANSTAT-CI.

Document source : {titre} ({annee})
Section du document : {section}

Voici un passage extrait de ce document :
<passage>
{chunk_text}
</passage>

En 2 à 3 phrases COURTES, décris le contexte de ce passage :
- De quel sujet traite-t-il précisément ?
- Quelles données clés contient-il (taux, régions, populations concernées) ?
- Comment s'inscrit-il dans le document global ?

Réponds UNIQUEMENT avec le contexte, sans introduction ni conclusion."""


def _call_vllm_chat(llm_url: str, llm_model: str,
                    system_msg: str, user_msg: str,
                    max_tokens: int = 150, timeout: int = 60) -> str:
    """
    Appel vLLM via endpoint /v1/chat/completions (format OpenAI messages).
    Compatible avec : vLLM, LM Studio, OpenAI API.
    """
    resp = requests.post(
        llm_url,
        headers={"Content-Type": "application/json"},
        json={
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            "max_tokens":  max_tokens,
            "temperature": 0,
            "stop":        ["\n\n"],
        },
        timeout=timeout,
    )
    if resp.status_code == 200:
        data = resp.json()
        # Format chat/completions : choices[0].message.content
        return data["choices"][0]["message"]["content"].strip()
    return ""


def generate_contextual_description(chunk_text: str,
                                    doc_info: Dict,
                                    section: str = "",
                                    llm_url: str = "http://192.168.1.230:30179/v1/chat/completions",
                                    llm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
                                    use_ollama: bool = False,
                                    ollama_model: str = "qwen2.5:7b-instruct") -> str:
    """
    Génère une description contextuelle du chunk via LLM local.
    Supporte :
      - vLLM avec endpoint /v1/chat/completions (format OpenAI messages)
      - Ollama via /api/generate
    Fallback : contexte minimal basé sur les métadonnées si LLM indisponible.
    """
    if not REQUESTS_AVAILABLE:
        return _minimal_context(chunk_text, doc_info, section)

    titre     = doc_info.get("titre", "Document ANSTAT-CI")
    annee     = doc_info.get("annee", "")
    annee_str = str(annee) if annee else "année non précisée"

    system_msg = (
        "Tu es un assistant spécialisé dans les documents statistiques de l'ANSTAT-CI. "
        "Tu génères des descriptions contextuelles courtes et factuelles."
    )
    user_msg = CONTEXTUAL_PROMPT.format(
        titre=titre, annee=annee_str, section=section,
        chunk_text=chunk_text[:800],
    )

    try:
        if use_ollama:
            # Ollama : format /api/generate avec stream=False
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":   ollama_model,
                    "prompt":  f"{system_msg}\n\n{user_msg}",
                    "stream":  False,
                    "options": {"num_predict": 150, "temperature": 0},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        else:
            # vLLM : format /v1/chat/completions (OpenAI messages)
            result = _call_vllm_chat(
                llm_url, llm_model, system_msg, user_msg,
                max_tokens=150, timeout=60,
            )
            if result:
                return result
    except requests.exceptions.Timeout:
        pass  # Timeout → fallback silencieux
    except Exception:
        pass

    return _minimal_context(chunk_text, doc_info, section)


def _minimal_context(chunk_text: str, doc_info: Dict, section: str) -> str:
    """Contexte minimal si LLM indisponible — basé sur métadonnées."""
    titre  = doc_info.get("titre", "Document ANSTAT-CI")
    annee  = doc_info.get("annee", "")
    themes = ", ".join(doc_info.get("themes", []))
    parts  = [f"Extrait de : {titre}"]
    if annee:
        parts.append(f"Année : {annee}")
    if section:
        parts.append(f"Section : {section}")
    if themes:
        parts.append(f"Thèmes : {themes}")
    return " | ".join(parts)

# ============================================================================
# ███ BLOC 7 — RAPTOR : ARBRE RÉCURSIF 4 NIVEAUX ███
# ============================================================================
#
#  NIVEAU 0 — Chunks feuilles (256 tokens ≈ 1400 chars)
#             Passage précis, réponse à une question spécifique
#
#  NIVEAU 1 — Résumés de section (agrège 4-6 chunks N0)
#             "Section 3.2 — Taux de pauvreté par région"
#
#  NIVEAU 2 — Résumés de chapitre (agrège 3-5 résumés N1)
#             "Chapitre 3 — Pauvreté monétaire"
#
#  NIVEAU 3 — Résumé document (1 seul, agrège tout)
#             Vue d'ensemble du rapport complet
#
# ============================================================================

def raptor_chunk_level0(pages_data: List[Dict],
                        config: "PipelineConfig") -> List[Dict]:
    """
    NIVEAU 0 — Chunks feuilles fins (256 tokens / ~1400 chars).
    Respecte les frontières de paragraphes, puis de phrases.
    """
    chunks   = []
    para_sep = re.compile(r'\n\s*\n')
    sent_sep = re.compile(r'[.!?]+[\s\n]+')

    for page in pages_data:
        text     = page["text"]
        page_num = page["page_number"]
        method   = page.get("extraction_method", "unknown")

        if not text or len(text) < config.min_chunk_size:
            if text and len(text) > 50:
                chunks.append(_make_raw_chunk(text, page_num, method, 0, "leaf"))
            continue

        # Découper par paragraphes d'abord
        paragraphs = [p.strip() for p in para_sep.split(text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        current, current_len = [], 0

        for para in paragraphs:
            para_len = len(para)

            # Paragraphe trop long → découper par phrases
            if para_len > config.chunk_size_l0:
                # Flush current
                if current:
                    chunk_text = "\n\n".join(current)
                    if len(chunk_text) >= config.min_chunk_size:
                        chunks.append(_make_raw_chunk(
                            chunk_text, page_num, method, len(chunks), "leaf"))
                    current, current_len = [], 0
                # Découper le long paragraphe par phrases
                sentences = [s.strip() for s in sent_sep.split(para) if s.strip()]
                cur_s, cur_s_len = [], 0
                for sent in sentences:
                    if cur_s_len + len(sent) <= config.chunk_size_l0:
                        cur_s.append(sent)
                        cur_s_len += len(sent)
                    else:
                        if cur_s:
                            ct = " ".join(cur_s)
                            if len(ct) >= config.min_chunk_size:
                                chunks.append(_make_raw_chunk(
                                    ct, page_num, method, len(chunks), "leaf"))
                        # Overlap : garder les 2 dernières phrases
                        overlap = cur_s[-2:] if len(cur_s) >= 2 else cur_s[-1:]
                        cur_s     = overlap + [sent]
                        cur_s_len = sum(len(s) for s in cur_s)
                if cur_s:
                    ct = " ".join(cur_s)
                    if len(ct) >= config.min_chunk_size:
                        chunks.append(_make_raw_chunk(
                            ct, page_num, method, len(chunks), "leaf"))
                continue

            # Paragraphe normal
            if current_len + para_len <= config.chunk_size_l0:
                current.append(para)
                current_len += para_len
            else:
                if current:
                    chunk_text = "\n\n".join(current)
                    if len(chunk_text) >= config.min_chunk_size:
                        chunks.append(_make_raw_chunk(
                            chunk_text, page_num, method, len(chunks), "leaf"))
                # Overlap : garder dernier paragraphe si pas trop grand
                if para_len <= config.chunk_overlap:
                    current, current_len = [para], para_len
                else:
                    current, current_len = [para], para_len

        # Flush final
        if current:
            chunk_text = "\n\n".join(current)
            if len(chunk_text) >= config.min_chunk_size:
                chunks.append(_make_raw_chunk(
                    chunk_text, page_num, method, len(chunks), "leaf"))

    return chunks


def _make_raw_chunk(text: str, page_num: int, method: str,
                    idx: int, level: str) -> Dict:
    return {
        "text":     text,
        "page_num": page_num,
        "method":   method,
        "idx":      idx,
        "level":    level,
    }


def raptor_build_level1(leaf_chunks: List[Dict],
                        doc_info: Dict,
                        config: "PipelineConfig",
                        llm_url: str = "",
                        use_ollama: bool = False,
                        llm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ") -> List[Dict]:
    """
    NIVEAU 1 — Résumés de sections (agrège 4-6 chunks feuilles).
    Groupe par page et par thème, génère un résumé LLM.
    """
    if not leaf_chunks:
        return []

    # Grouper par plages de pages (window de 3 pages)
    sections = []
    group_size = config.raptor_group_size_l1  # ex: 5 chunks par section

    for i in range(0, len(leaf_chunks), group_size):
        group  = leaf_chunks[i:i + group_size]
        combined_text = "\n\n".join(c["text"] for c in group)
        pages_in_group = sorted({c["page_num"] for c in group})

        # Résumé via LLM si disponible, sinon troncature intelligente
        if llm_url or use_ollama:
            summary = _summarize_group(combined_text, doc_info,
                                       llm_url, use_ollama, llm_model=llm_model)
        else:
            # Résumé : première phrase de chaque chunk
            summary_parts = []
            for c in group:
                first_sent = c["text"].split('.')[0].strip()
                if first_sent and len(first_sent) > 20:
                    summary_parts.append(first_sent + ".")
            summary = " ".join(summary_parts[:4])

        if not summary or len(summary) < 30:
            summary = combined_text[:500]

        sections.append({
            "text":     summary,
            "page_num": pages_in_group[0] if pages_in_group else 1,
            "method":   "raptor_level1",
            "idx":      len(sections),
            "level":    "section",
            "source_chunk_indices": list(range(i, min(i+group_size, len(leaf_chunks)))),
        })

    return sections


def raptor_build_level2(section_chunks: List[Dict],
                        doc_info: Dict,
                        config: "PipelineConfig",
                        llm_url: str = "",
                        use_ollama: bool = False,
                        llm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ") -> List[Dict]:
    """
    NIVEAU 2 — Résumés de chapitres (agrège sections).
    """
    if not section_chunks or len(section_chunks) < 2:
        return []

    chapters = []
    group_size = config.raptor_group_size_l2  # ex: 4 sections par chapitre

    for i in range(0, len(section_chunks), group_size):
        group         = section_chunks[i:i + group_size]
        combined_text = "\n\n".join(c["text"] for c in group)

        if llm_url or use_ollama:
            summary = _summarize_group(combined_text, doc_info, llm_url, use_ollama,
                                       llm_model=llm_model, summary_type="chapitre")
        else:
            summary = combined_text[:800]

        if not summary or len(summary) < 30:
            summary = combined_text[:600]

        chapters.append({
            "text":     summary,
            "page_num": group[0]["page_num"],
            "method":   "raptor_level2",
            "idx":      len(chapters),
            "level":    "chapter",
        })

    return chapters


def raptor_build_level3(chapter_chunks: List[Dict],
                        doc_info: Dict,
                        llm_url: str = "",
                        use_ollama: bool = False,
                        llm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ") -> Optional[Dict]:
    """
    NIVEAU 3 — Résumé global du document (1 seul chunk).
    """
    if not chapter_chunks:
        return None

    combined = "\n\n".join(c["text"] for c in chapter_chunks)
    titre    = doc_info.get("titre", "Document ANSTAT-CI")
    annee    = doc_info.get("annee", "")

    if llm_url or use_ollama:
        summary = _summarize_group(combined, doc_info, llm_url, use_ollama,
                                   summary_type="document")
    else:
        # Résumé minimal sans LLM
        summary = (
            f"Ce document ({titre}, {annee}) couvre les thèmes suivants : "
            f"{', '.join(doc_info.get('themes', []))}. "
            f"Il contient des données sur {combined[:300]}..."
        )

    return {
        "text":     summary,
        "page_num": 1,
        "method":   "raptor_level3",
        "idx":      0,
        "level":    "document",
    }


def _summarize_group(text: str, doc_info: Dict,
                     llm_url: str, use_ollama: bool,
                     llm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
                     summary_type: str = "section") -> str:
    """
    Résumé LLM d'un groupe de chunks.
    Utilise /v1/chat/completions pour vLLM, /api/generate pour Ollama.
    """
    titre = doc_info.get("titre", "Document ANSTAT-CI")
    annee = doc_info.get("annee", "")

    system_msg = (
        "Tu es un expert en statistiques et données économiques de Côte d'Ivoire. "
        "Tu produis des résumés factuels, précis et concis."
    )
    user_msg = (
        f"Résume ces passages du rapport : {titre} ({annee}).\n\n"
        f"Passages :\n{text[:1200]}\n\n"
        f"Résumé FACTUEL en 3-4 phrases. Cite les chiffres clés. "
        f"Commence directement par les faits sans introduction."
    )
    try:
        if use_ollama:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":   "qwen2.5:7b-instruct",
                    "prompt":  f"{system_msg}\n\n{user_msg}",
                    "stream":  False,
                    "options": {"num_predict": 200, "temperature": 0},
                },
                timeout=90,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        elif llm_url:
            result = _call_vllm_chat(
                llm_url, llm_model, system_msg, user_msg,
                max_tokens=200, timeout=90,
            )
            if result:
                return result
    except requests.exceptions.Timeout:
        pass
    except Exception:
        pass
    return ""

# ============================================================================
# ███ BLOC 8 — LOOKUP DOCUMENT + DÉTECTION THÈMES ███
# ============================================================================

def lookup_document_info(pdf_path: Path) -> Dict[str, Any]:
    def normalize(s: str) -> str:
        s = s.lower()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = re.sub(r"[_\-\s]", "", s)
        return s

    stem_raw  = pdf_path.stem
    stem_norm = normalize(stem_raw)
    stem_basic = stem_raw.lower().replace(" ", "_")

    if stem_basic in DOCUMENT_CATALOG:
        return DOCUMENT_CATALOG[stem_basic]
    for key, info in DOCUMENT_CATALOG.items():
        if normalize(key) == stem_norm:
            return info
    for key, info in DOCUMENT_CATALOG.items():
        key_norm = normalize(key)
        if key_norm in stem_norm or stem_norm in key_norm:
            return info
    annees  = re.findall(r"\d{4}", stem_raw)
    sigles  = [normalize(s) for s in re.findall(r"[a-zA-Z]{2,}", stem_raw)]
    for key, info in DOCUMENT_CATALOG.items():
        key_norm = normalize(key)
        if any(a in key for a in annees) and \
           any(s in key_norm for s in sigles if len(s) >= 3):
            return info
    return {"titre": pdf_path.stem, "url": "", "themes": [], "annee": None}


def detect_themes(text: str, doc_themes: List[str]) -> List[str]:
    text_lower = text.lower()
    detected   = set(doc_themes)
    for theme, keywords in THEME_KEYWORDS.items():
        if sum(1 for kw in keywords if kw.lower() in text_lower) >= 2:
            detected.add(theme)
    return list(detected)[:6]

# ============================================================================
# ███ BLOC 9 — ASSEMBLAGE FINAL DES CHUNKS (format JSON identique à l'original)
# ============================================================================

def assemble_chunk(text: str, page_num: int, chunk_idx: int,
                   doc_info: Dict, doc_metadata: Dict,
                   level: str, extraction_method: str,
                   context_description: str = "",
                   table_texts: List[str] = None,
                   is_table_chunk: bool = False) -> Optional[Dict]:
    """
    Crée un chunk final au format JSON identique à l'original.
    Ajoute : niveau RAPTOR, description contextuelle, flag tableau.
    """
    if not text or len(text.strip()) < 30:
        return None

    # Nettoyage C5
    if not is_table_chunk:
        text = clean_chunk_intra(text)
    if not text or len(text.strip()) < 20:
        return None

    # Filtres C6/C8 (pas sur les tableaux)
    if not is_table_chunk:
        is_junk, reason = is_true_junk_chunk(text)
        if is_junk:
            return None
        if is_questionnaire_chunk(text)[0]:
            return None

    # Texte enrichi = contexte + chunk original
    enriched_text = text
    if context_description:
        enriched_text = f"[CONTEXTE: {context_description}]\n\n{text}"

    chunk_id = hashlib.md5(
        f"{doc_metadata.get('document_name', '')}_{page_num}_{chunk_idx}_{level}".encode()
    ).hexdigest()

    # Détection type de contenu
    if is_table_chunk:
        content_type = "tableau"
    elif level in ("section", "chapter", "document"):
        content_type = f"résumé_{level}"
    elif len(text) < 200 and (text.isupper() or text.endswith(':')):
        content_type = "titre"
    else:
        content_type = "paragraphe"

    is_exec = is_executive_summary_chunk(text)
    themes  = detect_themes(text, doc_info.get("themes", []))

    # ── Format JSON identique à l'original ────────────────────────────────
    return {
        "chunk_id": chunk_id,
        "text":     enriched_text,
        "metadata": {
            "document_name":        doc_metadata.get("document_name", ""),
            "document_title":       doc_info.get("titre", ""),
            "source_url":           doc_info.get("url", ""),
            "page_number":          page_num,
            "chunk_index":          chunk_idx,
            "themes":               themes,
            "content_type":         content_type,
            "raptor_level":         level,           # NOUVEAU : niveau RAPTOR
            "is_table":             is_table_chunk,  # NOUVEAU : flag tableau
            "is_executive_summary": is_exec,
            "context_description":  context_description,  # NOUVEAU
            "word_count":           len(text.split()),
            "char_count":           len(text),
            "sentence_count":       text.count('.') + text.count('!') + text.count('?'),
            "extraction_method":    extraction_method,
            "annee":                doc_info.get("annee"),  # NOUVEAU
            "created_at":           datetime.now(timezone.utc).isoformat(),
        }
    }

# ============================================================================
# ███ BLOC 10 — CONFIGURATION ███
# ============================================================================

@dataclass
class PipelineConfig:
    # Chemins
    documents_dir: Path = Path("./documents")
    output_dir:    Path = Path("./chunks_output_v2")

    # Chunking RAPTOR
    chunk_size_l0: int = 1400    # Niveau 0 feuilles (~256 tokens)
    chunk_overlap:  int = 200    # Overlap entre chunks
    min_chunk_size: int = 150    # Minimum absolu
    max_chunk_size: int = 3000   # Maximum absolu (pour tableaux longs)
    raptor_group_size_l1: int = 5  # Chunks N0 par résumé N1
    raptor_group_size_l2: int = 4  # Résumés N1 par résumé N2
    build_raptor_l1: bool = True
    build_raptor_l2: bool = True
    build_raptor_l3: bool = True

    # LLM pour contexte et RAPTOR
    use_llm_context:  bool = True  # True = enrichissement LLM activé
    use_ollama:       bool = False   # True = Ollama, False = vLLM
    ollama_model:     str  = "qwen2.5:7b-instruct"
    vllm_url:         str  = "http://192.168.1.230:30179/v1/chat/completions"
    vllm_model:       str  = "Qwen/Qwen2.5-7B-Instruct-AWQ"

    # Filtrage
    dedup_threshold: float = 0.80
    use_camelot:     bool  = True
    use_pdfplumber:  bool  = True

    # Parallélisme
    max_workers: int = None

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        if self.max_workers is None:
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)

# ============================================================================
# ███ BLOC 11 — PIPELINE PRINCIPAL ███
# ============================================================================

class RaptorRAGPipeline:
    """
    Pipeline RAG Ultime ANSTAT-CI.
    Extraction triple + RAPTOR 4 niveaux + Contextual Retrieval + filtrage corrigé.
    Sortie JSON identique au format original chunk_with_opendataloader.py.
    """

    def __init__(self, config: PipelineConfig):
        self.config  = config
        self.log_file = (
            config.output_dir / "logs" /
            f"pipeline_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.stats = {
            "total": 0, "success": 0, "failed": 0,
            "odl": 0, "pymupdf": 0, "chunks_l0": 0,
            "chunks_l1": 0, "chunks_l2": 0, "chunks_l3": 0,
            "table_chunks": 0, "dedup_removed": 0,
        }

    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
        if level in ("ERROR", "WARNING") or self.config.max_workers == 1:
            print(line)

    # ─── EXTRACTION TEXTE ────────────────────────────────────────────────────

    def extract_pages(self, pdf_path: Path) -> Tuple[List[Dict], str, str]:
        """Extrait le texte — OpenDataLoader principal, PyMuPDF fallback."""
        pages_data, method, success = extract_with_opendataloader(pdf_path)
        if success and pages_data:
            self.stats["odl"] += 1
            return pages_data, method, "opendataloader"

        self.log(f"ODL échoué pour {pdf_path.name}, fallback PyMuPDF", "WARNING")
        pages_data, method = extract_text_pymupdf_fallback(pdf_path)
        if pages_data:
            self.stats["pymupdf"] += 1
        return pages_data, method, "pymupdf"

    # ─── FILTRAGE PAGES ──────────────────────────────────────────────────────

    def filter_pages(self, pages_data: List[Dict],
                     pdf_path: Path) -> List[Dict]:
        """Applique C0, C1, C2, C4 au niveau des pages."""
        repeated_lines = detect_repeated_headers_footers(pages_data)
        filtered = []
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            doc = None

        for page in pages_data:
            text = clean_raw_text(page["text"])          # C0
            text = remove_header_footer_lines(text)       # C4
            text = remove_repeated_lines(text, repeated_lines)  # C4
            page["text"] = text
            if not text.strip():
                continue
            is_noise, _ = is_structural_noise_page(text) # C1
            if is_noise:
                continue
            # C2 bookmarks
            if doc:
                try:
                    toc = doc.get_toc()
                    if toc:
                        page_idx = page["page_number"] - 1
                        section_title = ""
                        for _, title, p1 in toc:
                            if p1 - 1 <= page_idx:
                                section_title = title.lower()
                            else:
                                break
                        if any(kw in section_title
                               for kw in STRUCTURAL_SECTION_KEYWORDS):
                            continue
                except Exception:
                    pass
            filtered.append(page)

        if doc:
            doc.close()
        return filtered

    # ─── EXTRACTION TABLEAUX ─────────────────────────────────────────────────

    def extract_tables(self, pdf_path: Path,
                       doc_info: Dict) -> Dict[int, List[str]]:
        """Triple moteur : pdfplumber + Camelot → fusion."""
        pdfplumber_tables, camelot_tables = {}, {}

        if self.config.use_pdfplumber:
            pdfplumber_tables = extract_tables_pdfplumber(pdf_path, doc_info)
            self.log(
                f"  pdfplumber: {sum(len(v) for v in pdfplumber_tables.values())} tableaux"
            )

        if self.config.use_camelot:
            camelot_tables = extract_tables_camelot(pdf_path, doc_info)
            self.log(
                f"  Camelot:    {sum(len(v) for v in camelot_tables.values())} tableaux"
            )

        return merge_table_sources(pdfplumber_tables, camelot_tables)

    # ─── PIPELINE DOCUMENT UNIQUE ─────────────────────────────────────────────

    def process_document(self, pdf_path: Path) -> Dict:
        doc_start = time.time()
        result = {
            "file": pdf_path.name, "success": False,
            "chunks_count": 0, "processing_time": 0, "error": None,
        }

        try:
            self.log(f"⚙️  Traitement: {pdf_path.name}")
            doc_info = lookup_document_info(pdf_path)
            self.log(f"  📚 Document: {doc_info['titre']}")

            # ── 1. Extraction texte ──────────────────────────────────────────
            pages_data, method, engine = self.extract_pages(pdf_path)
            if not pages_data:
                result["error"] = "Aucun texte extrait"
                return result
            self.log(f"  📄 {len(pages_data)} pages extraites ({engine})")

            # ── 2. Filtrage pages ────────────────────────────────────────────
            pages_data = self.filter_pages(pages_data, pdf_path)
            self.log(f"  🔍 {len(pages_data)} pages après filtrage")

            if not pages_data:
                result["error"] = "Aucune page après filtrage"
                return result

            # ── 3. Extraction tableaux ───────────────────────────────────────
            tables_by_page = self.extract_tables(pdf_path, doc_info)
            n_tables = sum(len(v) for v in tables_by_page.values())
            self.log(f"  📊 {n_tables} tableaux extraits")

            # ── 4. RAPTOR Niveau 0 — chunks feuilles ────────────────────────
            leaf_chunks = raptor_chunk_level0(pages_data, self.config)
            self.log(f"  🌿 {len(leaf_chunks)} chunks N0 (feuilles)")

            # ── 5. Métadonnées document ──────────────────────────────────────
            doc_metadata = {
                "document_name":     pdf_path.name,
                "document_title":    doc_info["titre"],
                "document_url":      doc_info.get("url", ""),
                "document_themes":   doc_info.get("themes", []),
                "document_path":     str(pdf_path),
                "document_size":     pdf_path.stat().st_size,
                "annee":             doc_info.get("annee"),
                "total_pages":       len(pages_data),
                "extraction_method": method,
                "extraction_engine": engine,
                "extraction_date":   datetime.now(timezone.utc).isoformat(),
            }

            # ── 6. Assemblage chunks N0 avec filtre + contexte ───────────────
            llm_url = self.config.vllm_url if not self.config.use_ollama else ""

            all_chunks = []
            for i, raw in enumerate(leaf_chunks):
                # Filtres C3/C7 corrigés
                is_junk, _ = is_true_junk_chunk(raw["text"])
                if is_junk:
                    continue

                # Contexte LLM (si activé)
                ctx = ""
                if self.config.use_llm_context:
                    ctx = generate_contextual_description(
                        raw["text"], doc_info,
                        llm_url=llm_url,
                        use_ollama=self.config.use_ollama,
                        ollama_model=self.config.ollama_model,
                    )

                chunk = assemble_chunk(
                    text=raw["text"],
                    page_num=raw["page_num"],
                    chunk_idx=i,
                    doc_info=doc_info,
                    doc_metadata=doc_metadata,
                    level="leaf",
                    extraction_method=raw["method"],
                    context_description=ctx,
                )
                if chunk:
                    all_chunks.append(chunk)

            self.stats["chunks_l0"] += len(all_chunks)
            self.log(f"  ✅ {len(all_chunks)} chunks N0 valides")

            # ── 7. Chunks tableaux (niveau 0 aussi, flag is_table=True) ──────
            table_chunks_count = 0
            for page_num, table_texts in tables_by_page.items():
                for j, table_text in enumerate(table_texts):
                    if len(table_text) < 50:
                        continue
                    ctx = ""
                    if self.config.use_llm_context:
                        ctx = generate_contextual_description(
                            table_text, doc_info,
                            llm_url=llm_url,
                            use_ollama=self.config.use_ollama,
                        )
                    chunk = assemble_chunk(
                        text=table_text,
                        page_num=page_num,
                        chunk_idx=len(all_chunks) + j,
                        doc_info=doc_info,
                        doc_metadata=doc_metadata,
                        level="leaf",
                        extraction_method="table_extraction",
                        context_description=ctx,
                        is_table_chunk=True,
                    )
                    if chunk:
                        all_chunks.append(chunk)
                        table_chunks_count += 1

            self.stats["table_chunks"] += table_chunks_count
            self.log(f"  📊 {table_chunks_count} chunks tableaux ajoutés")

            # ── 8. RAPTOR Niveau 1 — résumés de sections ────────────────────
            if self.config.build_raptor_l1 and len(leaf_chunks) >= 3:
                l1_raw = raptor_build_level1(
                    leaf_chunks, doc_info, self.config,
                    llm_url=llm_url, use_ollama=self.config.use_ollama,
                    llm_model=self.config.vllm_model,
                )
                for i, raw in enumerate(l1_raw):
                    chunk = assemble_chunk(
                        text=raw["text"],
                        page_num=raw["page_num"],
                        chunk_idx=len(all_chunks) + i,
                        doc_info=doc_info,
                        doc_metadata=doc_metadata,
                        level="section",
                        extraction_method="raptor_level1",
                    )
                    if chunk:
                        all_chunks.append(chunk)
                self.stats["chunks_l1"] += len(l1_raw)
                self.log(f"  🌿 {len(l1_raw)} résumés N1 (sections)")

                # ── 9. RAPTOR Niveau 2 — résumés de chapitres ───────────────
                if self.config.build_raptor_l2 and len(l1_raw) >= 3:
                    l2_raw = raptor_build_level2(
                        l1_raw, doc_info, self.config,
                        llm_url=llm_url, use_ollama=self.config.use_ollama,
                        llm_model=self.config.vllm_model,
                    )
                    for i, raw in enumerate(l2_raw):
                        chunk = assemble_chunk(
                            text=raw["text"],
                            page_num=raw["page_num"],
                            chunk_idx=len(all_chunks) + i,
                            doc_info=doc_info,
                            doc_metadata=doc_metadata,
                            level="chapter",
                            extraction_method="raptor_level2",
                        )
                        if chunk:
                            all_chunks.append(chunk)
                    self.stats["chunks_l2"] += len(l2_raw)
                    self.log(f"  🌳 {len(l2_raw)} résumés N2 (chapitres)")

                    # ── 10. RAPTOR Niveau 3 — résumé global ─────────────────
                    if self.config.build_raptor_l3 and len(l2_raw) >= 2:
                        l3 = raptor_build_level3(
                            l2_raw, doc_info,
                            llm_url=llm_url, use_ollama=self.config.use_ollama,
                            llm_model=self.config.vllm_model,
                        )
                        if l3:
                            chunk = assemble_chunk(
                                text=l3["text"],
                                page_num=1,
                                chunk_idx=len(all_chunks),
                                doc_info=doc_info,
                                doc_metadata=doc_metadata,
                                level="document",
                                extraction_method="raptor_level3",
                            )
                            if chunk:
                                all_chunks.append(chunk)
                                self.stats["chunks_l3"] += 1
                            self.log(f"  🌲 Résumé N3 (document global)")

            # ── 11. Déduplication finale (C9) ────────────────────────────────
            before_dedup = len(all_chunks)
            all_chunks, n_removed = deduplicate_chunks(
                all_chunks, self.config.dedup_threshold
            )
            self.stats["dedup_removed"] += n_removed
            if n_removed > 0:
                self.log(f"  🗑️  C9: {n_removed} doublons supprimés "
                         f"({before_dedup} → {len(all_chunks)})")

            # ── 12. Sauvegarde JSON (format identique à l'original) ──────────
            if all_chunks:
                output_file = (
                    self.config.output_dir /
                    f"{pdf_path.stem.replace(' ', '_')}_chunks.json"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "metadata":        doc_metadata,
                        "chunks_count":    len(all_chunks),
                        "raptor_stats": {
                            "level0_leaf":    self.stats["chunks_l0"],
                            "level1_section": self.stats["chunks_l1"],
                            "level2_chapter": self.stats["chunks_l2"],
                            "level3_document": self.stats["chunks_l3"],
                            "table_chunks":   self.stats["table_chunks"],
                        },
                        "processing_date": datetime.now(timezone.utc).isoformat(),
                        "chunks":          all_chunks,
                    }, f, ensure_ascii=False, indent=2)
                self.log(f"  💾 Sauvegardé: {output_file.name} "
                         f"({len(all_chunks)} chunks)")

            elapsed = time.time() - doc_start
            result.update({
                "success":         True,
                "chunks_count":    len(all_chunks),
                "processing_time": elapsed,
                "document_title":  doc_info["titre"],
                "source_url":      doc_info.get("url", ""),
                "themes":          doc_info.get("themes", []),
                "engine":          engine,
            })
            self.stats["success"] += 1
            print(f"  ✅ {pdf_path.name}: {len(all_chunks)} chunks en {elapsed:.1f}s")

        except Exception as e:
            result["error"] = str(e)
            self.stats["failed"] += 1
            self.log(f"✗ {pdf_path.name}: {str(e)}", "ERROR")
            traceback.print_exc()

        return result

    # ─── TRAITEMENT BATCH PARALLÈLE ─────────────────────────────────────────

    def process_all(self) -> Dict:
        self.log("=" * 60)
        self.log("PIPELINE RAG V2 — RAPTOR + TABLEAUX + CONTEXTUAL RETRIEVAL")
        self.log("=" * 60)

        pdf_files = list(self.config.documents_dir.glob("*.pdf"))
        if not pdf_files:
            pdf_files = list(self.config.documents_dir.glob("**/*.pdf"))
        if not pdf_files:
            return {"error": "Aucun PDF trouvé"}

        self.stats["total"] = len(pdf_files)
        self.log(f"📚 {len(pdf_files)} documents à traiter")
        self.log(f"🔬 Moteurs: OpenDataLoader + pdfplumber + Camelot")
        self.log(f"🌲 RAPTOR: N0(feuilles) + N1(sections) + N2(chapitres) + N3(doc)")
        self.log(f"🧠 Contexte LLM: {'✅ activé' if self.config.use_llm_context else '❌ désactivé'}")

        all_results = []
        start_time  = time.time()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.process_document, pdf): pdf
                for pdf in pdf_files
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                all_results.append(result)
                progress = (completed / len(pdf_files)) * 100
                elapsed  = time.time() - start_time
                eta      = (elapsed / completed) * (len(pdf_files) - completed)
                print(
                    f"\r📊 {completed}/{len(pdf_files)} "
                    f"({progress:.1f}%) — {elapsed:.0f}s — ETA:{eta:.0f}s",
                    end="", flush=True
                )

        print()
        total_time   = time.time() - start_time
        successful   = [r for r in all_results if r["success"]]
        failed       = [r for r in all_results if not r["success"]]
        total_chunks = sum(r["chunks_count"] for r in successful)

        report = {
            "timestamp":             datetime.now(timezone.utc).isoformat(),
            "total_documents":       len(pdf_files),
            "successful_documents":  len(successful),
            "failed_documents":      len(failed),
            "total_chunks":          total_chunks,
            "raptor_stats":          {
                "chunks_l0":      self.stats["chunks_l0"],
                "chunks_l1":      self.stats["chunks_l1"],
                "chunks_l2":      self.stats["chunks_l2"],
                "chunks_l3":      self.stats["chunks_l3"],
                "table_chunks":   self.stats["table_chunks"],
                "dedup_removed":  self.stats["dedup_removed"],
            },
            "total_time_seconds":    total_time,
            "docs_per_minute":       (len(successful) / total_time) * 60 if total_time > 0 else 0,
            "successful_files": [
                {"file": r["file"], "chunks": r["chunks_count"],
                 "title": r.get("document_title", ""), "engine": r.get("engine", "")}
                for r in successful
            ],
            "failed_files": [
                {"file": r["file"], "error": r["error"]} for r in failed
            ],
        }

        report_path = self.config.output_dir / "pipeline_v2_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._print_summary(report)
        return report

    def _print_summary(self, report: Dict):
        print("\n" + "=" * 65)
        print("RAPPORT — PIPELINE RAG V2 (RAPTOR + TABLEAUX)")
        print("=" * 65)
        print(f"✅ Documents réussis   : {report['successful_documents']}/{report['total_documents']}")
        print(f"📦 Total chunks        : {report['total_chunks']}")
        rs = report.get("raptor_stats", {})
        print(f"\n🌲 Répartition RAPTOR :")
        print(f"   N0 feuilles        : {rs.get('chunks_l0', 0)}")
        print(f"   N1 sections        : {rs.get('chunks_l1', 0)}")
        print(f"   N2 chapitres       : {rs.get('chunks_l2', 0)}")
        print(f"   N3 document        : {rs.get('chunks_l3', 0)}")
        print(f"   Tableaux           : {rs.get('table_chunks', 0)}")
        print(f"   Doublons supprimés : {rs.get('dedup_removed', 0)}")
        print(f"\n⏱️  Temps total         : {report['total_time_seconds']:.1f}s")
        print(f"🚀 Vitesse             : {report['docs_per_minute']:.1f} docs/min")
        if report.get("failed_files"):
            print(f"\n❌ Échecs : {report['failed_documents']}")
            for f in report["failed_files"][:5]:
                print(f"   • {f['file']}: {str(f['error'])[:60]}")
        print(f"\n📁 Résultats : {self.config.output_dir}")
        print("=" * 65)

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    print("\n" + "=" * 65)
    print("PIPELINE RAG V2 — OpenDataLoader + RAPTOR + Tableaux")
    print("=" * 65 + "\n")

    config = PipelineConfig(
        documents_dir=Path(r"C:\Users\KABA\rag_anstat\pro\documents"),
        output_dir=Path("./chunks_output_v2"),

        # RAPTOR
        chunk_size_l0=1400,          # ~256 tokens pour N0
        chunk_overlap=200,
        min_chunk_size=150,
        max_chunk_size=3000,
        raptor_group_size_l1=5,      # 5 chunks N0 → 1 résumé N1
        raptor_group_size_l2=4,      # 4 résumés N1 → 1 résumé N2
        build_raptor_l1=True,
        build_raptor_l2=True,
        build_raptor_l3=True,

        # LLM contextuel (désactiver si pas de LLM disponible)
        use_llm_context=True,       # Mettre True si Ollama ou vLLM actif
        use_ollama=False,
        ollama_model="qwen2.5:7b-instruct",
        vllm_url="http://192.168.1.230:30179/v1/chat/completions",
        vllm_model="Qwen/Qwen2.5-7B-Instruct-AWQ",

        # Tableaux
        use_camelot=True,
        use_pdfplumber=True,

        # Déduplication
        dedup_threshold=0.80,

        # Parallélisme
        max_workers=1,
    )

    if not config.documents_dir.exists():
        print(f"❌ Répertoire introuvable: {config.documents_dir}")
        return

    pdf_files = list(config.documents_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Aucun PDF trouvé dans: {config.documents_dir}")
        return

    print(f"📁 Source      : {config.documents_dir}")
    print(f"📁 Destination : {config.output_dir}")
    print(f"📚 Documents   : {len(pdf_files)}")
    print(f"\n🌲 RAPTOR      : N0(1400 chars) + N1(5 chunks) + N2(4 sections) + N3(doc)")
    print(f"📊 Tableaux    : pdfplumber + Camelot (fusion intelligente)")
    print(f"🧠 Contexte LLM: {'✅ activé' if config.use_llm_context else '❌ désactivé (activer avec use_llm_context=True)'}")
    print(f"\n🚀 Démarrage dans 3 secondes...")
    time.sleep(3)

    pipeline = RaptorRAGPipeline(config)
    try:
        report = pipeline.process_all()
        if report.get("successful_documents", 0) > 0:
            print("\n✅ TRAITEMENT TERMINÉ AVEC SUCCÈS!")
        else:
            print("\n⚠️  Aucun document traité avec succès")
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur critique: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()