# ============================================================
# session_store.py
# Persistance complète des résultats (préparation + ML + XAI)
# pour survivre à un redémarrage de Streamlit / du terminal.
# ============================================================

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import joblib


DEFAULT_SESSIONS_DIR = os.path.join("models", "sessions")
POINTER_FILENAME = "last_session.json"


def _pointer_path(sessions_dir: str) -> str:
    return os.path.join(sessions_dir, POINTER_FILENAME)


def save_full_session(
    run_id: str,
    prep_result: Optional[Dict[str, Any]] = None,
    ml_result: Optional[Dict[str, Any]] = None,
    clustered_dataset=None,
    anomaly_dataset=None,
    interpretation: Optional[Dict[str, Any]] = None,
    sessions_dir: str = DEFAULT_SESSIONS_DIR,
) -> str:
    """
    Sauvegarde un instantané complet de session sur disque (un seul
    fichier joblib par run_id, plus un pointeur JSON vers le dernier).

    Ce fichier contient tout ce qui n'existe normalement que dans
    st.session_state (X_ready, ml_result, datasets enrichis, XAI...).
    Il permet de restaurer l'état après fermeture du terminal, un
    redémarrage de Streamlit ou l'ouverture d'une nouvelle session.

    C'est aussi ce que le planificateur (scheduler_service.py) écrit
    après une exécution automatique : le dashboard peut ainsi afficher
    le dernier résultat sans qu'un technicien n'ait rien à faire.

    Note : merge avec le dernier instantané existant si des éléments
    ne sont pas fournis (ex : la page technique ne sauvegarde que
    prep_result, la page analytique complète ensuite avec ml_result).
    """

    os.makedirs(sessions_dir, exist_ok=True)

    existing = None

    pointer = _read_pointer(sessions_dir)

    if pointer and pointer.get("run_id") == run_id:
        existing = _load_bundle(pointer.get("bundle_path"))

    bundle = existing or {
        "run_id": run_id,
        "prep_result": None,
        "ml_result": None,
        "clustered_dataset": None,
        "anomaly_dataset": None,
        "interpretation": None,
    }

    bundle["run_id"] = run_id
    bundle["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if prep_result is not None:
        bundle["prep_result"] = prep_result

    if ml_result is not None:
        bundle["ml_result"] = ml_result

    if clustered_dataset is not None:
        bundle["clustered_dataset"] = clustered_dataset

    if anomaly_dataset is not None:
        bundle["anomaly_dataset"] = anomaly_dataset

    if interpretation is not None:
        bundle["interpretation"] = interpretation

    bundle_path = os.path.join(sessions_dir, f"session_{run_id}.pkl")

    # Ecriture atomique : on écrit dans un fichier temporaire puis on
    # renomme (os.replace est atomique sur POSIX et sur Windows). Ceci
    # évite qu'un autre processus (le dashboard Streamlit, qui peut lire
    # ce fichier au même moment) ne tombe sur un fichier partiellement
    # écrit — un vrai risque dès lors que planificateur et dashboard
    # tournent en parallèle et que ce fichier peut peser plusieurs Mo.
    tmp_bundle_path = bundle_path + ".tmp"
    joblib.dump(bundle, tmp_bundle_path)
    os.replace(tmp_bundle_path, bundle_path)

    pointer = {
        "run_id": run_id,
        "bundle_path": bundle_path,
        "saved_at": bundle["saved_at"],
        "has_prep_result": bundle["prep_result"] is not None,
        "has_ml_result": bundle["ml_result"] is not None,
        "has_interpretation": bundle["interpretation"] is not None,
    }

    pointer_path = _pointer_path(sessions_dir)
    tmp_pointer_path = pointer_path + ".tmp"

    with open(tmp_pointer_path, "w", encoding="utf-8") as file:
        json.dump(pointer, file, indent=4, ensure_ascii=False)

    os.replace(tmp_pointer_path, pointer_path)

    return bundle_path


def _read_pointer(sessions_dir: str) -> Optional[Dict[str, Any]]:
    pointer_path = _pointer_path(sessions_dir)

    if not os.path.exists(pointer_path):
        return None

    try:
        with open(pointer_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def _load_bundle(bundle_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not bundle_path or not os.path.exists(bundle_path):
        return None

    try:
        return joblib.load(bundle_path)
    except Exception:
        return None


def load_last_full_session(
    sessions_dir: str = DEFAULT_SESSIONS_DIR
) -> Optional[Dict[str, Any]]:
    """
    Recharge le dernier instantané complet sauvegardé, ou None si
    aucun n'existe encore (première utilisation du projet).
    """

    pointer = _read_pointer(sessions_dir)

    if pointer is None:
        return None

    return _load_bundle(pointer.get("bundle_path"))


def get_last_session_info(
    sessions_dir: str = DEFAULT_SESSIONS_DIR
) -> Optional[Dict[str, Any]]:
    """
    Retourne uniquement les métadonnées légères du dernier run (sans
    charger les DataFrames/modèles) — utile pour un bandeau d'info
    rapide à afficher en tête de page.
    """

    return _read_pointer(sessions_dir)