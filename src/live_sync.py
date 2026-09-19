# ============================================================
# live_sync.py
# Synchronisation en direct entre le planificateur (processus
# indépendant) et le dashboard Streamlit.
#
# Problème résolu : le planificateur écrit ses résultats sur disque
# (session_store.py) mais ne peut pas toucher au st.session_state
# d'une session Streamlit déjà ouverte dans un navigateur — ce sont
# deux processus Python séparés. Ce module fournit donc le mécanisme
# de VERIFICATION (le dashboard va lui-même comparer périodiquement
# ce qu'il affiche à ce qui existe sur disque) et de RECHARGEMENT
# ciblé, à appeler à chaque exécution du script Streamlit (déclenchée
# soit par une interaction utilisateur, soit par l'auto-rafraîchissement
# léger mis en place dans scheduler_ui.py).
#
# Principe de coût : la vérification (has_newer_run_on_disk) ne lit
# qu'un petit pointeur JSON — pas les DataFrames/modèles. Le
# rechargement complet (joblib, potentiellement plusieurs Mo) n'a lieu
# que si un run réellement différent a été détecté.
# ============================================================

from typing import Any, Dict, Optional

import streamlit as st

from .session_store import get_last_session_info, load_last_full_session


ACTIVE_RUN_KEY = "_active_run_id"


def get_active_run_id() -> Optional[str]:
    """Run_id actuellement affiché par CETTE session Streamlit."""
    return st.session_state.get(ACTIVE_RUN_KEY)


def set_active_run_id(run_id: Optional[str]) -> None:
    st.session_state[ACTIVE_RUN_KEY] = run_id


def has_newer_run_on_disk() -> Optional[Dict[str, Any]]:
    """
    Vérification légère : compare le run_id actuellement affiché à
    celui du dernier pointeur écrit sur disque (par ce dashboard ou par
    le planificateur, peu importe le processus). Retourne les
    métadonnées du run si celui sur disque diffère, sinon None.
    """

    info = get_last_session_info()

    if info is None:
        return None

    if info.get("run_id") != get_active_run_id():
        return info

    return None


def sync_technique_page() -> Optional[Dict[str, Any]]:
    """
    A appeler en tête de 2_Dashboard_Technique.py, à CHAQUE exécution
    du script (premier chargement, interaction utilisateur, ou tick
    d'auto-rafraîchissement). Recharge pipeline_result depuis le disque
    si un run plus récent existe — que ce run vienne du planificateur
    ou d'une autre session Streamlit.

    Retourne les métadonnées du nouveau run si un rechargement a eu
    lieu, sinon None (rien de neuf, ou rechargement impossible).
    """

    newer = has_newer_run_on_disk()

    if newer is None:
        return None

    bundle = load_last_full_session()

    if bundle is None or bundle.get("prep_result") is None:
        # Pointeur présent mais bundle illisible ou incomplet (ex : lu
        # au mauvais moment). On ne marque PAS le run comme actif, afin
        # de retenter au prochain tick plutôt que de rester bloqué.
        return None

    st.session_state.pipeline_result = bundle["prep_result"]
    set_active_run_id(newer["run_id"])

    return newer


def sync_analytical_page() -> Optional[Dict[str, Any]]:
    """
    Equivalent de sync_technique_page() pour 1_Dashboard_Analytique.py :
    recharge ml_result, les datasets enrichis et l'interprétation XAI.
    """

    newer = has_newer_run_on_disk()

    if newer is None:
        return None

    bundle = load_last_full_session()

    if bundle is None or bundle.get("ml_result") is None:
        return None

    st.session_state.analytical_ml_result = bundle["ml_result"]
    st.session_state.clustered_dataset = bundle.get("clustered_dataset")
    st.session_state.anomaly_dataset = bundle.get("anomaly_dataset")
    st.session_state.interpretation = bundle.get("interpretation")

    if bundle.get("prep_result") is not None:
        st.session_state.pipeline_result = bundle["prep_result"]

    set_active_run_id(newer["run_id"])

    return newer