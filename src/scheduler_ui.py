# ============================================================
# scheduler_ui.py
# Petits composants Streamlit partagés entre les deux dashboards :
#   - bandeau "résultat restauré automatiquement depuis le disque"
#   - contrôles de synchronisation en direct avec le planificateur
#   - panneau de statut du planificateur (lecture seule, ne démarre
#     jamais le planificateur lui-même : il tourne dans son propre
#     processus, voir scheduler_service.py)
# ============================================================

from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

from .scheduler_service import get_scheduler_status, DEFAULT_CONFIG_PATH
from .session_store import get_last_session_info
from .live_sync import get_active_run_id
from .ui_theme import pill, card


def render_restore_banner(session_info: Optional[dict]) -> None:
    """
    Affiche un bandeau discret indiquant qu'un résultat a été restauré
    automatiquement depuis le disque (après redémarrage du terminal,
    de Streamlit, ou dans une nouvelle session).
    """

    if not session_info:
        return

    st.markdown(
        pill(
            f"Dernier résultat restauré automatiquement "
            f"(run {session_info.get('run_id')} du {session_info.get('saved_at')})",
            tone="accent"
        ),
        unsafe_allow_html=True
    )
    st.write("")


def _trigger_soft_autorefresh(interval_seconds: int, key: str) -> None:
    """
    Déclenche un "soft rerun" périodique du script Streamlit — PAS un
    rechargement de page. C'est la différence essentielle : un rerun
    Streamlit (comme celui provoqué par un clic sur un widget) conserve
    st.session_state intact ; un F5 / meta-refresh crée une toute
    nouvelle session côté serveur et perdrait les filtres en cours.

    Utilise le composant communautaire `streamlit-autorefresh` s'il est
    installé (recommandé). A défaut, replie sur un rafraîchissement
    complet de la page (fonctionne, mais plus brutal) en avertissant
    l'utilisateur.
    """

    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_seconds * 1000, key=key)
    except ImportError:
        st.markdown(
            f'<meta http-equiv="refresh" content="{interval_seconds}">',
            unsafe_allow_html=True
        )
        st.caption(
            "⚠️ Paquet `streamlit-autorefresh` absent : rechargement complet "
            "de la page utilisé à la place (fonctionne, mais réinitialise "
            "les filtres à chaque cycle). `pip install streamlit-autorefresh` "
            "pour une actualisation fluide qui préserve votre état."
        )


def render_live_sync_controls(
    sync_result: Optional[Dict[str, Any]],
    label: str = "les résultats",
    default_enabled: bool = True,
    default_interval: int = 20,
    key_prefix: str = "live_sync"
) -> None:
    """
    Composant de synchronisation en direct avec le planificateur.

    Le contrôle-check (sync_technique_page / sync_analytical_page) doit
    déjà avoir été exécuté PLUS HAUT dans le script, avant tout autre
    traitement dépendant de st.session_state — son résultat est passé
    ici uniquement pour affichage (séparation logique/affichage). Cette
    fonction se contente de :
        1. proposer d'activer/désactiver l'auto-rafraîchissement et
           d'en régler l'intervalle ;
        2. déclencher ce rafraîchissement (soft rerun) si activé ;
        3. signaler visuellement qu'un nouveau run vient d'être chargé.
    """

    current_run_id = get_active_run_id()

    col1, col2 = st.columns([1, 2])

    with col1:
        enabled = st.checkbox(
            "Synchronisation automatique",
            value=st.session_state.get(f"{key_prefix}_enabled", default_enabled),
            key=f"{key_prefix}_enabled",
            help=(
                "Vérifie régulièrement si le planificateur (processus "
                "indépendant) a produit un nouveau run, et le charge "
                "automatiquement — sans recharger la page."
            )
        )

    with col2:
        interval = st.slider(
            "Intervalle de vérification (secondes)",
            min_value=5,
            max_value=120,
            value=st.session_state.get(f"{key_prefix}_interval", default_interval),
            key=f"{key_prefix}_interval",
            disabled=not enabled
        )

    if enabled:
        _trigger_soft_autorefresh(interval, key=f"{key_prefix}_autorefresh")

    st.caption(
        f"Run actuellement affiché : **{current_run_id or 'aucun'}** — "
        f"dernière vérification à {datetime.now().strftime('%H:%M:%S')}"
        + (" (auto)" if enabled else " (manuelle, à chaque interaction)")
    )

    if sync_result is not None:
        st.toast(
            f"Nouveau run détecté : {sync_result['run_id']}",
            icon="🔄"
        )
        st.success(
            f"🔄 Nouveau résultat chargé automatiquement — {label} à jour : "
            f"run **{sync_result['run_id']}** (produit le "
            f"{sync_result.get('saved_at', '?')}), sans recharger la page."
        )


def render_scheduler_panel(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """
    Affiche le statut du planificateur système : configuration trouvée
    ou non, prochaines exécutions programmées, dernier run connu,
    historique récent. Cette page ne fait que LIRE l'état écrit sur
    disque par le processus scheduler_service.py — elle ne le démarre
    jamais depuis Streamlit (un planificateur doit rester actif même
    quand le dashboard est fermé, donc il tourne à part).
    """

    status = get_scheduler_status(config_path)

    if not status.get("config_found"):
        st.info(
            "Aucun planificateur configuré. Copiez "
            "`scheduler_config.example.json` vers `scheduler_config.json` "
            "à la racine du projet, puis lancez-le dans un terminal séparé : "
            "`python -m src.scheduler_service`."
        )
        return

    if status.get("config_error"):
        st.error(f"Erreur de configuration du planificateur : {status['config_error']}")
        return

    state = status.get("state", {})
    next_fire_times = status.get("next_fire_times", {})
    data_trigger = status.get("data_change_trigger", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dernier statut",
            state.get("last_run_status") or "aucun run pour l'instant"
        )

    with col2:
        st.metric(
            "Dernier run",
            state.get("last_run_id") or "-"
        )

    with col3:
        st.metric(
            "Dernière exécution",
            state.get("last_run_at") or "-"
        )

    col_a, col_b = st.columns(2)

    with col_a:
        card(
            "Prochaine exécution planifiée",
            (
                f"<b>Hebdomadaire</b> : {next_fire_times.get('hebdomadaire', 'désactivé')}<br>"
                f"<b>Mensuelle</b> : {next_fire_times.get('mensuel', 'désactivé')}"
            )
        )

    with col_b:
        if data_trigger.get("enabled"):
            body = (
                f"Vérification toutes les "
                f"{data_trigger.get('check_interval_minutes', 30)} minutes.<br>"
                f"Déclenchement dès que le volume de données augmente de "
                f"plus de <b>{data_trigger.get('threshold_pct', 10)}%</b> "
                f"par rapport au dernier run réussi "
                f"({state.get('last_row_count') or '?'} lignes actuellement)."
            )
            tone = "success"
        else:
            body = "Le déclenchement par volume de nouvelles données est désactivé."
            tone = "neutral"

        card("Déclenchement par volume de données", body)

    if state.get("last_run_status") == "failed":
        st.warning(
            "Le dernier run planifié a échoué. Consultez `logs/scheduler.log` "
            "pour le détail de l'erreur."
        )

    history = state.get("history", [])

    if history:
        with st.expander("Historique récent des exécutions planifiées"):
            st.dataframe(
                list(reversed(history)),
                width="stretch"
            )

    st.caption(
        "Le planificateur tourne dans un processus indépendant "
        "(`python -m src.scheduler_service`). Ce panneau ne fait "
        "qu'afficher son dernier état connu — fermer ce dashboard "
        "n'arrête pas le planificateur, et inversement."
    )