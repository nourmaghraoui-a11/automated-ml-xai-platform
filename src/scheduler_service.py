# ============================================================
# scheduler_service.py
# Planificateur système : déclenche automatiquement le pipeline
# complet (préparation + ML + XAI) selon 2 familles de règles :
#
#   1. Temporel (cron)      : ex. tous les dimanches à 2h, ou le
#                             1er de chaque mois à 3h.
#   2. Volumétrique         : dès que le volume de données a
#                             augmenté de plus de X% (ex. 10%) par
#                             rapport au dernier run réussi.
#
# Ce module est INDEPENDANT de Streamlit : il tourne comme un
# PROCESSUS SEPARE (terminal dédié, service systemd, tâche planifiée
# Windows...). Le dashboard Streamlit ne fait que LIRE l'état qu'il
# écrit sur disque (models/scheduler_state.json) via
# get_scheduler_status() — il n'a pas besoin que ce processus tourne
# pour afficher le dernier statut connu.
#
# Lancement :
#   python -m src.scheduler_service
# (depuis la racine du projet, avec scheduler_config.json à côté)
# ============================================================

import os
import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .ingestion import load_data, load_multiple_sources
from .full_pipeline_runner import run_full_pipeline


DEFAULT_CONFIG_PATH = "scheduler_config.json"
DEFAULT_STATE_PATH = os.path.join("models", "scheduler_state.json")
DEFAULT_LOG_PATH = os.path.join("logs", "scheduler.log")


# ============================================================
# 1. CONFIGURATION
# ============================================================

def load_scheduler_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Charge la configuration du planificateur (source de données,
    règles de déclenchement, paramètres ML). Voir
    scheduler_config.example.json pour un modèle commenté.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Fichier de configuration du planificateur introuvable : "
            f"{config_path}. Copiez scheduler_config.example.json vers "
            f"scheduler_config.json et adaptez-le à votre source de données."
        )

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 2. ETAT PERSISTANT (comparaison de volume + suivi des runs)
# ============================================================

def _load_state(state_path: str = DEFAULT_STATE_PATH) -> Dict[str, Any]:
    if not os.path.exists(state_path):
        return {
            "last_run_id": None,
            "last_run_at": None,
            "last_run_status": None,
            "last_row_count": None,
            "last_trigger_reason": None,
            "history": []
        }

    with open(state_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_state(state: Dict[str, Any], state_path: str = DEFAULT_STATE_PATH) -> None:
    directory = os.path.dirname(state_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(state_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4, ensure_ascii=False)


def reset_scheduler_state(state_path: str = DEFAULT_STATE_PATH) -> None:
    """Supprime l'état connu (utile après un changement de source)."""

    if os.path.exists(state_path):
        os.remove(state_path)


# ============================================================
# 3. CHARGEMENT DE LA SOURCE CONFIGUREE
# ============================================================

def _load_source(source_config: Dict[str, Any]):
    """
    Charge la donnée source configurée pour le planificateur.
    Supporte les mêmes formats qu'ingestion.py :
        - {"type": "csv", "path": "..."}
        - {"type": "sql", "path": "...", "query": "..."}
        - {"sources": [ {...}, {...} ]}  (sources multiples)
    """

    if "sources" in source_config:
        return load_multiple_sources(source_config["sources"])

    return load_data(
        source_type=source_config.get("type", "csv"),
        source_path=source_config["path"],
        query=source_config.get("query"),
        sep=source_config.get("sep", ","),
        encoding=source_config.get("encoding", "utf-8")
    )


def _count_rows(data) -> int:
    if isinstance(data, dict):
        return sum(len(df) for df in data.values())

    return len(data)


# ============================================================
# 4. EXECUTION D'UN RUN (planifié ou déclenché par le volume)
# ============================================================

def _run_job(config: Dict[str, Any], reason: str, logger: logging.Logger) -> None:
    """
    Exécute le pipeline complet et met à jour l'état.

    Ne lève JAMAIS d'exception : un job planifié qui échoue ne doit
    jamais arrêter le planificateur, seulement être journalisé (log +
    scheduler_state.json), afin que les prochaines exécutions restent
    programmées.
    """

    state_path = config.get("state_path", DEFAULT_STATE_PATH)
    state = _load_state(state_path)

    logger.info(f"Déclenchement du pipeline — raison : {reason}")

    try:
        data = _load_source(config["source"])
        row_count = _count_rows(data)

        result = run_full_pipeline(
            data=data,
            integration_config=config.get("integration_config"),
            manual_features_to_keep=config.get("manual_features_to_keep"),
            allow_high_risk_features=config.get("allow_high_risk_features", False),
            ml_config=config.get("ml_config"),
            interpretation_config=config.get("interpretation_config"),
            models_base_dir=config.get("models_base_dir", "models"),
            reports_base_dir=config.get("reports_base_dir", "reports"),
        )

        state.update({
            "last_run_id": result["run_id"],
            "last_run_at": result["ended_at"],
            "last_run_status": "success",
            "last_row_count": row_count,
            "last_trigger_reason": reason,
        })

        state.setdefault("history", []).append({
            "run_id": result["run_id"],
            "ended_at": result["ended_at"],
            "status": "success",
            "reason": reason,
            "row_count": row_count
        })

        logger.info(
            f"Run {result['run_id']} terminé avec succès "
            f"({row_count} lignes, raison : {reason})."
        )

    except Exception as error:
        state.update({
            "last_run_status": "failed",
            "last_trigger_reason": reason,
        })

        state.setdefault("history", []).append({
            "run_id": None,
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "failed",
            "reason": reason,
            "error": str(error)
        })

        logger.error(
            f"Échec du pipeline (raison : {reason}) : {error}\n"
            f"{traceback.format_exc()}"
        )

    # Historique borné pour ne pas grossir indéfiniment.
    state["history"] = state.get("history", [])[-50:]

    _save_state(state, state_path)


def _check_new_data_job(config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Vérifie si le volume de données a augmenté de plus de
    `threshold_pct` par rapport au dernier run réussi. Si aucune référence
    n'existe encore, le volume courant est enregistré comme référence sans
    déclencher le pipeline.
    """

    trigger_config = config.get("data_change_trigger", {})

    if not trigger_config.get("enabled", False):
        return

    state_path = config.get("state_path", DEFAULT_STATE_PATH)
    state = _load_state(state_path)

    last_row_count = state.get("last_row_count")

    try:
        data = _load_source(config["source"])
        current_row_count = _count_rows(data)
    except Exception as error:
        logger.error(f"Impossible de vérifier le volume de données : {error}")
        return

    if last_row_count is None:
        logger.info(
            "Aucune référence de volume précédente : enregistrement du "
            "volume courant comme référence, pipeline non déclenché."
        )
        state["last_row_count"] = current_row_count
        _save_state(state, state_path)
        return

    if last_row_count == 0:
        logger.warning(
            "Référence de volume précédente invalide (0 lignes) : "
            "enregistrement du volume courant comme référence, pipeline non déclenché."
        )
        state["last_row_count"] = current_row_count
        _save_state(state, state_path)
        return

    growth_pct = ((current_row_count - last_row_count) / last_row_count) * 100
    threshold_pct = trigger_config.get("threshold_pct", 10)

    logger.info(
        f"Vérification volumétrique : {current_row_count} lignes "
        f"({growth_pct:+.2f}% vs dernier run de {last_row_count} lignes, "
        f"seuil = {threshold_pct}%)."
    )

    if growth_pct >= threshold_pct:
        _run_job(
            config,
            reason=f"nouvelles_donnees_{round(growth_pct, 1)}pct",
            logger=logger
        )


# ============================================================
# 5. LOGGER
# ============================================================

def _build_logger(log_path: str = DEFAULT_LOG_PATH) -> logging.Logger:
    directory = os.path.dirname(log_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    logger = logging.getLogger("pipeline_scheduler")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


# ============================================================
# 6. CONSTRUCTION DU PLANIFICATEUR
# ============================================================

def build_scheduler(config_path: str = DEFAULT_CONFIG_PATH) -> BlockingScheduler:
    """
    Construit (sans le démarrer) le planificateur configuré avec :
        - un job hebdomadaire (ex : tous les dimanches)
        - un job mensuel (ex : le 1er de chaque mois)
        - un job périodique de vérification du volume de données
    """

    config = load_scheduler_config(config_path)
    logger = _build_logger(config.get("log_path", DEFAULT_LOG_PATH))

    scheduler = BlockingScheduler(timezone=config.get("timezone", "UTC"))

    schedule_config = config.get("schedule", {})

    weekly = schedule_config.get("weekly", {})
    if weekly.get("enabled", False):
        scheduler.add_job(
            _run_job,
            trigger=CronTrigger(
                day_of_week=weekly.get("day_of_week", "sun"),
                hour=weekly.get("hour", 2),
                minute=weekly.get("minute", 0)
            ),
            args=[config, "hebdomadaire", logger],
            id="weekly_run",
            replace_existing=True
        )
        logger.info("Job hebdomadaire programmé.")

    monthly = schedule_config.get("monthly", {})
    if monthly.get("enabled", False):
        scheduler.add_job(
            _run_job,
            trigger=CronTrigger(
                day=monthly.get("day", 1),
                hour=monthly.get("hour", 3),
                minute=monthly.get("minute", 0)
            ),
            args=[config, "mensuel", logger],
            id="monthly_run",
            replace_existing=True
        )
        logger.info("Job mensuel programmé.")

    data_trigger = config.get("data_change_trigger", {})
    if data_trigger.get("enabled", False):
        scheduler.add_job(
            _check_new_data_job,
            trigger=IntervalTrigger(
                minutes=data_trigger.get("check_interval_minutes", 30)
            ),
            args=[config, logger],
            id="data_change_check",
            replace_existing=True
        )
        logger.info(
            "Vérification volumétrique programmée toutes les "
            f"{data_trigger.get('check_interval_minutes', 30)} minutes "
            f"(seuil {data_trigger.get('threshold_pct', 10)}%)."
        )

    if not scheduler.get_jobs():
        logger.warning(
            "Aucune règle de déclenchement n'est activée dans "
            "scheduler_config.json (schedule.weekly / schedule.monthly / "
            "data_change_trigger sont tous désactivés)."
        )

    return scheduler


def start_scheduler(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """
    Point d'entrée bloquant : à lancer dans un processus dédié
    (terminal séparé, service systemd, tâche planifiée Windows...).
    Ce processus doit rester actif pour que les jobs se déclenchent.
    """

    scheduler = build_scheduler(config_path)
    scheduler.print_jobs()
    scheduler.start()


# ============================================================
# 7. STATUT POUR AFFICHAGE (lu depuis un autre processus)
# ============================================================

def _describe_weekly(weekly: Dict[str, Any]) -> str:
    return (
        f"chaque semaine — {weekly.get('day_of_week', 'sun')} à "
        f"{int(weekly.get('hour', 2)):02d}:{int(weekly.get('minute', 0)):02d}"
    )


def _describe_monthly(monthly: Dict[str, Any]) -> str:
    return (
        f"chaque mois — jour {monthly.get('day', 1)} à "
        f"{int(monthly.get('hour', 3)):02d}:{int(monthly.get('minute', 0)):02d}"
    )


def get_scheduler_status(
    config_path: str = DEFAULT_CONFIG_PATH,
    state_path: str = DEFAULT_STATE_PATH
) -> Dict[str, Any]:
    """
    Ne démarre PAS le planificateur : lit uniquement la configuration
    et l'état déjà écrits sur disque. Conçu pour être appelé depuis
    Streamlit (processus différent de celui du planificateur, qui
    tourne en arrière-plan) afin d'afficher un statut sans dépendre
    d'un objet scheduler partagé entre processus.
    """

    status: Dict[str, Any] = {
        "config_found": os.path.exists(config_path),
        "state": _load_state(state_path),
        "next_fire_times": {},
        "data_change_trigger": {}
    }

    if not status["config_found"]:
        return status

    try:
        config = load_scheduler_config(config_path)
    except Exception as error:
        status["config_error"] = str(error)
        return status

    schedule_config = config.get("schedule", {})
    now = datetime.now()

    weekly = schedule_config.get("weekly", {})
    if weekly.get("enabled", False):
        try:
            trigger = CronTrigger(
                day_of_week=weekly.get("day_of_week", "sun"),
                hour=weekly.get("hour", 2),
                minute=weekly.get("minute", 0)
            )
            next_fire = trigger.get_next_fire_time(None, now.astimezone())
            status["next_fire_times"]["hebdomadaire"] = str(next_fire)
        except Exception:
            status["next_fire_times"]["hebdomadaire"] = _describe_weekly(weekly)

    monthly = schedule_config.get("monthly", {})
    if monthly.get("enabled", False):
        try:
            trigger = CronTrigger(
                day=monthly.get("day", 1),
                hour=monthly.get("hour", 3),
                minute=monthly.get("minute", 0)
            )
            next_fire = trigger.get_next_fire_time(None, now.astimezone())
            status["next_fire_times"]["mensuel"] = str(next_fire)
        except Exception:
            status["next_fire_times"]["mensuel"] = _describe_monthly(monthly)

    status["data_change_trigger"] = config.get("data_change_trigger", {})

    return status


if __name__ == "__main__":
    start_scheduler()