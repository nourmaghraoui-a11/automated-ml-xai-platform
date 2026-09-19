# ============================================================
# full_pipeline_runner.py
# Orchestration bout-en-bout : préparation -> ML -> XAI -> persistance
#
# Utilisé par :
#   - scheduler_service.py (exécutions automatiques planifiées)
#   - un bouton "Tout lancer" optionnel côté dashboard
#   - un usage en ligne de commande (tests, cron externe, etc.)
# ============================================================

from datetime import datetime
from typing import Any, Dict, Optional, Union

import pandas as pd

from .pipeline_controller import start_pipeline
from .ml_controller import (
    run_ml_pipeline,
    attach_cluster_labels,
    attach_anomaly_results
)
from .interpretation import generate_full_interpretation
from .artifacts_manager import generate_run_id
from .session_store import save_full_session


def run_full_pipeline(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    integration_config: Optional[Dict[str, Any]] = None,
    manual_features_to_keep: Optional[list] = None,
    allow_high_risk_features: bool = False,
    ml_config: Optional[Dict[str, Any]] = None,
    interpretation_config: Optional[Dict[str, Any]] = None,
    models_base_dir: str = "models",
    reports_base_dir: str = "reports",
    run_id: Optional[str] = None,
    persist: bool = True
) -> Dict[str, Any]:
    """
    Exécute l'intégralité du pipeline (préparation des données,
    segmentation, détection d'anomalies, explicabilité) et persiste
    un instantané complet sur disque via session_store, afin qu'il
    survive à un redémarrage de Streamlit ou du terminal.

    Args:
        data: DataFrame unique ou dictionnaire de sources (comme
            start_pipeline). C'est typiquement le planificateur qui
            recharge cette donnée depuis la source configurée avant
            d'appeler cette fonction.
        ml_config: kwargs transmis à run_ml_pipeline (enable_clustering,
            contamination, etc.).
        interpretation_config: kwargs transmis à generate_full_interpretation.
        persist: si False, n'écrit pas l'instantané sur disque (utile
            pour des tests unitaires).

    Returns:
        dict: résultat complet (prep_result, ml_result, clustered_dataset,
        anomaly_dataset, interpretation) + quelques métadonnées.
    """

    ml_config = dict(ml_config or {})
    interpretation_config = dict(interpretation_config or {})

    if run_id is None:
        run_id = generate_run_id(prefix="auto")

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------
    # 1. Préparation des données
    # ------------------------------------------------------------

    prep_result = start_pipeline(
        data=data,
        integration_config=integration_config,
        manual_features_to_keep=manual_features_to_keep,
        allow_high_risk_features=allow_high_risk_features,
        models_base_dir=models_base_dir,
        reports_base_dir=reports_base_dir,
        run_id=run_id
    )

    X_ready = prep_result["X_ready"]
    artifacts_path = prep_result["preprocessing_info"].get("artifacts_saved_at")

    # ------------------------------------------------------------
    # 2. Machine Learning (segmentation + anomalies)
    # ------------------------------------------------------------

    ml_result = run_ml_pipeline(
        X_ready=X_ready,
        run_id=run_id,
        models_base_dir=models_base_dir,
        reports_base_dir=reports_base_dir,
        **ml_config
    )

    clustered_dataset = None
    anomaly_dataset = None

    if ml_result.get("cluster_labels") is not None:
        clustered_dataset = attach_cluster_labels(
            X_ready=X_ready,
            labels=ml_result["cluster_labels"]
        )

    if ml_result.get("anomaly_labels") is not None:
        anomaly_dataset = attach_anomaly_results(
            X_ready=X_ready,
            anomaly_labels=ml_result["anomaly_labels"],
            anomaly_scores=ml_result["anomaly_scores"]
        )

    # ------------------------------------------------------------
    # 3. Explicabilité (XAI)
    # ------------------------------------------------------------

    interpretation = generate_full_interpretation(
        ml_result=ml_result,
        clustered_dataset=clustered_dataset,
        anomaly_dataset=anomaly_dataset,
        artifacts_path=artifacts_path,
        **interpretation_config
    )

    # ------------------------------------------------------------
    # 4. Persistance complète (survit aux redémarrages)
    # ------------------------------------------------------------

    if persist:
        save_full_session(
            run_id=run_id,
            prep_result=prep_result,
            ml_result=ml_result,
            clustered_dataset=clustered_dataset,
            anomaly_dataset=anomaly_dataset,
            interpretation=interpretation,
        )

    ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "prep_result": prep_result,
        "ml_result": ml_result,
        "clustered_dataset": clustered_dataset,
        "anomaly_dataset": anomaly_dataset,
        "interpretation": interpretation,
        "input_rows": int(X_ready.shape[0]),
        "input_columns": int(X_ready.shape[1]),
    }