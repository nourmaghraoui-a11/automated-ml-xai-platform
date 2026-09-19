# ============================================================
# pipeline_controller.py
# Orchestration du module de préparation des données
# ============================================================

import os
from datetime import datetime
from typing import Dict, Any, Optional, Union

import pandas as pd

from .ingestion import load_data, load_multiple_sources
from .data_integration import build_unified_dataset, get_integration_report
from .structure_detection import detect_structure, generate_structure_summary
from .preprocessing import run_preprocessing_pipeline, transform_pipeline

from .reports import (
    generate_structure_report,
    generate_preprocessing_report,
    generate_data_preparation_report,
    safe_json_dump
)

from .artifacts_manager import (
    generate_run_id,
    update_artifact_registry,
    generate_artifacts_status_report
)

from .warehouse import save_data_preparation_to_warehouse


# ============================================================
# 1. PREPARATION DES DOSSIERS D'EXECUTION
# ============================================================

def create_run_directories(
    run_id: str,
    models_base_dir: str = "models",
    reports_base_dir: str = "reports"
) -> Dict[str, str]:
    """
    Crée les dossiers associés à une exécution du module de préparation.
    """

    models_run_dir = os.path.join(models_base_dir, run_id)
    reports_run_dir = os.path.join(reports_base_dir, run_id)

    os.makedirs(models_run_dir, exist_ok=True)
    os.makedirs(reports_run_dir, exist_ok=True)

    return {
        "models_run_dir": models_run_dir,
        "reports_run_dir": reports_run_dir,
        "artifacts_path": os.path.join(models_run_dir, "pipeline_artifacts.pkl"),
        "data_preparation_report_path": os.path.join(
            reports_run_dir,
            "data_preparation_report.json"
        ),
        "structure_report_path": os.path.join(
            reports_run_dir,
            "structure_report.json"
        ),
        "preprocessing_report_path": os.path.join(
            reports_run_dir,
            "preprocessing_report.json"
        ),
        "integration_report_path": os.path.join(
            reports_run_dir,
            "integration_report.json"
        )
    }


# ============================================================
# 2. PREPARATION DU DATASET FINAL
# ============================================================

def prepare_dataset(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    integration_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prépare le dataset final avant la détection de structure.

    Cas 1 :
        DataFrame unique vers standardisation simple.

    Cas 2 :
        Dictionnaire de DataFrames vers intégration multi-source.
    """

    integration_config = integration_config or {}

    dataset_final = build_unified_dataset(
        dataframes=data,
        config=integration_config
    )

    integration_report = get_integration_report(dataset_final)

    return {
        "dataset_final": dataset_final,
        "integration_report": integration_report
    }


# ============================================================
# 3. LANCEMENT DU MODULE DE PREPARATION DES DONNEES
# ============================================================

def start_pipeline(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    integration_config: Optional[Dict[str, Any]] = None,
    manual_features_to_keep: Optional[list] = None,
    allow_high_risk_features: bool = False,
    models_base_dir: str = "models",
    reports_base_dir: str = "reports",
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lance le module complet de préparation des données.

    Flux :
        1. Intégration et harmonisation
        2. Détection automatique de structure
        3. Preprocessing automatique
        4. Génération des rapports techniques
        5. Sauvegarde des artefacts
        6. Historisation dans le Data Warehouse

    Args:
        data:
            DataFrame unique ou dictionnaire de DataFrames.
        integration_config:
            Configuration d'intégration multi-source.
        manual_features_to_keep:
            Liste des colonnes exclues à réintégrer.
        allow_high_risk_features:
            Autorise la réintégration des colonnes risquées
            comme les identifiants ou labels potentiels.
        models_base_dir:
            Dossier racine des artefacts.
        reports_base_dir:
            Dossier racine des rapports.
        run_id:
            Identifiant manuel d'exécution.

    Returns:
        dict: Résultats complets du module de préparation.
    """

    if run_id is None:
        run_id = generate_run_id(prefix="data_prep")

    paths = create_run_directories(
        run_id=run_id,
        models_base_dir=models_base_dir,
        reports_base_dir=reports_base_dir
    )

    start_time = datetime.now()

    try:
        # --------------------------------------------------------
        # 1. Intégration et harmonisation
        # --------------------------------------------------------

        prepared = prepare_dataset(
            data=data,
            integration_config=integration_config
        )

        dataset_final = prepared["dataset_final"]
        integration_report = prepared["integration_report"]

        safe_json_dump(
            integration_report,
            paths["integration_report_path"]
        )

        # --------------------------------------------------------
        # 2. Détection automatique de structure
        # --------------------------------------------------------

        structure = detect_structure(dataset_final)
        structure_summary = generate_structure_summary(structure)

        structure_report = generate_structure_report(structure)

        safe_json_dump(
            structure_report,
            paths["structure_report_path"]
        )

        # --------------------------------------------------------
        # 3. Preprocessing automatique
        # --------------------------------------------------------

        X_ready, y, preprocessing_info, preprocessing_reports = run_preprocessing_pipeline(
            dataset_final,
            structure,
            mode="fit",
            manual_features_to_keep=manual_features_to_keep,
            allow_high_risk_features=allow_high_risk_features,
            artifacts_dir=paths["models_run_dir"]
        )

        # --------------------------------------------------------
        # 4. Rapport de preprocessing
        # --------------------------------------------------------

        preprocessing_report = generate_preprocessing_report(
            preprocessing_info=preprocessing_info,
            reports=preprocessing_reports
        )

        safe_json_dump(
            preprocessing_report,
            paths["preprocessing_report_path"]
        )

        # --------------------------------------------------------
        # 5. Rapport global du module de préparation
        # --------------------------------------------------------

        data_preparation_report = generate_data_preparation_report(
            structure=structure,
            preprocessing_info=preprocessing_info,
            reports=preprocessing_reports,
            output_path=paths["data_preparation_report_path"]
        )

        # --------------------------------------------------------
        # 6. Métadonnées d'exécution
        # --------------------------------------------------------

        end_time = datetime.now()
        duration_seconds = round((end_time - start_time).total_seconds(), 3)

        run_metadata = {
            "run_id": run_id,
            "created_at": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
            "status": "success",

            "input_shape": list(dataset_final.shape),
            "output_shape": list(X_ready.shape),

            "models_run_dir": paths["models_run_dir"],
            "reports_run_dir": paths["reports_run_dir"],
            "artifacts_path": preprocessing_info.get("artifacts_saved_at"),

            "structure_report_path": paths["structure_report_path"],
            "preprocessing_report_path": paths["preprocessing_report_path"],
            "integration_report_path": paths["integration_report_path"],
            "data_preparation_report_path": paths["data_preparation_report_path"],

            "label_column": preprocessing_info.get("label_column"),
            "excluded_columns": preprocessing_info.get("excluded_columns", []),
            "reintegrated_columns": preprocessing_info.get("reintegrated_columns", []),
            "manual_feature_warnings": preprocessing_info.get("manual_feature_warnings", []),
            "allow_high_risk_features": preprocessing_info.get("allow_high_risk_features", False),
            "final_features": preprocessing_info.get("final_features", []),
            "scaler_used": preprocessing_info.get("scaler_used")
        }

        update_artifact_registry(run_metadata)

        # --------------------------------------------------------
        # 7. Résultat final
        # --------------------------------------------------------

        result = {
            "run_id": run_id,
            "dataset_final": dataset_final,
            "X_ready": X_ready,
            "y": y,

            "structure": structure,
            "structure_summary": structure_summary,

            "integration_report": integration_report,
            "preprocessing_info": preprocessing_info,
            "preprocessing_reports": preprocessing_reports,

            "structure_report": structure_report,
            "preprocessing_report": preprocessing_report,
            "data_preparation_report": data_preparation_report,

            "run_metadata": run_metadata,
            "paths": paths
        }

        # --------------------------------------------------------
        # 8. Historisation dans le Data Warehouse
        # --------------------------------------------------------

        try:
            warehouse_status = save_data_preparation_to_warehouse(result)
            result["warehouse_status"] = warehouse_status

        except Exception as warehouse_error:
            result["warehouse_status"] = {
                "status": "failed",
                "error": str(warehouse_error)
            }

        print("Module de préparation des données terminé avec succès.")
        print(f"Run ID : {run_id}")
        print(f"Input shape : {dataset_final.shape}")
        print(f"Output shape : {X_ready.shape}")
        print(f"Artefacts : {preprocessing_info.get('artifacts_saved_at')}")
        print(f"Rapports : {paths['reports_run_dir']}")
        print(f"Colonnes réintégrées : {preprocessing_info.get('reintegrated_columns', [])}")
        print(f"Réintégration risquée autorisée : {preprocessing_info.get('allow_high_risk_features', False)}")

        if result.get("warehouse_status"):
            print(f"Data Warehouse : {result['warehouse_status']}")

        return result

    except Exception as e:
        end_time = datetime.now()
        duration_seconds = round((end_time - start_time).total_seconds(), 3)

        error_metadata = {
            "run_id": run_id,
            "created_at": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
            "status": "failed",
            "error": str(e),
            "models_run_dir": paths["models_run_dir"],
            "reports_run_dir": paths["reports_run_dir"]
        }

        safe_json_dump(
            error_metadata,
            os.path.join(paths["reports_run_dir"], "error_report.json")
        )

        update_artifact_registry(error_metadata)

        raise Exception(
            f"Erreur durant l'exécution du module de préparation des données : {e}"
        )


# ============================================================
# 4. LANCEMENT DEPUIS UNE SOURCE UNIQUE
# ============================================================

def start_pipeline_from_source(
    source_type: str,
    source_path: str,
    query: Optional[str] = None,
    integration_config: Optional[Dict[str, Any]] = None,
    manual_features_to_keep: Optional[list] = None,
    allow_high_risk_features: bool = False,
    models_base_dir: str = "models",
    reports_base_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Lance le module de préparation à partir d'une seule source :
        - CSV
        - SQL
    """

    df = load_data(
        source_type=source_type,
        source_path=source_path,
        query=query
    )

    return start_pipeline(
        data=df,
        integration_config=integration_config,
        manual_features_to_keep=manual_features_to_keep,
        allow_high_risk_features=allow_high_risk_features,
        models_base_dir=models_base_dir,
        reports_base_dir=reports_base_dir
    )


# ============================================================
# 5. LANCEMENT DEPUIS PLUSIEURS SOURCES
# ============================================================

def start_pipeline_from_multiple_sources(
    sources_config: list,
    integration_config: Optional[Dict[str, Any]] = None,
    manual_features_to_keep: Optional[list] = None,
    allow_high_risk_features: bool = False,
    models_base_dir: str = "models",
    reports_base_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Lance le module de préparation à partir de plusieurs sources.

    Exemple :
        sources_config = [
            {"name": "users", "type": "csv", "path": "data/raw/users.csv"},
            {"name": "transactions", "type": "csv", "path": "data/raw/transactions.csv"}
        ]
    """

    dataframes = load_multiple_sources(sources_config)

    return start_pipeline(
        data=dataframes,
        integration_config=integration_config,
        manual_features_to_keep=manual_features_to_keep,
        allow_high_risk_features=allow_high_risk_features,
        models_base_dir=models_base_dir,
        reports_base_dir=reports_base_dir
    )


# ============================================================
# 6. RELANCE AVEC FEATURES MODIFIEES
# ============================================================

def rerun_pipeline(
    dataset_final: pd.DataFrame,
    structure: Optional[Dict[str, Any]] = None,
    manual_features_to_keep: Optional[list] = None,
    allow_high_risk_features: bool = False,
    models_base_dir: str = "models",
    reports_base_dir: str = "reports",
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Relance le module de préparation après réintégration manuelle des features.
    """

    if structure is None:
        structure = detect_structure(dataset_final)

    return start_pipeline(
        data=dataset_final,
        integration_config=None,
        manual_features_to_keep=manual_features_to_keep,
        allow_high_risk_features=allow_high_risk_features,
        models_base_dir=models_base_dir,
        reports_base_dir=reports_base_dir,
        run_id=run_id
    )


# ============================================================
# 7. TRANSFORMATION DE NOUVELLES DONNEES
# ============================================================

def transform_new_data(
    df_new: pd.DataFrame,
    artifacts_path: str
) -> pd.DataFrame:
    """
    Transforme de nouvelles données avec les artefacts déjà appris.
    """

    X_new = transform_pipeline(
        df_new,
        artifacts_path=artifacts_path
    )

    return X_new


# ============================================================
# 8. RAPPORT D'ETAT DES ARTEFACTS
# ============================================================

def get_artifacts_status() -> Dict[str, Any]:
    """
    Retourne l'état des artefacts sauvegardés.
    """

    return generate_artifacts_status_report()