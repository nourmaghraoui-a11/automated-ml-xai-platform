# ============================================================
# artifacts_manager.py
# Sprint 1 : Gestion et sauvegarde des artefacts du pipeline
# ============================================================

import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, List


# ============================================================
# 1. OUTILS GENERAUX
# ============================================================

def create_directory(path: str) -> None:
    """
    Crée un dossier s'il n'existe pas.
    """

    if path:
        os.makedirs(path, exist_ok=True)


def generate_run_id(prefix: str = "run") -> str:
    """
    Génère un identifiant unique pour une exécution du pipeline.

    Exemple :
        run_20260706_113500
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def get_file_size_kb(path: str) -> Optional[float]:
    """
    Retourne la taille d'un fichier en Ko.
    """

    if not os.path.exists(path):
        return None

    return round(os.path.getsize(path) / 1024, 2)


# ============================================================
# 2. SAUVEGARDE / CHARGEMENT JOBLIB
# ============================================================

def save_joblib_object(obj: Any, path: str) -> str:
    """
    Sauvegarde un objet Python avec joblib.

    Args:
        obj: Objet à sauvegarder.
        path: Chemin de sauvegarde.

    Returns:
        str: Chemin de sauvegarde.
    """

    directory = os.path.dirname(path)
    create_directory(directory)

    joblib.dump(obj, path)

    return path


def load_joblib_object(path: str) -> Any:
    """
    Charge un objet sauvegardé avec joblib.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Artefact introuvable : {path}")

    return joblib.load(path)


# ============================================================
# 3. SAUVEGARDE / CHARGEMENT JSON
# ============================================================

def save_json(data: Dict[str, Any], path: str) -> str:
    """
    Sauvegarde un dictionnaire en JSON.
    """

    directory = os.path.dirname(path)
    create_directory(directory)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False, default=str)

    return path


def load_json(path: str) -> Dict[str, Any]:
    """
    Charge un fichier JSON.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier JSON introuvable : {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 4. SAUVEGARDE DES ARTEFACTS DU PIPELINE
# ============================================================

def save_pipeline_artifacts(
    artifacts: Dict[str, Any],
    base_dir: str = "models",
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sauvegarde les artefacts du pipeline dans un dossier dédié.

    Args:
        artifacts: Dictionnaire contenant les artefacts du pipeline.
        base_dir: Dossier principal de sauvegarde.
        run_id: Identifiant de l'exécution. Si None, il est généré automatiquement.

    Returns:
        dict: Métadonnées des artefacts sauvegardés.
    """

    if run_id is None:
        run_id = generate_run_id()

    run_dir = os.path.join(base_dir, run_id)
    create_directory(run_dir)

    artifacts_path = os.path.join(run_dir, "pipeline_artifacts.pkl")
    metadata_path = os.path.join(run_dir, "metadata.json")

    save_joblib_object(artifacts, artifacts_path)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "artifacts_path": artifacts_path,
        "metadata_path": metadata_path,
        "file_size_kb": get_file_size_kb(artifacts_path),
        "available_keys": list(artifacts.keys())
    }

    save_json(metadata, metadata_path)

    return metadata


def load_pipeline_artifacts(
    run_id: Optional[str] = None,
    base_dir: str = "models",
    artifacts_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Charge les artefacts du pipeline.

    Deux modes possibles :
        1. Charger par run_id
        2. Charger directement par artifacts_path

    Args:
        run_id: Identifiant de l'exécution.
        base_dir: Dossier principal.
        artifacts_path: Chemin direct vers pipeline_artifacts.pkl.

    Returns:
        dict: Artefacts chargés.
    """

    if artifacts_path is None:

        if run_id is None:
            raise ValueError("run_id ou artifacts_path doit être fourni.")

        artifacts_path = os.path.join(
            base_dir,
            run_id,
            "pipeline_artifacts.pkl"
        )

    return load_joblib_object(artifacts_path)


# ============================================================
# 5. SAUVEGARDE DES RAPPORTS
# ============================================================

def save_report(
    report: Dict[str, Any],
    report_name: str,
    base_dir: str = "reports",
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sauvegarde un rapport JSON.

    Args:
        report: Rapport à sauvegarder.
        report_name: Nom du rapport sans extension.
        base_dir: Dossier des rapports.
        run_id: Identifiant d'exécution.

    Returns:
        dict: Métadonnées du rapport.
    """

    if run_id is None:
        run_id = generate_run_id()

    report_dir = os.path.join(base_dir, run_id)
    create_directory(report_dir)

    report_path = os.path.join(report_dir, f"{report_name}.json")

    save_json(report, report_path)

    metadata = {
        "run_id": run_id,
        "report_name": report_name,
        "report_path": report_path,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size_kb": get_file_size_kb(report_path)
    }

    return metadata


def load_report(path: str) -> Dict[str, Any]:
    """
    Charge un rapport JSON.
    """

    return load_json(path)


# ============================================================
# 6. REGISTRE DES ARTEFACTS
# ============================================================

def update_artifact_registry(
    metadata: Dict[str, Any],
    registry_path: str = "models/artifacts_registry.json"
) -> Dict[str, Any]:
    """
    Met à jour le registre global des artefacts.

    Le registre garde l'historique des exécutions et des artefacts sauvegardés.
    """

    directory = os.path.dirname(registry_path)
    create_directory(directory)

    if os.path.exists(registry_path):
        registry = load_json(registry_path)
    else:
        registry = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "runs": []
        }

    registry["runs"].append(metadata)
    registry["last_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_json(registry, registry_path)

    return registry


def load_artifact_registry(
    registry_path: str = "models/artifacts_registry.json"
) -> Dict[str, Any]:
    """
    Charge le registre des artefacts.
    """

    if not os.path.exists(registry_path):
        return {
            "created_at": None,
            "last_updated_at": None,
            "runs": []
        }

    return load_json(registry_path)


def get_last_run_id(
    registry_path: str = "models/artifacts_registry.json"
) -> Optional[str]:
    """
    Récupère le dernier run_id sauvegardé.
    """

    registry = load_artifact_registry(registry_path)

    runs = registry.get("runs", [])

    if not runs:
        return None

    return runs[-1].get("run_id")


def get_last_artifacts_path(
    registry_path: str = "models/artifacts_registry.json"
) -> Optional[str]:
    """
    Récupère le chemin des derniers artefacts sauvegardés.
    """

    registry = load_artifact_registry(registry_path)

    runs = registry.get("runs", [])

    if not runs:
        return None

    return runs[-1].get("artifacts_path")


# ============================================================
# 7. SAUVEGARDE COMPLETE D'UNE EXECUTION SPRINT 1
# ============================================================

def save_data_preparation_outputs(
    artifacts: Dict[str, Any],
    structure_report: Optional[Dict[str, Any]] = None,
    preprocessing_report: Optional[Dict[str, Any]] = None,
    base_models_dir: str = "models",
    base_reports_dir: str = "reports",
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sauvegarde complète des sorties du Sprint 1 :
        - artefacts preprocessing
        - rapport de structure
        - rapport preprocessing
        - registre des artefacts

    Args:
        artifacts: Artefacts du pipeline.
        structure_report: Rapport de structure.
        preprocessing_report: Rapport de preprocessing.
        base_models_dir: Dossier modèles.
        base_reports_dir: Dossier rapports.
        run_id: Identifiant d'exécution.

    Returns:
        dict: Métadonnées complètes de sauvegarde.
    """

    if run_id is None:
        run_id = generate_run_id()

    artifact_metadata = save_pipeline_artifacts(
        artifacts=artifacts,
        base_dir=base_models_dir,
        run_id=run_id
    )

    report_metadata = {}

    if structure_report is not None:
        report_metadata["structure_report"] = save_report(
            report=structure_report,
            report_name="structure_report",
            base_dir=base_reports_dir,
            run_id=run_id
        )

    if preprocessing_report is not None:
        report_metadata["preprocessing_report"] = save_report(
            report=preprocessing_report,
            report_name="preprocessing_report",
            base_dir=base_reports_dir,
            run_id=run_id
        )

    full_metadata = {
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_metadata": artifact_metadata,
        "report_metadata": report_metadata
    }

    update_artifact_registry(full_metadata)

    return full_metadata


# ============================================================
# 8. NETTOYAGE / POLITIQUE DE RETENTION
# ============================================================

def list_saved_runs(base_dir: str = "models") -> List["str"]:
    """
    Liste les exécutions sauvegardées dans le dossier models.
    """

    if not os.path.exists(base_dir):
        return []

    runs = [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]

    runs = sorted(runs)

    return runs


def apply_retention_policy(
    base_dir: str = "models",
    keep_last: int = 5
) -> List["str"]:
    """
    Supprime les anciennes exécutions et garde seulement les N dernières.

    Args:
        base_dir: Dossier contenant les runs.
        keep_last: Nombre de runs à conserver.

    Returns:
        list: runs supprimés.
    """

    runs = list_saved_runs(base_dir)

    if len(runs) <= keep_last:
        return []

    runs_to_delete = runs[:-keep_last]
    deleted_runs = []

    for run_id in runs_to_delete:
        run_dir = os.path.join(base_dir, run_id)

        try:
            for root, dirs, files in os.walk(run_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))

                for folder in dirs:
                    os.rmdir(os.path.join(root, folder))

            os.rmdir(run_dir)
            deleted_runs.append(run_id)

        except Exception:
            continue

    return deleted_runs


# ============================================================
# 9. RAPPORT SUR LES ARTEFACTS
# ============================================================

def generate_artifacts_status_report(
    base_dir: str = "models",
    registry_path: str = "models/artifacts_registry.json"
) -> Dict[str, Any]:
    """
    Génère un rapport d'état des artefacts sauvegardés.
    """

    runs = list_saved_runs(base_dir)
    registry = load_artifact_registry(registry_path)

    last_run_id = get_last_run_id(registry_path)
    last_artifacts_path = get_last_artifacts_path(registry_path)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": base_dir,
        "n_saved_runs": len(runs),
        "saved_runs": runs,
        "last_run_id": last_run_id,
        "last_artifacts_path": last_artifacts_path,
        "last_artifacts_exist": (
            os.path.exists(last_artifacts_path)
            if last_artifacts_path is not None
            else False
        ),
        "registry_path": registry_path,
        "registry_runs_count": len(registry.get("runs", []))
    }

    return report


# ============================================================
# 10. TEST LOCAL RAPIDE
# ============================================================

if __name__ == "__main__":

    artifacts_test = {
        "structure": {
            "numeric_columns": ["age"],
            "categorical_columns": ["country"],
            "date_columns": ["signup_date"],
            "id_columns": ["user_id"],
            "possible_label_columns": ["churn"]
        },
        "final_features": [
            "age",
            "country",
            "signup_date_year",
            "signup_date_month"
        ],
        "scaler_name": "StandardScaler",
        "excluded_columns": ["user_id", "churn"]
    }

    structure_report_test = {
        "summary": {
            "nb_numeric": 1,
            "nb_categorical": 1,
            "nb_dates": 1,
            "nb_ids": 1,
            "nb_possible_labels": 1
        }
    }

    preprocessing_report_test = {
        "pipeline_summary": {
            "initial_shape": [5, 5],
            "final_shape": [5, 4],
            "scaler_used": "StandardScaler"
        }
    }

    metadata = save_data_preparation_outputs(
        artifacts=artifacts_test,
        structure_report=structure_report_test,
        preprocessing_report=preprocessing_report_test,
        base_models_dir="models",
        base_reports_dir="reports"
    )

    print("===== METADATA SAUVEGARDE =====")
    print(metadata)

    status_report = generate_artifacts_status_report()

    print("\n===== STATUS ARTEFACTS =====")
    print(status_report)