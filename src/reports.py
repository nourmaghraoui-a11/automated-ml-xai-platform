# ============================================================
# reports.py
# Sprint 1 : Génération des rapports techniques
# ============================================================
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd


# ============================================================
# 1. OUTILS DE CONVERSION
# ============================================================

def dataframe_to_records(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Convertit un DataFrame en liste de dictionnaires.
    Utile pour sauvegarder les rapports en JSON.
    """

    if df is None:
        return []

    if not isinstance(df, pd.DataFrame):
        return []

    if df.empty:
        return []

    return df.to_dict(orient="records")


def safe_json_dump(data: Dict[str, Any], path: str) -> None:
    """
    Sauvegarde un dictionnaire en JSON.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False, default=str)


def safe_json_load(path: str) -> Dict[str, Any]:
    """
    Charge un rapport JSON.
    """

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 2. RAPPORT DE STRUCTURE
# ============================================================

def generate_structure_report(structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère un rapport de structure détectée.

    Args:
        structure (dict): Résultat retourné par detect_structure().

    Returns:
        dict: Rapport de structure.
    """

    if structure is None:
        return {}

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "dataset_shape": {
            "n_rows": structure.get("n_rows", 0),
            "n_columns": structure.get("n_columns", 0)
        },

        "columns": structure.get("columns", []),
        "dtypes": structure.get("dtypes", {}),
        "missing_values": structure.get("missing_values", {}),

        "detected_structure": {
            "numeric_columns": structure.get("numeric_columns", []),
            "categorical_columns": structure.get("categorical_columns", []),
            "date_columns": structure.get("date_columns", []),
            "id_columns": structure.get("id_columns", []),
            "possible_label_columns": structure.get("possible_label_columns", []),
            "feature_columns": structure.get("feature_columns", []),
            "excluded_columns": structure.get("excluded_columns", [])
        },

        "possible_labels_info": structure.get("possible_labels_info", []),
        "exclusion_reasons": structure.get("exclusion_reasons", {}),

        "summary": {
            "nb_numeric": len(structure.get("numeric_columns", [])),
            "nb_categorical": len(structure.get("categorical_columns", [])),
            "nb_dates": len(structure.get("date_columns", [])),
            "nb_ids": len(structure.get("id_columns", [])),
            "nb_possible_labels": len(structure.get("possible_label_columns", [])),
            "nb_features": len(structure.get("feature_columns", [])),
            "nb_excluded": len(structure.get("excluded_columns", []))
        }
    }

    return report


# ============================================================
# 3. RAPPORT DES VALEURS MANQUANTES
# ============================================================

def generate_missing_report(missing_report: pd.DataFrame) -> Dict[str, Any]:
    """
    Génère un rapport des valeurs manquantes.

    Args:
        missing_report (DataFrame): Rapport retourné par detect_missing_values().

    Returns:
        dict: Rapport structuré.
    """

    records = dataframe_to_records(missing_report)

    total_columns_with_missing = len(records)

    max_missing_percentage = 0

    if records:
        max_missing_percentage = max(
            item.get("missing_percentage", 0)
            for item in records
        )

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_columns_with_missing": total_columns_with_missing,
        "max_missing_percentage": max_missing_percentage,
        "details": records
    }

    return report


# ============================================================
# 4. RAPPORT DES COLONNES EXCLUES
# ============================================================

def generate_excluded_features_report(
    excluded_columns: List[str],
    exclusion_reasons: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Génère un rapport des colonnes exclues.

    Args:
        excluded_columns (list): Colonnes exclues.
        exclusion_reasons (dict): Raisons d'exclusion.

    Returns:
        dict: Rapport des exclusions.
    """

    exclusion_reasons = exclusion_reasons or {}

    details = []

    for col in excluded_columns:
        details.append({
            "column": col,
            "reason": exclusion_reasons.get(
                col,
                "Colonne exclue automatiquement par le pipeline"
            )
        })

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_excluded_columns": len(excluded_columns),
        "excluded_columns": excluded_columns,
        "details": details
    }

    return report


# ============================================================
# 5. RAPPORT DE PREPROCESSING
# ============================================================

def generate_preprocessing_report(
    preprocessing_info: Dict[str, Any],
    reports: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Génère un rapport global du preprocessing.

    Args:
        preprocessing_info (dict): Infos retournées par fit_pipeline().
        reports (dict): Rapports produits pendant le preprocessing.

    Returns:
        dict: Rapport global.
    """

    if preprocessing_info is None:
        preprocessing_info = {}

    if reports is None:
        reports = {}

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "pipeline_summary": {
            "initial_shape": preprocessing_info.get("initial_shape"),
            "final_shape": preprocessing_info.get("final_shape"),
            "label_column": preprocessing_info.get("label_column"),
            "scaler_used": preprocessing_info.get("scaler_used"),
            "artifacts_saved_at": preprocessing_info.get("artifacts_saved_at")
        },

        "features": {
            "excluded_columns": preprocessing_info.get("excluded_columns", []),
            "reintegrated_columns": preprocessing_info.get("reintegrated_columns", []),
            "final_features": preprocessing_info.get("final_features", []),
            "n_final_features": len(preprocessing_info.get("final_features", []))
        },

        "missing_values": dataframe_to_records(
            reports.get("missing_values")
        ),

        "dropped_high_missing_columns": reports.get(
            "dropped_high_missing_columns",
            []
        ),

        "categorical_variants_cleaning": reports.get(
            "categorical_variants_cleaning",
            {}
        ),

        "skewness": dataframe_to_records(
            reports.get("skewness")
        ),

        "skew_transformations": dataframe_to_records(
            reports.get("skew_transformations")
        ),

        "outliers_detected": dataframe_to_records(
            reports.get("outliers_detected")
        ),

        "outlier_treatment": dataframe_to_records(
            reports.get("outlier_treatment")
        ),

        "removed_low_variance": reports.get(
            "removed_low_variance",
            []
        ),

        "removed_correlated_features": reports.get(
            "removed_correlated_features",
            []
        ),

        "manual_feature_warnings": reports.get(
            "manual_feature_warnings",
            []
        )
    }

    return report


# ============================================================
# 6. RAPPORT DE REINTEGRATION DES FEATURES
# ============================================================

def generate_feature_override_report(
    requested_columns: List[str],
    reintegrated_columns: List[str],
    warnings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Génère un rapport de réintégration manuelle des features.

    Args:
        requested_columns (list): Colonnes demandées par le technicien.
        reintegrated_columns (list): Colonnes réellement réintégrées.
        warnings (list): Avertissements générés.

    Returns:
        dict: Rapport de réintégration.
    """

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_columns": requested_columns,
        "reintegrated_columns": reintegrated_columns,
        "warnings": warnings,
        "n_requested": len(requested_columns),
        "n_reintegrated": len(reintegrated_columns),
        "n_warnings": len(warnings)
    }

    return report


# ============================================================
# 7. RAPPORT DES ARTEFACTS
# ============================================================

def generate_artifacts_report(
    artifacts_path: str,
    artifacts_exist: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Génère un rapport sur les artefacts sauvegardés.

    Args:
        artifacts_path (str): Chemin des artefacts.
        artifacts_exist (bool): Optionnel, indique si les artefacts existent.

    Returns:
        dict: Rapport des artefacts.
    """

    if artifacts_exist is None:
        artifacts_exist = os.path.exists(artifacts_path)

    file_size_kb = None

    if artifacts_exist and os.path.isfile(artifacts_path):
        file_size_kb = round(os.path.getsize(artifacts_path) / 1024, 2)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "artifacts_path": artifacts_path,
        "artifacts_exist": artifacts_exist,
        "file_size_kb": file_size_kb
    }

    return report


# ============================================================
# RAPPORT GLOBAL DU MODULE DE PREPARATION DES DONNEES
# ============================================================

def generate_data_preparation_report(
    structure: Dict[str, Any],
    preprocessing_info: Dict[str, Any],
    reports: Dict[str, Any],
    output_path: str = "reports/data_preparation_report.json"
) -> Dict[str, Any]:
    """
    Génère un rapport global pour le module de préparation des données.

    Args:
        structure (dict): Structure détectée.
        preprocessing_info (dict): Informations du preprocessing.
        reports (dict): Rapports internes du preprocessing.
        output_path (str): Chemin de sauvegarde JSON.

    Returns:
        dict: Rapport complet du module de préparation des données.
    """

    structure_report = generate_structure_report(structure)

    missing_report = generate_missing_report(
        reports.get("missing_values")
    )

    excluded_report = generate_excluded_features_report(
        excluded_columns=preprocessing_info.get("excluded_columns", []),
        exclusion_reasons=structure.get("exclusion_reasons", {})
    )

    preprocessing_report = generate_preprocessing_report(
        preprocessing_info=preprocessing_info,
        reports=reports
    )

    artifacts_report = generate_artifacts_report(
        artifacts_path=preprocessing_info.get("artifacts_saved_at", "")
    )

    data_preparation_report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": "Data Preparation",
        "title": "Ingestion, intégration, détection de structure et preprocessing automatique",

        "structure_report": structure_report,
        "missing_report": missing_report,
        "excluded_features_report": excluded_report,
        "preprocessing_report": preprocessing_report,
        "artifacts_report": artifacts_report,

        "status": "generated"
    }

    safe_json_dump(data_preparation_report, output_path)

    return data_preparation_report
# ============================================================
# 9. EXPORT SIMPLE EN CSV
# ============================================================

def export_report_section_to_csv(
    records: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Exporte une section de rapport en CSV.

    Args:
        records (list): Liste de dictionnaires.
        output_path (str): Chemin de sortie CSV.
    """

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding="utf-8")
