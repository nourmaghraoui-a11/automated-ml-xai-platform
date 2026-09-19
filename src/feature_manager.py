# ============================================================
# feature_manager.py
# Sprint 1 : Gestion des features exclues et réintégration contrôlée
# ============================================================

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional


# ============================================================
# 1. RECUPERATION DES COLONNES EXCLUES
# ============================================================

def get_excluded_columns(preprocessing_info: Dict[str, Any]) -> List["str"]:
    """
    Récupère les colonnes exclues depuis preprocessing_info.

    Args:
        preprocessing_info (dict): Informations retournées par preprocessing.py

    Returns:
        list: colonnes exclues
    """

    if preprocessing_info is None:
        return []

    return preprocessing_info.get("excluded_columns", [])


def get_reintegrated_columns(preprocessing_info: Dict[str, Any]) -> List["str"]:
    """
    Récupère les colonnes réintégrées manuellement.

    Args:
        preprocessing_info (dict): Informations de preprocessing.

    Returns:
        list: colonnes réintégrées
    """

    if preprocessing_info is None:
        return []

    return preprocessing_info.get("reintegrated_columns", [])


def get_final_features(preprocessing_info: Dict[str, Any]) -> List["str"]:
    """
    Récupère la liste finale des features utilisées.

    Args:
        preprocessing_info (dict): Informations de preprocessing.

    Returns:
        list: features finales
    """

    if preprocessing_info is None:
        return []

    return preprocessing_info.get("final_features", [])


# ============================================================
# 2. RAISONS D'EXCLUSION
# ============================================================

def get_exclusion_reasons(
    structure: Dict[str, Any],
    excluded_columns: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Retourne les raisons d'exclusion des colonnes.

    Args:
        structure (dict): Structure détectée par structure_detection.py
        excluded_columns (list): Colonnes exclues

    Returns:
        dict: colonne -> raison
    """

    if structure is None:
        return {}

    if excluded_columns is None:
        excluded_columns = structure.get("excluded_columns", [])

    existing_reasons = structure.get("exclusion_reasons", {})

    id_columns = structure.get("id_columns", [])
    label_columns = structure.get("possible_label_columns", [])

    reasons = {}

    for col in excluded_columns:

        if col in existing_reasons:
            reasons[col] = existing_reasons[col]

        elif col in id_columns:
            reasons[col] = "Identifiant détecté automatiquement"

        elif col in label_columns:
            reasons[col] = "Label potentiel détecté automatiquement"

        else:
            reasons[col] = "Colonne exclue automatiquement par le pipeline"

    return reasons


# ============================================================
# 3. EVALUATION DU RISQUE
# ============================================================

def evaluate_feature_risk(
    column: str,
    structure: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Évalue le risque d'une colonne avant réintégration.

    Args:
        column (str): Colonne sélectionnée.
        structure (dict): Structure détectée.

    Returns:
        dict: informations de risque
    """

    id_columns = structure.get("id_columns", [])
    label_columns = structure.get("possible_label_columns", [])
    date_columns = structure.get("date_columns", [])
    numeric_columns = structure.get("numeric_columns", [])
    categorical_columns = structure.get("categorical_columns", [])

    risk_level = "faible"
    risk_reason = "Aucun risque majeur détecté"
    allowed = True

    if column in id_columns:
        risk_level = "élevé"
        risk_reason = "La colonne est détectée comme identifiant. Elle peut provoquer du bruit ou une fuite d'information."
        allowed = False

    elif column in label_columns:
        risk_level = "élevé"
        risk_reason = "La colonne est détectée comme label potentiel. Elle peut provoquer une fuite d'information."
        allowed = False

    elif column in date_columns:
        risk_level = "moyen"
        risk_reason = "La colonne est une date. Elle doit être transformée avant d'être utilisée."
        allowed = True

    elif column in numeric_columns:
        risk_level = "faible"
        risk_reason = "La colonne est numérique et peut être utilisée si elle est pertinente."
        allowed = True

    elif column in categorical_columns:
        risk_level = "faible"
        risk_reason = "La colonne est catégorielle et peut être encodée avant entraînement."
        allowed = True

    else:
        risk_level = "moyen"
        risk_reason = "La colonne n'est pas clairement typée dans la structure détectée."
        allowed = True

    return {
        "column": column,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "allowed": allowed
    }


# ============================================================
# 4. VALIDATION DES FEATURES MANUELLES
# ============================================================

def validate_manual_features(
    selected_columns: List[str],
    structure: Dict[str, Any],
    allow_high_risk: bool = False
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Valide les colonnes sélectionnées par le technicien.

    Args:
        selected_columns (list): colonnes à réintégrer
        structure (dict): structure détectée
        allow_high_risk (bool): autoriser ou non les colonnes à risque élevé

    Returns:
        allowed_columns (list): colonnes acceptées
        warnings (list): avertissements
    """

    if selected_columns is None:
        selected_columns = []

    allowed_columns = []
    warnings = []

    dataset_columns = structure.get("columns", [])

    for col in selected_columns:

        if col not in dataset_columns:
            warnings.append({
                "column": col,
                "risk_level": "invalide",
                "warning": "La colonne n'existe pas dans le dataset.",
                "allowed": False
            })
            continue

        risk_info = evaluate_feature_risk(col, structure)

        if risk_info["risk_level"] == "élevé" and not allow_high_risk:
            warnings.append({
                "column": col,
                "risk_level": risk_info["risk_level"],
                "warning": risk_info["risk_reason"],
                "allowed": False
            })
            continue

        allowed_columns.append(col)

        if risk_info["risk_level"] in ["moyen", "élevé"]:
            warnings.append({
                "column": col,
                "risk_level": risk_info["risk_level"],
                "warning": risk_info["risk_reason"],
                "allowed": True
            })

    return allowed_columns, warnings


# ============================================================
# 5. REINTEGRATION DES FEATURES
# ============================================================

def reintegrate_features(
    selected_columns: List[str],
    structure: Dict[str, Any],
    allow_high_risk: bool = False
) -> Dict[str, Any]:
    """
    Réintègre les colonnes sélectionnées après validation.

    Args:
        selected_columns (list): colonnes sélectionnées
        structure (dict): structure détectée
        allow_high_risk (bool): autoriser colonnes à risque élevé

    Returns:
        dict: résultat de réintégration
    """

    allowed_columns, warnings = validate_manual_features(
        selected_columns=selected_columns,
        structure=structure,
        allow_high_risk=allow_high_risk
    )

    excluded_columns = structure.get("excluded_columns", [])

    remaining_excluded_columns = [
        col for col in excluded_columns
        if col not in allowed_columns
    ]

    updated_feature_columns = list(
        set(structure.get("feature_columns", []) + allowed_columns)
    )

    result = {
        "requested_columns": selected_columns,
        "reintegrated_columns": allowed_columns,
        "remaining_excluded_columns": remaining_excluded_columns,
        "updated_feature_columns": updated_feature_columns,
        "warnings": warnings,
        "allow_high_risk": allow_high_risk,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return result


# ============================================================
# 6. RAPPORT DE REINTEGRATION
# ============================================================

def generate_feature_override_report(
    reintegration_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Génère un rapport structuré de réintégration.

    Args:
        reintegration_result (dict): résultat de reintegrate_features()

    Returns:
        dict: rapport
    """

    if reintegration_result is None:
        return {}

    report = {
        "date_decision": reintegration_result.get("timestamp"),
        "requested_columns": reintegration_result.get("requested_columns", []),
        "reintegrated_columns": reintegration_result.get("reintegrated_columns", []),
        "remaining_excluded_columns": reintegration_result.get("remaining_excluded_columns", []),
        "warnings": reintegration_result.get("warnings", []),
        "allow_high_risk": reintegration_result.get("allow_high_risk", False),
        "n_requested": len(reintegration_result.get("requested_columns", [])),
        "n_reintegrated": len(reintegration_result.get("reintegrated_columns", [])),
        "n_warnings": len(reintegration_result.get("warnings", []))
    }

    return report


# ============================================================
# 7. SAUVEGARDE / CHARGEMENT DES DECISIONS
# ============================================================

def save_feature_decisions(
    report: Dict[str, Any],
    path: str = "reports/feature_decisions.json"
) -> None:
    """
    Sauvegarde le rapport de décision des features.

    Args:
        report (dict): rapport de réintégration
        path (str): chemin de sauvegarde
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


def load_feature_decisions(
    path: str = "reports/feature_decisions.json"
) -> Dict[str, Any]:
    """
    Charge les décisions de réintégration sauvegardées.

    Args:
        path (str): chemin du fichier JSON

    Returns:
        dict: rapport chargé
    """

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 8. REGISTRE DES FEATURES
# ============================================================

def build_feature_registry(
    structure: Dict[str, Any],
    preprocessing_info: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Construit un registre complet des features pour le dashboard technique.

    Args:
        structure (dict): structure détectée
        preprocessing_info (dict): infos preprocessing

    Returns:
        list: registre des colonnes
    """

    if structure is None:
        return []

    columns = structure.get("columns", [])
    excluded_columns = structure.get("excluded_columns", [])
    feature_columns = structure.get("feature_columns", [])
    id_columns = structure.get("id_columns", [])
    label_columns = structure.get("possible_label_columns", [])
    date_columns = structure.get("date_columns", [])
    numeric_columns = structure.get("numeric_columns", [])
    categorical_columns = structure.get("categorical_columns", [])

    reintegrated_columns = []

    if preprocessing_info is not None:
        reintegrated_columns = preprocessing_info.get("reintegrated_columns", [])

    exclusion_reasons = get_exclusion_reasons(structure, excluded_columns)

    registry = []

    for col in columns:

        if col in id_columns:
            detected_type = "identifiant"
        elif col in label_columns:
            detected_type = "label potentiel"
        elif col in date_columns:
            detected_type = "date"
        elif col in numeric_columns:
            detected_type = "numérique"
        elif col in categorical_columns:
            detected_type = "catégorielle"
        else:
            detected_type = "inconnu"

        is_excluded = col in excluded_columns
        is_feature = col in feature_columns
        is_reintegrated = col in reintegrated_columns

        registry.append({
            "column": col,
            "detected_type": detected_type,
            "is_excluded": is_excluded,
            "is_feature": is_feature,
            "is_reintegrated": is_reintegrated,
            "exclusion_reason": exclusion_reasons.get(col, None),
            "risk": evaluate_feature_risk(col, structure)
        })

    return registry


# ============================================================
# 9. RESUME POUR DASHBOARD TECHNIQUE
# ============================================================

def summarize_feature_status(
    feature_registry: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Résume l'état des features.

    Args:
        feature_registry (list): registre des features

    Returns:
        dict: résumé
    """

    if not feature_registry:
        return {
            "total_columns": 0,
            "total_features": 0,
            "total_excluded": 0,
            "total_reintegrated": 0,
            "high_risk_columns": 0
        }

    total_columns = len(feature_registry)

    total_features = sum(
        1 for item in feature_registry
        if item.get("is_feature")
    )

    total_excluded = sum(
        1 for item in feature_registry
        if item.get("is_excluded")
    )

    total_reintegrated = sum(
        1 for item in feature_registry
        if item.get("is_reintegrated")
    )

    high_risk_columns = sum(
        1 for item in feature_registry
        if item.get("risk", {}).get("risk_level") == "élevé"
    )

    return {
        "total_columns": total_columns,
        "total_features": total_features,
        "total_excluded": total_excluded,
        "total_reintegrated": total_reintegrated,
        "high_risk_columns": high_risk_columns
    }

