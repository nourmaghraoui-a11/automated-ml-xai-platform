"""
Module : structure_detection.py

Rôle :
    - Détecter automatiquement la structure d'un dataset
    - Identifier les colonnes numériques, catégorielles, dates
    - Identifier les identifiants
    - Identifier les labels potentiels
    - Déterminer les colonnes exploitables pour le Machine Learning
    - Exclure automatiquement les colonnes risquées

Sprint 1 :
    Ingestion, intégration, détection de structure et preprocessing.
"""

import re
from typing import Dict, List, Any

import pandas as pd


# ============================================================
# 1. OUTILS GÉNÉRAUX
# ============================================================

def normalize_column_name(col: str) -> str:
    """
    Normalise le nom d'une colonne pour faciliter la détection.
    """
    return str(col).strip().lower()


def is_probably_date_column(series: pd.Series, threshold: float = 0.8) -> bool:
    """
    Vérifie si une colonne peut être considérée comme une date.
    """

    if series.empty:
        return False

    # Éviter de transformer une colonne numérique en date
    if pd.api.types.is_numeric_dtype(series):
        return False

    sample = series.dropna().astype(str).head(20)

    if sample.empty:
        return False

    # Pré-filtrage : vérifier si les valeurs ressemblent à des dates
    date_like_pattern = re.compile(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"
        r"|^\d{1,2}[-/]\d{1,2}[-/]\d{4}$"
        r"|^\d{4}[-/]\d{1,2}$"
    )

    date_like_ratio = sample.apply(
        lambda value: bool(date_like_pattern.match(value.strip()))
    ).mean()

    # Si les valeurs ne ressemblent pas du tout à des dates,
    # inutile d'appeler pd.to_datetime
    if date_like_ratio < 0.5:
        return False

    try:
        converted = pd.to_datetime(series, errors="coerce")
        valid_ratio = converted.notna().mean()

        return valid_ratio >= threshold

    except Exception:
        return False

# ============================================================
# 2. DÉTECTION DES COLONNES NUMÉRIQUES
# ============================================================

def detect_numeric_columns(df: pd.DataFrame) -> List["str"]:
    """
    Détecte les colonnes numériques.
    """
    numeric_columns = df.select_dtypes(
        include=["int64", "float64", "int32", "float32", "int", "float"]
    ).columns.tolist()

    return numeric_columns


# ============================================================
# 3. DÉTECTION DES COLONNES DATES
# ============================================================

def detect_date_columns(df: pd.DataFrame) -> List["str"]:
    """
    Détecte les colonnes de type date.
    """

    date_columns = []

    date_keywords = [
        "date",
        "time",
        "datetime",
        "created_at",
        "updated_at",
        "signup",
        "login",
        "birth",
        "timestamp"
    ]

    for col in df.columns:
        col_lower = normalize_column_name(col)

        # Détection par nom de colonne
        if any(keyword in col_lower for keyword in date_keywords):
            if is_probably_date_column(df[col], threshold=0.5):
                date_columns.append(col)
                continue

        # Détection par conversion automatique
        if is_probably_date_column(df[col], threshold=0.8):
            date_columns.append(col)

    return list(set(date_columns))


# ============================================================
# 4. DÉTECTION DES COLONNES IDENTIFIANTS
# ============================================================

def detect_id_columns(df: pd.DataFrame) -> List["str"]:
    """
    Détecte les colonnes identifiants.

    Critères :
        - nom de colonne contenant id, uuid, identifier
        - taux d'unicité très élevé
        - valeurs ressemblant à des UUID
    """

    id_columns = []
    n_rows = len(df)

    id_keywords = [
        "id",
        "user_id",
        "customer_id",
        "client_id",
        "uuid",
        "identifier",
        "transaction_id",
        "order_id"
    ]

    for col in df.columns:
        col_lower = normalize_column_name(col)

        # 1. Détection par nom
        if (
            col_lower == "id"
            or col_lower.endswith("_id")
            or col_lower in id_keywords
            or "uuid" in col_lower
            or "identifier" in col_lower
        ):
            id_columns.append(col)
            continue

        # 2. Détection par unicité élevée
        if n_rows > 20:
            unique_ratio = df[col].nunique(dropna=True) / n_rows

            if unique_ratio > 0.95:
                id_columns.append(col)
                continue

        # 3. Détection simple des UUID
        sample_values = df[col].dropna().astype(str).head(20).tolist()

        uuid_pattern = re.compile(
            r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
            re.IGNORECASE
        )

        if sample_values and all(uuid_pattern.match(value) for value in sample_values):
            id_columns.append(col)

    return list(set(id_columns))


# ============================================================
# 5. DÉTECTION DES COLONNES CATÉGORIELLES
# ============================================================


def detect_categorical_columns(
    df: pd.DataFrame,
    numeric_columns: List[str],
    date_columns: List[str]
) -> List[str]:
    """
    Détecte les colonnes catégorielles sans utiliser select_dtypes("object"),
    afin d'éviter les warnings de compatibilité pandas.
    """

    categorical_columns = []

    for col in df.columns:

        if col in numeric_columns:
            continue

        if col in date_columns:
            continue

        dtype = df[col].dtype

        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            categorical_columns.append(col)

    return categorical_columns


# ============================================================
# 6. DÉTECTION DES TYPES DE COLONNES
# ============================================================

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Détecte automatiquement les types de colonnes :
        - numériques
        - catégorielles
        - dates
        - identifiants
    """

    numeric_columns = detect_numeric_columns(df)
    date_columns = detect_date_columns(df)
    id_columns = detect_id_columns(df)

    categorical_columns = detect_categorical_columns(
        df=df,
        numeric_columns=numeric_columns,
        date_columns=date_columns
    )

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "id_columns": id_columns
    }


# ============================================================
# 7. DÉTECTION DES LABELS POTENTIELS
# ============================================================

def detect_possible_label_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Détecte automatiquement les colonnes qui peuvent être des labels.

    Critères :
        - nom de la colonne
        - nombre de valeurs uniques
        - ratio de valeurs uniques
        - position de la colonne
        - valeurs typiques de labels
        - score de confiance
    """

    label_keywords = [
        "label",
        "target",
        "class",
        "classe",
        "status",
        "result",
        "outcome",
        "is_anomaly",
        "anomaly",
        "fraud",
        "is_fraud",
        "churn",
        "is_churn",
        "target_class",
        "prediction"
    ]

    typical_label_values = [
        "0",
        "1",
        "yes",
        "no",
        "true",
        "false",
        "normal",
        "anomaly",
        "fraud",
        "not_fraud",
        "churn",
        "not_churn",
        "active",
        "inactive"
    ]

    possible_labels = []

    n_rows = len(df)
    columns = list(df.columns)

    for col in df.columns:
        score = 0
        reasons = []

        col_lower = normalize_column_name(col)

        # 1. Nom de colonne
        for keyword in label_keywords:
            if keyword in col_lower:
                score += 3
                reasons.append(f"Nom contenant le mot-clé '{keyword}'")
                break

        # 2. Nombre de valeurs uniques
        unique_count = df[col].nunique(dropna=True)

        if unique_count <= 2:
            score += 2
            reasons.append("Colonne binaire ou quasi-binaire")

        elif unique_count <= 10:
            score += 1
            reasons.append("Faible nombre de classes possibles")

        # 3. Ratio de valeurs uniques
        if n_rows > 0:
            unique_ratio = unique_count / n_rows

            if unique_ratio < 0.05:
                score += 1
                reasons.append("Faible ratio de valeurs uniques")

        # 4. Position de la colonne
        if col == columns[-1]:
            score += 1
            reasons.append("Colonne située en dernière position")

        # 5. Valeurs typiques
        values = (
            df[col]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
            .tolist()
        )

        if any(value in typical_label_values for value in values):
            score += 1
            reasons.append("Valeurs typiques d'une colonne cible")

        # 6. Seuil de décision
        if score >= 3:
            possible_labels.append({
                "column": col,
                "score": score,
                "unique_values": int(unique_count),
                "reasons": reasons
            })

    return possible_labels


# ============================================================
# 8. RAISONS D’EXCLUSION
# ============================================================

def build_exclusion_reasons(
    id_columns: List[str],
    possible_label_columns: List[str]
) -> Dict[str, str]:
    """
    Génère les raisons d'exclusion des colonnes.
    """

    reasons = {}

    for col in id_columns:
        reasons[col] = "Identifiant détecté automatiquement"

    for col in possible_label_columns:
        reasons[col] = "Label potentiel détecté automatiquement"

    return reasons


# ============================================================
# 9. DÉTECTION COMPLÈTE DE STRUCTURE
# ============================================================

def detect_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Détecte la structure complète du dataset :
        - types de colonnes
        - labels potentiels
        - colonnes à exclure
        - features exploitables
        - raisons d'exclusion
        - statistiques générales
    """

    if df is None or df.empty:
        raise ValueError("Le DataFrame est vide ou invalide.")

    column_types = detect_column_types(df)
    possible_labels_info = detect_possible_label_columns(df)

    possible_label_columns = [
        item["column"] for item in possible_labels_info
    ]

    id_columns = column_types["id_columns"]

    excluded_columns = list(set(id_columns + possible_label_columns))

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    exclusion_reasons = build_exclusion_reasons(
        id_columns=id_columns,
        possible_label_columns=possible_label_columns
    )

    structure = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),

        "numeric_columns": column_types["numeric_columns"],
        "categorical_columns": column_types["categorical_columns"],
        "date_columns": column_types["date_columns"],
        "id_columns": id_columns,

        "possible_label_columns": possible_label_columns,
        "possible_labels_info": possible_labels_info,

        "excluded_columns": excluded_columns,
        "exclusion_reasons": exclusion_reasons,

        "feature_columns": feature_columns
    }

    return structure


# ============================================================
# 10. RÉSUMÉ DE STRUCTURE POUR DASHBOARD TECHNIQUE
# ============================================================

def generate_structure_summary(structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Génère un résumé simple de la structure détectée.
    """

    summary = {
        "nombre_lignes": structure["n_rows"],
        "nombre_colonnes": structure["n_columns"],
        "nb_colonnes_numeriques": len(structure["numeric_columns"]),
        "nb_colonnes_categorielles": len(structure["categorical_columns"]),
        "nb_colonnes_dates": len(structure["date_columns"]),
        "nb_identifiants": len(structure["id_columns"]),
        "nb_labels_potentiels": len(structure["possible_label_columns"]),
        "nb_colonnes_exclues": len(structure["excluded_columns"]),
        "nb_features_exploitables": len(structure["feature_columns"])
    }

    return summary
