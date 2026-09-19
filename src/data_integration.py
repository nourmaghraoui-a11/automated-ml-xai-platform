"""
Module : data_integration.py

Rôle :
    - Harmoniser les colonnes
    - Appliquer des mappings de noms de colonnes
    - Convertir les types de données
    - Fusionner plusieurs sources
    - Agréger les tables événementielles
    - Construire un dataset unifié prêt pour la détection de structure

Ce module appartient au Sprint 1 :
    Ingestion, intégration, détection de structure et preprocessing.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Union

import pandas as pd


# ============================================================
# 1. STANDARDISATION DES NOMS DE COLONNES
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalise un texte :
    - minuscules
    - suppression accents
    - remplacement espaces par underscores
    - suppression caractères spéciaux

    Args:
        text (str): Texte à normaliser.

    Returns:
        str: Texte normalisé.
    """

    if text is None:
        return ""

    text = str(text).strip().lower()

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")

    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_]", "", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")

    return text


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les noms des colonnes d'un DataFrame.

    Exemple :
        "User ID" -> "user_id"
        "Date Achat" -> "date_achat"

    Args:
        df (pd.DataFrame): DataFrame d'entrée.

    Returns:
        pd.DataFrame: DataFrame avec colonnes standardisées.
    """

    df = df.copy()
    df.columns = [normalize_text(col) for col in df.columns]

    return df


# ============================================================
# 2. MAPPING DES COLONNES
# ============================================================

def apply_column_mapping(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Applique un mapping de colonnes.

    Exemple :
        {
            "customer_id": "user_id",
            "id_user": "user_id",
            "montant": "amount"
        }

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        mapping (dict): Dictionnaire ancien_nom -> nouveau_nom.

    Returns:
        pd.DataFrame: DataFrame renommé.
    """

    df = df.copy()

    if mapping is None:
        return df

    normalized_mapping = {
        normalize_text(old): normalize_text(new)
        for old, new in mapping.items()
    }

    existing_mapping = {
        old: new
        for old, new in normalized_mapping.items()
        if old in df.columns
    }

    df = df.rename(columns=existing_mapping)

    return df


# ============================================================
# 3. CONVERSION DES TYPES
# ============================================================

def cast_column_types(
    df: pd.DataFrame,
    schema: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Convertit les colonnes selon un schéma donné.

    Exemple :
        {
            "user_id": "str",
            "amount": "float",
            "transaction_date": "datetime"
        }

    Types supportés :
        - str
        - int
        - float
        - numeric
        - datetime
        - category

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        schema (dict): Dictionnaire colonne -> type.

    Returns:
        pd.DataFrame: DataFrame avec types convertis.
    """

    df = df.copy()

    if schema is None:
        return df

    for col, target_type in schema.items():
        col = normalize_text(col)

        if col not in df.columns:
            continue

        target_type = target_type.lower().strip()

        try:
            if target_type == "str":
                df[col] = df[col].astype(str)

            elif target_type == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            elif target_type in ["float", "numeric"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            elif target_type == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")

            elif target_type == "category":
                df[col] = df[col].astype("category")

        except Exception:
            # On continue sans bloquer tout le pipeline
            pass

    return df


# ============================================================
# 4. DÉTECTION DE CLÉS DE JOINTURE
# ============================================================

def detect_join_keys(
    dataframes: Dict[str, pd.DataFrame],
    candidate_keys: Optional[List[str]] = None
) -> List[str]:
    """
    Détecte les clés communes possibles entre plusieurs DataFrames.

    Par défaut, on cherche :
        - user_id
        - customer_id
        - client_id
        - id_user

    Args:
        dataframes (dict): Dictionnaire nom_source -> DataFrame.
        candidate_keys (list): Liste de clés candidates.

    Returns:
        list: Clés communes trouvées.
    """

    if candidate_keys is None:
        candidate_keys = [
            "user_id",
            "customer_id",
            "client_id",
            "id_user",
            "id_client"
        ]

    normalized_keys = [normalize_text(key) for key in candidate_keys]

    common_keys = []

    for key in normalized_keys:
        count = 0

        for df in dataframes.values():
            if key in df.columns:
                count += 1

        if count >= 2:
            common_keys.append(key)

    return common_keys


# ============================================================
# 5. AGRÉGATION DES TABLES ÉVÉNEMENTIELLES
# ============================================================

def aggregate_event_table(
    df: pd.DataFrame,
    group_key: str,
    aggregations: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Agrège une table événementielle.

    Exemple :
        transactions :
            user_id, amount, transaction_id

        Résultat :
            user_id, amount_sum, amount_mean, transaction_id_count

    Args:
        df (pd.DataFrame): Table événementielle.
        group_key (str): Clé de groupement.
        aggregations (dict): Dictionnaire colonne -> liste agrégations.

    Returns:
        pd.DataFrame: Table agrégée.
    """

    df = df.copy()
    group_key = normalize_text(group_key)

    if group_key not in df.columns:
        raise ValueError(f"Clé de groupement introuvable : {group_key}")

    agg_dict = {}

    for col, funcs in aggregations.items():
        col = normalize_text(col)

        if col in df.columns:
            agg_dict[col] = funcs

    if not agg_dict:
        raise ValueError("Aucune colonne valide pour l'agrégation.")

    aggregated = df.groupby(group_key).agg(agg_dict)

    aggregated.columns = [
        f"{col}_{func}"
        for col, func in aggregated.columns
    ]

    aggregated = aggregated.reset_index()

    return aggregated


# ============================================================
# 6. FUSION DES SOURCES
# ============================================================

def merge_sources(
    dataframes: Dict[str, pd.DataFrame],
    join_key: str,
    how: str = "left"
) -> pd.DataFrame:
    """
    Fusionne plusieurs sources autour d'une clé commune.

    Args:
        dataframes (dict): Dictionnaire nom_source -> DataFrame.
        join_key (str): Clé de jointure.
        how (str): Type de jointure.

    Returns:
        pd.DataFrame: Dataset fusionné.
    """

    if not dataframes:
        raise ValueError("Aucune source fournie pour la fusion.")

    join_key = normalize_text(join_key)

    source_names = list(dataframes.keys())

    base_name = source_names[0]
    merged_df = dataframes[base_name].copy()

    if join_key not in merged_df.columns:
        raise ValueError(
            f"La clé '{join_key}' est absente de la source principale '{base_name}'."
        )

    for name in source_names[1:]:
        df = dataframes[name].copy()

        if join_key not in df.columns:
            continue

        merged_df = merged_df.merge(
            df,
            on=join_key,
            how=how,
            suffixes=("", f"_{name}")
        )

    return merged_df


# ============================================================
# 7. SUPPRESSION DES COLONNES DUPLIQUÉES
# ============================================================

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes dupliquées après fusion.

    Args:
        df (pd.DataFrame): DataFrame fusionné.

    Returns:
        pd.DataFrame: DataFrame sans colonnes dupliquées.
    """

    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ============================================================
# 8. VALIDATION SOURCE UNIQUE
# ============================================================

def validate_single_source(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None,
    schema: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Valide et standardise une source unique.

    Args:
        df (pd.DataFrame): DataFrame source.
        mapping (dict): Mapping optionnel.
        schema (dict): Schéma optionnel.

    Returns:
        pd.DataFrame: Dataset final standardisé.
    """

    df = df.copy()

    df = standardize_column_names(df)
    df = apply_column_mapping(df, mapping)
    df = cast_column_types(df, schema)
    df = remove_duplicate_columns(df)

    return df


# ============================================================
# 9. INTÉGRATION MULTI-SOURCE
# ============================================================

def integrate_sources(
    dataframes: Dict[str, pd.DataFrame],
    mappings: Optional[Dict[str, Dict[str, str]]] = None,
    schemas: Optional[Dict[str, Dict[str, str]]] = None,
    join_key: Optional[str] = None,
    event_aggregations: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    """
    Intègre plusieurs sources de données.

    Args:
        dataframes (dict): Dictionnaire source -> DataFrame.
        mappings (dict): Mapping par source.
        schemas (dict): Schéma de types par source.
        join_key (str): Clé de jointure.
        event_aggregations (dict): Configuration d'agrégation.

    Exemple event_aggregations :
        {
            "transactions": {
                "group_key": "user_id",
                "aggregations": {
                    "amount": ["sum", "mean"],
                    "transaction_id": ["count"]
                }
            }
        }

    Returns:
        pd.DataFrame: Dataset unifié.
    """

    if not dataframes:
        raise ValueError("Aucune source à intégrer.")

    cleaned_sources = {}

    for name, df in dataframes.items():
        source_mapping = mappings.get(name, {}) if mappings else {}
        source_schema = schemas.get(name, {}) if schemas else {}

        clean_df = validate_single_source(
            df=df,
            mapping=source_mapping,
            schema=source_schema
        )

        cleaned_sources[name] = clean_df

    # Agrégation des tables événementielles
    if event_aggregations:
        for source_name, config in event_aggregations.items():
            if source_name not in cleaned_sources:
                continue

            group_key = config.get("group_key")
            aggregations = config.get("aggregations")

            if group_key and aggregations:
                cleaned_sources[source_name] = aggregate_event_table(
                    cleaned_sources[source_name],
                    group_key=group_key,
                    aggregations=aggregations
                )

    # Détection automatique de clé si non fournie
    if join_key is None:
        detected_keys = detect_join_keys(cleaned_sources)

        if not detected_keys:
            raise ValueError(
                "Aucune clé de jointure détectée automatiquement. "
                "Veuillez fournir join_key."
            )

        join_key = detected_keys[0]

    dataset_unified = merge_sources(
        dataframes=cleaned_sources,
        join_key=join_key,
        how="left"
    )

    dataset_unified = remove_duplicate_columns(dataset_unified)

    return dataset_unified


# ============================================================
# 10. FONCTION PRINCIPALE
# ============================================================

def build_unified_dataset(
    dataframes: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Construit le dataset final du Sprint 1.

    Cas 1 :
        Source unique -> standardisation simple

    Cas 2 :
        Plusieurs sources -> intégration multi-source

    Args:
        dataframes (DataFrame ou Dict[str, DataFrame]): Données source.
        config (dict): Configuration optionnelle.

    Returns:
        pd.DataFrame: Dataset final.
    """

    config = config or {}

    if isinstance(dataframes, pd.DataFrame):
        return validate_single_source(
            df=dataframes,
            mapping=config.get("mapping"),
            schema=config.get("schema")
        )

    elif isinstance(dataframes, dict):
        return integrate_sources(
            dataframes=dataframes,
            mappings=config.get("mappings"),
            schemas=config.get("schemas"),
            join_key=config.get("join_key"),
            event_aggregations=config.get("event_aggregations")
        )

    else:
        raise TypeError(
            "dataframes doit être soit un DataFrame pandas, "
            "soit un dictionnaire de DataFrames."
        )


# ============================================================
# 11. INFORMATIONS SUR LE DATASET FINAL
# ============================================================

def get_integration_report(
    dataset: pd.DataFrame
) -> Dict:
    """
    Génère un petit rapport sur le dataset final.

    Args:
        dataset (pd.DataFrame): Dataset unifié.

    Returns:
        dict: Rapport d'intégration.
    """

    report = {
        "n_rows": dataset.shape[0],
        "n_columns": dataset.shape[1],
        "columns": list(dataset.columns),
        "missing_values": dataset.isnull().sum().to_dict(),
        "duplicated_rows": int(dataset.duplicated().sum())
    }

    return report
