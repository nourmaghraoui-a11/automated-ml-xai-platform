"""
Module : ingestion.py
Rôle :
    - Charger des données depuis un fichier CSV
    - Charger des données depuis une base SQL SQLite
    - Charger une ou plusieurs sources de données
    - Retourner les données sous forme de DataFrame pandas

Ce module appartient au Sprint 1 :
    Ingestion, intégration, détection de structure et preprocessing.
"""

import os
import sqlite3
from typing import Dict, List, Optional, Union

import pandas as pd


# ============================================================
# VALIDATION
# ============================================================

def validate_file_exists(file_path: str) -> None:
    """
    Vérifie si le fichier existe.

    Args:
        file_path (str): Chemin du fichier.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

def validate_dataframe(df: pd.DataFrame, source_name: str = "source") -> None:
    """
    Vérifie si le DataFrame est valide.

    Args:
        df (pd.DataFrame): DataFrame à vérifier.
        source_name (str): Nom de la source.

    Raises:
        ValueError: Si le DataFrame est vide ou invalide.
    """
    if df is None:
        raise ValueError(f"La source {source_name} a retourné None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"La source {source_name} ne retourne pas un DataFrame pandas.")

    if df.empty:
        raise ValueError(f"La source {source_name} est vide.")
# ============================================================
# CHARGEMENT CSV
# ============================================================

def load_csv(
    file_path: str,
    sep: str = ",",
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Charge un fichier CSV et retourne un DataFrame.

    Args:
        file_path (str): Chemin du fichier CSV.
        sep (str): Séparateur utilisé dans le fichier CSV.
        encoding (str): Encodage du fichier.

    Returns:
        pd.DataFrame: Données chargées.
    """
    try:
        validate_file_exists(file_path)

        df = pd.read_csv(
            file_path,
            sep=sep,
            encoding=encoding
        )

        validate_dataframe(df, source_name=file_path)

        return df

    except UnicodeDecodeError:
        try:
            # Tentative avec un encodage fréquent en cas d'erreur UTF-8
            df = pd.read_csv(file_path, sep=sep, encoding="latin1")
            validate_dataframe(df, source_name=file_path)
            return df
        except Exception as e:
            raise Exception(f"Erreur d'encodage lors du chargement CSV : {e}")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du fichier CSV '{file_path}' : {e}")


# ============================================================
# CHARGEMENT SQL
# ============================================================

def load_sql(
    database_path: str,
    query: str
) -> pd.DataFrame:
    """
    Charge des données depuis une base SQLite avec une requête SQL.

    Args:
        database_path (str): Chemin vers la base SQLite.
        query (str): Requête SQL à exécuter.

    Returns:
        pd.DataFrame: Données chargées.
    """
    try:
        validate_file_exists(database_path)

        if query is None or query.strip() == "":
            raise ValueError("La requête SQL ne peut pas être vide.")

        with sqlite3.connect(database_path) as conn:
            df = pd.read_sql_query(query, conn)

        validate_dataframe(df, source_name=database_path)

        return df

    except Exception as e:
        raise Exception(f"Erreur lors du chargement SQL depuis '{database_path}' : {e}")
# ============================================================
# CHARGEMENT GENERIQUE
# ============================================================

def load_data(
    source_type: str,
    source_path: str,
    query: Optional[str] = None,
    sep: str = ",",
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Charge une source de données selon son type.

    Args:
        source_type (str): Type de source : 'csv' ou 'sql'.
        source_path (str): Chemin du fichier CSV ou de la base SQL.
        query (str, optional): Requête SQL si source SQL.
        sep (str): Séparateur CSV.
        encoding (str): Encodage CSV.

    Returns:
        pd.DataFrame: Données chargées.
    """
    source_type = source_type.lower().strip()

    if source_type == "csv":
        return load_csv(
            file_path=source_path,
            sep=sep,
            encoding=encoding
        )

    elif source_type == "sql":
        if query is None:
            raise ValueError("Une requête SQL doit être fournie pour une source SQL.")

        return load_sql(
            database_path=source_path,
            query=query
        )

    else:
        raise ValueError(
            "Type de source non supporté. "
            "Utiliser uniquement : 'csv' ou 'sql'."
        )

# ============================================================
# CHARGEMENT MULTI-SOURCE
# ============================================================

def load_multiple_sources(
    sources_config: List[Dict]
) -> Dict[str, pd.DataFrame]:
    """
    Charge plusieurs sources de données.

    Exemple de configuration :

    sources_config = [
        {
            "name": "users",
            "type": "csv",
            "path": "data/raw/users.csv"
        },
        {
            "name": "transactions",
            "type": "sql",
            "path": "data/raw/database.db",
            "query": "SELECT * FROM transactions"
        }
    ]

    Args:
        sources_config (List[Dict]): Liste de configurations de sources.

    Returns:
        Dict[str, pd.DataFrame]: Dictionnaire contenant les DataFrames chargés.
    """
    if not sources_config:
        raise ValueError("La configuration des sources est vide.")

    loaded_sources = {}

    for source in sources_config:
        try:
            source_name = source.get("name")
            source_type = source.get("type")
            source_path = source.get("path")
            query = source.get("query")
            sep = source.get("sep", ",")
            encoding = source.get("encoding", "utf-8")

            if not source_name:
                raise ValueError("Chaque source doit avoir un champ 'name'.")

            if not source_type:
                raise ValueError(f"La source '{source_name}' doit avoir un champ 'type'.")

            if not source_path:
                raise ValueError(f"La source '{source_name}' doit avoir un champ 'path'.")

            df = load_data(
                source_type=source_type,
                source_path=source_path,
                query=query,
                sep=sep,
                encoding=encoding
            )

            loaded_sources[source_name] = df

        except Exception as e:
            raise Exception(f"Erreur lors du chargement de la source '{source.get('name', 'unknown')}' : {e}")

    return loaded_sources


# ============================================================
# INFORMATIONS SUR LES SOURCES
# ============================================================

def get_dataset_info(df: pd.DataFrame) -> Dict:
    """
    Retourne des informations générales sur un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame analysé.

    Returns:
        Dict: Informations générales.
    """
    validate_dataframe(df)

    info = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3)
    }

    return info

def get_multiple_sources_info(
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """
    Retourne les informations générales de plusieurs DataFrames.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionnaire de DataFrames.

    Returns:
        Dict[str, Dict]: Informations par source.
    """
    if not dataframes:
        raise ValueError("Aucune source chargée.")

    sources_info = {}

    for name, df in dataframes.items():
        sources_info[name] = get_dataset_info(df)

    return sources_info
