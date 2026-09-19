# ============================================================
# warehouse.py
# Data Warehouse : historisation des exécutions du pipeline
# ============================================================

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd


# ============================================================
# 1. CONNEXION AU DATA WAREHOUSE
# ============================================================

def get_connection(db_path: str = "data/warehouse/pipeline_warehouse.db"):
    """
    Crée une connexion SQLite vers le Data Warehouse.
    """

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    return conn


# ============================================================
# 2. INITIALISATION DES TABLES
# ============================================================

def initialize_warehouse(db_path: str = "data/warehouse/pipeline_warehouse.db"):
    """
    Initialise les tables principales du Data Warehouse.
    """

    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Table des exécutions du module de préparation des données
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT,
            ended_at TEXT,
            duration_seconds REAL,
            status TEXT,
            input_rows INTEGER,
            input_columns INTEGER,
            output_rows INTEGER,
            output_columns INTEGER,
            label_column TEXT,
            scaler_used TEXT,
            artifacts_path TEXT,
            reports_path TEXT
        )
    """)

    # Table des colonnes exclues
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS excluded_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            column_name TEXT,
            reason TEXT,
            created_at TEXT,
            FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
        )
    """)

    # Table des colonnes réintégrées
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reintegrated_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            column_name TEXT,
            created_at TEXT,
            FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
        )
    """)

    # Table des features finales
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS final_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            feature_name TEXT,
            created_at TEXT,
            FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
        )
    """)

    # Table des rapports techniques
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS technical_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            report_type TEXT,
            report_json TEXT,
            created_at TEXT,
            FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
        )
    """)

    # Table des exécutions Machine Learning
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT,
            ended_at TEXT,
            duration_seconds REAL,
            status TEXT,
            input_rows INTEGER,
            input_columns INTEGER,
            enable_clustering INTEGER,
            enable_anomaly_detection INTEGER,
            models_dir TEXT,
            reports_dir TEXT
        )
    """)

    # Table des résultats de clustering
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            n_clusters INTEGER,
            algorithm_used TEXT,
            is_large_dataset INTEGER,
            silhouette_score REAL,
            davies_bouldin_score REAL,
            metrics_sample_size INTEGER,
            created_at TEXT,
            FOREIGN KEY(run_id) REFERENCES ml_runs(run_id)
        )
    """)

    # Table des profils des clusters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_profiles_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            cluster_id INTEGER,
            size INTEGER,
            percentage REAL,
            dominant_features TEXT,
            created_at TEXT,
            FOREIGN KEY(run_id) REFERENCES ml_runs(run_id)
        )
    """)

    # Table des résultats d'anomalies
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            total_rows INTEGER,
            anomaly_count INTEGER,
            anomaly_ratio REAL,
            avg_anomaly_score REAL,
            max_anomaly_score REAL,
            severity_counts TEXT,
            created_at TEXT,
            FOREIGN KEY(run_id) REFERENCES ml_runs(run_id)
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# 3. SAUVEGARDER UNE EXECUTION DU MODULE DE PREPARATION
# ============================================================

def save_pipeline_run(
    run_metadata: Dict[str, Any],
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde les métadonnées d'une exécution du module de préparation des données.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    input_shape = run_metadata.get("input_shape", [0, 0])
    output_shape = run_metadata.get("output_shape", [0, 0])

    cursor.execute("""
        INSERT OR REPLACE INTO pipeline_runs (
            run_id,
            created_at,
            ended_at,
            duration_seconds,
            status,
            input_rows,
            input_columns,
            output_rows,
            output_columns,
            label_column,
            scaler_used,
            artifacts_path,
            reports_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_metadata.get("run_id"),
        run_metadata.get("created_at"),
        run_metadata.get("ended_at"),
        run_metadata.get("duration_seconds"),
        run_metadata.get("status"),
        input_shape[0],
        input_shape[1],
        output_shape[0],
        output_shape[1],
        run_metadata.get("label_column"),
        run_metadata.get("scaler_used"),
        run_metadata.get("artifacts_path"),
        run_metadata.get("reports_run_dir")
    ))

    conn.commit()
    conn.close()


# ============================================================
# 4. SAUVEGARDER LES FEATURES EXCLUES
# ============================================================

def save_excluded_features(
    run_id: str,
    excluded_columns: list,
    exclusion_reasons: Optional[Dict[str, str]] = None,
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde les colonnes exclues et leurs raisons.
    """

    initialize_warehouse(db_path)

    exclusion_reasons = exclusion_reasons or {}

    conn = get_connection(db_path)
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for col in excluded_columns:
        cursor.execute("""
            INSERT INTO excluded_features (
                run_id,
                column_name,
                reason,
                created_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            run_id,
            col,
            exclusion_reasons.get(col, "Raison non spécifiée"),
            created_at
        ))

    conn.commit()
    conn.close()


# ============================================================
# 5. SAUVEGARDER LES FEATURES REINTEGREES
# ============================================================

def save_reintegrated_features(
    run_id: str,
    reintegrated_columns: list,
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde les colonnes réintégrées manuellement.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for col in reintegrated_columns:
        cursor.execute("""
            INSERT INTO reintegrated_features (
                run_id,
                column_name,
                created_at
            )
            VALUES (?, ?, ?)
        """, (
            run_id,
            col,
            created_at
        ))

    conn.commit()
    conn.close()


# ============================================================
# 6. SAUVEGARDER LES FEATURES FINALES
# ============================================================

def save_final_features(
    run_id: str,
    final_features: list,
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde les features finales utilisées par le pipeline.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for feature in final_features:
        cursor.execute("""
            INSERT INTO final_features (
                run_id,
                feature_name,
                created_at
            )
            VALUES (?, ?, ?)
        """, (
            run_id,
            feature,
            created_at
        ))

    conn.commit()
    conn.close()


# ============================================================
# 7. SAUVEGARDER LES RAPPORTS TECHNIQUES
# ============================================================

def save_technical_report(
    run_id: str,
    report_type: str,
    report: Dict[str, Any],
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde un rapport technique sous format JSON.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_json = json.dumps(
        report,
        ensure_ascii=False,
        default=str
    )

    cursor.execute("""
        INSERT INTO technical_reports (
            run_id,
            report_type,
            report_json,
            created_at
        )
        VALUES (?, ?, ?, ?)
    """, (
        run_id,
        report_type,
        report_json,
        created_at
    ))

    conn.commit()
    conn.close()


# ============================================================
# 8. SAUVEGARDE COMPLETE DU MODULE DE PREPARATION DES DONNEES
# ============================================================

def save_data_preparation_to_warehouse(
    result: Dict[str, Any],
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde complète des résultats du module de préparation des données
    dans le Data Warehouse.
    """

    run_id = result["run_id"]
    run_metadata = result["run_metadata"]
    structure = result["structure"]
    preprocessing_info = result["preprocessing_info"]

    save_pipeline_run(
        run_metadata=run_metadata,
        db_path=db_path
    )

    save_excluded_features(
        run_id=run_id,
        excluded_columns=preprocessing_info.get("excluded_columns", []),
        exclusion_reasons=structure.get("exclusion_reasons", {}),
        db_path=db_path
    )

    save_reintegrated_features(
        run_id=run_id,
        reintegrated_columns=preprocessing_info.get("reintegrated_columns", []),
        db_path=db_path
    )

    save_final_features(
        run_id=run_id,
        final_features=preprocessing_info.get("final_features", []),
        db_path=db_path
    )

    save_technical_report(
        run_id=run_id,
        report_type="structure_report",
        report=result.get("structure_report", {}),
        db_path=db_path
    )

    save_technical_report(
        run_id=run_id,
        report_type="preprocessing_report",
        report=result.get("preprocessing_report", {}),
        db_path=db_path
    )

    save_technical_report(
        run_id=run_id,
        report_type="data_preparation_report",
        report=result.get("data_preparation_report", {}),
        db_path=db_path
    )

    return {
        "status": "saved",
        "run_id": run_id,
        "warehouse_path": db_path
    }


# ============================================================
# 9. HISTORISATION MACHINE LEARNING
# ============================================================

def save_ml_run(
    metadata: Dict[str, Any],
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde les métadonnées d'une exécution Machine Learning.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    input_shape = metadata.get("input_shape", [0, 0])

    cursor.execute("""
        INSERT OR REPLACE INTO ml_runs (
            run_id,
            created_at,
            ended_at,
            duration_seconds,
            status,
            input_rows,
            input_columns,
            enable_clustering,
            enable_anomaly_detection,
            models_dir,
            reports_dir
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        metadata.get("run_id"),
        metadata.get("created_at"),
        metadata.get("ended_at"),
        metadata.get("duration_seconds"),
        metadata.get("status"),
        input_shape[0],
        input_shape[1],
        int(metadata.get("enable_clustering", False)),
        int(metadata.get("enable_anomaly_detection", False)),
        metadata.get("models_dir"),
        metadata.get("reports_dir")
    ))

    conn.commit()
    conn.close()


def save_cluster_results(
    run_id: str,
    clustering_result: Dict[str, Any],
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde les métriques de clustering.
    """

    if clustering_result is None:
        return

    initialize_warehouse(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    metrics = clustering_result.get("metrics", {})
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO cluster_history (
            run_id,
            n_clusters,
            algorithm_used,
            is_large_dataset,
            silhouette_score,
            davies_bouldin_score,
            metrics_sample_size,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        clustering_result.get("n_clusters"),
        clustering_result.get("algorithm_used"),
        int(clustering_result.get("is_large_dataset", False)),
        metrics.get("silhouette_score"),
        metrics.get("davies_bouldin_score"),
        metrics.get("sample_size"),
        created_at
    ))

    conn.commit()
    conn.close()


def save_cluster_profiles(
    run_id: str,
    cluster_profiles: pd.DataFrame,
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde les profils des clusters.
    """

    if cluster_profiles is None or cluster_profiles.empty:
        return

    initialize_warehouse(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in cluster_profiles.iterrows():

        dominant_features = row.get("dominant_features")

        if isinstance(dominant_features, list):
            dominant_features = json.dumps(
                dominant_features,
                ensure_ascii=False
            )

        cursor.execute("""
            INSERT INTO cluster_profiles_history (
                run_id,
                cluster_id,
                size,
                percentage,
                dominant_features,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            int(row.get("cluster")),
            int(row.get("size")),
            float(row.get("percentage")),
            dominant_features,
            created_at
        ))

    conn.commit()
    conn.close()


def save_anomaly_results(
    run_id: str,
    anomaly_result: Dict[str, Any],
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde le résumé de détection d'anomalies.
    """

    if anomaly_result is None:
        return

    initialize_warehouse(db_path)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    summary = anomaly_result.get("summary", {})
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    severity_counts = summary.get("severity_counts", {})

    cursor.execute("""
        INSERT INTO anomaly_history (
            run_id,
            total_rows,
            anomaly_count,
            anomaly_ratio,
            avg_anomaly_score,
            max_anomaly_score,
            severity_counts,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        summary.get("total_rows"),
        summary.get("anomaly_count"),
        summary.get("anomaly_ratio"),
        summary.get("avg_anomaly_score"),
        summary.get("max_anomaly_score"),
        json.dumps(severity_counts, ensure_ascii=False),
        created_at
    ))

    conn.commit()
    conn.close()


def save_ml_results_to_warehouse(
    ml_result: Dict[str, Any],
    db_path: str = "data/warehouse/pipeline_warehouse.db"
):
    """
    Sauvegarde complète des résultats Machine Learning dans le Data Warehouse.
    """

    run_id = ml_result.get("run_id")
    metadata = ml_result.get("metadata", {})
    clustering_result = ml_result.get("clustering")
    anomaly_result = ml_result.get("anomaly_detection")
    cluster_profiles = ml_result.get("cluster_profiles")

    save_ml_run(
        metadata=metadata,
        db_path=db_path
    )

    save_cluster_results(
        run_id=run_id,
        clustering_result=clustering_result,
        db_path=db_path
    )

    save_cluster_profiles(
        run_id=run_id,
        cluster_profiles=cluster_profiles,
        db_path=db_path
    )

    save_anomaly_results(
        run_id=run_id,
        anomaly_result=anomaly_result,
        db_path=db_path
    )

    return {
        "status": "saved",
        "run_id": run_id,
        "warehouse_path": db_path
    }


# ============================================================
# 10. LECTURE DE L'HISTORIQUE PREPARATION DES DONNEES
# ============================================================

def get_pipeline_runs(
    db_path: str = "data/warehouse/pipeline_warehouse.db"
) -> pd.DataFrame:
    """
    Retourne l'historique des exécutions du module de préparation.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)

    df = pd.read_sql_query("""
        SELECT *
        FROM pipeline_runs
        ORDER BY created_at DESC
    """, conn)

    conn.close()

    return df


def get_features_history(
    table_name: str,
    db_path: str = "data/warehouse/pipeline_warehouse.db"
) -> pd.DataFrame:
    """
    Retourne l'historique des features.

    table_name peut être :
        - excluded_features
        - reintegrated_features
        - final_features
    """

    allowed_tables = [
        "excluded_features",
        "reintegrated_features",
        "final_features"
    ]

    if table_name not in allowed_tables:
        raise ValueError(f"Table non autorisée : {table_name}")

    initialize_warehouse(db_path)

    conn = get_connection(db_path)

    df = pd.read_sql_query(f"""
        SELECT *
        FROM {table_name}
        ORDER BY created_at DESC
    """, conn)

    conn.close()

    return df


def get_reports_history(
    db_path: str = "data/warehouse/pipeline_warehouse.db"
) -> pd.DataFrame:
    """
    Retourne l'historique des rapports techniques.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)

    df = pd.read_sql_query("""
        SELECT
            id,
            run_id,
            report_type,
            created_at
        FROM technical_reports
        ORDER BY created_at DESC
    """, conn)

    conn.close()

    return df


# ============================================================
# 11. LECTURE DE L'HISTORIQUE MACHINE LEARNING
# ============================================================

def get_ml_runs(
    db_path: str = "data/warehouse/pipeline_warehouse.db"
) -> pd.DataFrame:
    """
    Retourne l'historique des exécutions Machine Learning.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)

    df = pd.read_sql_query("""
        SELECT *
        FROM ml_runs
        ORDER BY created_at DESC
    """, conn)

    conn.close()

    return df


def get_cluster_history(
    db_path: str = "data/warehouse/pipeline_warehouse.db"
) -> pd.DataFrame:
    """
    Retourne l'historique des résultats de clustering.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)

    df = pd.read_sql_query("""
        SELECT *
        FROM cluster_history
        ORDER BY created_at DESC
    """, conn)

    conn.close()

    return df


def get_cluster_profiles_history(
    db_path: str = "data/warehouse/pipeline_warehouse.db"
) -> pd.DataFrame:
    """
    Retourne l'historique des profils de clusters.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)

    df = pd.read_sql_query("""
        SELECT *
        FROM cluster_profiles_history
        ORDER BY created_at DESC
    """, conn)

    conn.close()

    return df


def get_anomaly_history(
    db_path: str = "data/warehouse/pipeline_warehouse.db"
) -> pd.DataFrame:
    """
    Retourne l'historique des résultats d'anomalies.
    """

    initialize_warehouse(db_path)

    conn = get_connection(db_path)

    df = pd.read_sql_query("""
        SELECT *
        FROM anomaly_history
        ORDER BY created_at DESC
    """, conn)

    conn.close()

    return df
