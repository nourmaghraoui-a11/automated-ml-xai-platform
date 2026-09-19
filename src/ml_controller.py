# ============================================================
# ml_controller.py
# Orchestration du module Machine Learning
# ============================================================

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union

import pandas as pd

from .clustering import (
    run_clustering_pipeline,
    save_clustering_artifacts,
    load_kmeans_model,
    predict_clusters
)

from .anomaly_detection import (
    run_anomaly_detection_pipeline,
    save_anomaly_artifacts,
    load_anomaly_model,
    predict_anomalies
)

from .warehouse import save_ml_results_to_warehouse


# ============================================================
# 1. CREATION DES DOSSIERS D'EXECUTION ML
# ============================================================

def create_ml_run_directories(
    run_id: str,
    models_base_dir: str = "models",
    reports_base_dir: str = "reports"
) -> Dict[str, str]:
    """
    Crée les dossiers nécessaires pour sauvegarder les résultats
    du module Machine Learning.
    """

    ml_models_dir = os.path.join(models_base_dir, run_id)
    ml_reports_dir = os.path.join(reports_base_dir, run_id)

    clustering_models_dir = os.path.join(ml_models_dir, "clustering")
    anomaly_models_dir = os.path.join(ml_models_dir, "anomaly_detection")

    clustering_reports_dir = os.path.join(ml_reports_dir, "clustering")
    anomaly_reports_dir = os.path.join(ml_reports_dir, "anomaly_detection")

    os.makedirs(clustering_models_dir, exist_ok=True)
    os.makedirs(anomaly_models_dir, exist_ok=True)
    os.makedirs(clustering_reports_dir, exist_ok=True)
    os.makedirs(anomaly_reports_dir, exist_ok=True)

    return {
        "ml_models_dir": ml_models_dir,
        "ml_reports_dir": ml_reports_dir,

        "clustering_models_dir": clustering_models_dir,
        "anomaly_models_dir": anomaly_models_dir,

        "clustering_reports_dir": clustering_reports_dir,
        "anomaly_reports_dir": anomaly_reports_dir,

        "clustering_report_path": os.path.join(
            clustering_reports_dir,
            "clustering_report.json"
        ),
        "cluster_profiles_path": os.path.join(
            clustering_reports_dir,
            "cluster_profiles.csv"
        ),
        "pca_projection_path": os.path.join(
            clustering_reports_dir,
            "pca_projection.csv"
        ),

        "anomaly_report_path": os.path.join(
            anomaly_reports_dir,
            "anomaly_report.json"
        ),
        "top_anomalies_path": os.path.join(
            anomaly_reports_dir,
            "top_anomalies.csv"
        ),
        "anomaly_pca_path": os.path.join(
            anomaly_reports_dir,
            "anomaly_pca.csv"
        )
    }


# ============================================================
# 2. GENERATION D'UN IDENTIFIANT D'EXECUTION ML
# ============================================================

def generate_ml_run_id(prefix: str = "ml") -> str:
    """
    Génère un identifiant unique pour une exécution Machine Learning.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


# ============================================================
# 3. OUTILS DE SAUVEGARDE
# ============================================================

def save_json_report(
    report: Dict[str, Any],
    output_path: str
) -> str:
    """
    Sauvegarde un rapport JSON.
    """

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    return output_path


def save_dataframe_csv(
    df: pd.DataFrame,
    output_path: str
) -> str:
    """
    Sauvegarde un DataFrame en CSV.
    """

    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8")

    return output_path


# ============================================================
# 4. PIPELINE ML PRINCIPAL
# ============================================================

def run_ml_pipeline(
    X_ready: pd.DataFrame,
    run_id: Optional[str] = None,

    # Clustering
    enable_clustering: bool = True,
    use_hdbscan: bool = True,
    use_silhouette_fallback: bool = True,
    default_k: int = 3,
    min_cluster_size: int = 100,
    max_k: int = 10,
    large_dataset_threshold: int = 100000,
    hdbscan_sample_size: int = 30000,
    pca_sample_size: int = 10000,
    metrics_sample_size: int = 10000,
    batch_size: int = 4096,

    # Anomaly detection
    enable_anomaly_detection: bool = True,
    contamination: Union[str, float] = "auto",
    n_estimators: int = 200,
    max_samples: Union[str, int, float] = "auto",
    anomaly_pca_sample_size: int = 10000,

    # Général
    random_state: int = 42,
    models_base_dir: str = "models",
    reports_base_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Lance le pipeline Machine Learning complet.

    Modules intégrés :
        - Segmentation utilisateurs.
        - Détection d'anomalies.
        - Sauvegarde des artefacts.
        - Sauvegarde des rapports.
        - Historisation dans le Data Warehouse.

    Args:
        X_ready (pd.DataFrame): Dataset préparé par le module de préparation.

    Returns:
        dict: Résultats complets du module ML.
    """

    if X_ready is None:
        raise ValueError("X_ready ne peut pas être None.")

    if not isinstance(X_ready, pd.DataFrame):
        raise TypeError("X_ready doit être un DataFrame pandas.")

    if X_ready.empty:
        raise ValueError("X_ready est vide.")

    if run_id is None:
        run_id = generate_ml_run_id(prefix="ml")

    start_time = datetime.now()

    paths = create_ml_run_directories(
        run_id=run_id,
        models_base_dir=models_base_dir,
        reports_base_dir=reports_base_dir
    )

    clustering_result = None
    anomaly_result = None
    clustering_artifacts = {}
    anomaly_artifacts = {}

    try:
        # --------------------------------------------------------
        # 1. Segmentation utilisateurs
        # --------------------------------------------------------

        if enable_clustering:

            clustering_result = run_clustering_pipeline(
                X_ready=X_ready,
                use_hdbscan=use_hdbscan,
                use_silhouette_fallback=use_silhouette_fallback,
                default_k=default_k,
                min_cluster_size=min_cluster_size,
                max_k=max_k,
                random_state=random_state,
                large_dataset_threshold=large_dataset_threshold,
                hdbscan_sample_size=hdbscan_sample_size,
                pca_sample_size=pca_sample_size,
                metrics_sample_size=metrics_sample_size,
                batch_size=batch_size
            )

            clustering_artifacts = save_clustering_artifacts(
                clustering_result=clustering_result,
                output_dir=paths["clustering_models_dir"]
            )

            clustering_report = build_clustering_report(
                clustering_result=clustering_result,
                artifacts=clustering_artifacts
            )

            save_json_report(
                report=clustering_report,
                output_path=paths["clustering_report_path"]
            )

            if isinstance(clustering_result.get("cluster_profiles"), pd.DataFrame):
                save_dataframe_csv(
                    df=clustering_result["cluster_profiles"],
                    output_path=paths["cluster_profiles_path"]
                )

            if isinstance(clustering_result.get("pca_2d"), pd.DataFrame):
                save_dataframe_csv(
                    df=clustering_result["pca_2d"],
                    output_path=paths["pca_projection_path"]
                )

        # --------------------------------------------------------
        # 2. Détection d'anomalies
        # --------------------------------------------------------

        if enable_anomaly_detection:

            anomaly_result = run_anomaly_detection_pipeline(
                X_ready=X_ready,
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                pca_sample_size=anomaly_pca_sample_size,
                random_state=random_state
            )

            anomaly_artifacts = save_anomaly_artifacts(
                anomaly_result=anomaly_result,
                output_dir=paths["anomaly_models_dir"]
            )

            anomaly_report = build_anomaly_report(
                anomaly_result=anomaly_result,
                artifacts=anomaly_artifacts
            )

            save_json_report(
                report=anomaly_report,
                output_path=paths["anomaly_report_path"]
            )

            if isinstance(anomaly_result.get("top_anomalies"), pd.DataFrame):
                save_dataframe_csv(
                    df=anomaly_result["top_anomalies"],
                    output_path=paths["top_anomalies_path"]
                )

            if isinstance(anomaly_result.get("pca_2d"), pd.DataFrame):
                save_dataframe_csv(
                    df=anomaly_result["pca_2d"],
                    output_path=paths["anomaly_pca_path"]
                )

        # --------------------------------------------------------
        # 3. Métadonnées globales ML
        # --------------------------------------------------------

        end_time = datetime.now()
        duration_seconds = round((end_time - start_time).total_seconds(), 3)

        metadata = {
            "run_id": run_id,
            "created_at": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
            "status": "success",

            "module": "Machine Learning",
            "input_shape": list(X_ready.shape),

            "enable_clustering": enable_clustering,
            "enable_anomaly_detection": enable_anomaly_detection,

            "models_dir": paths["ml_models_dir"],
            "reports_dir": paths["ml_reports_dir"]
        }

        # --------------------------------------------------------
        # 4. Résultat final
        # --------------------------------------------------------

        result = {
            "run_id": run_id,
            "X_ready": X_ready,

            "clustering": clustering_result,
            "cluster_labels": (
                clustering_result.get("labels")
                if clustering_result is not None
                else None
            ),
            "cluster_profiles": (
                clustering_result.get("cluster_profiles")
                if clustering_result is not None
                else None
            ),
            "cluster_pca_2d": (
                clustering_result.get("pca_2d")
                if clustering_result is not None
                else None
            ),

            "anomaly_detection": anomaly_result,
            "anomaly_labels": (
                anomaly_result.get("anomaly_labels")
                if anomaly_result is not None
                else None
            ),
            "anomaly_scores": (
                anomaly_result.get("anomaly_scores")
                if anomaly_result is not None
                else None
            ),
            "anomaly_result_df": (
                anomaly_result.get("result_df")
                if anomaly_result is not None
                else None
            ),
            "top_anomalies": (
                anomaly_result.get("top_anomalies")
                if anomaly_result is not None
                else None
            ),
            "anomaly_pca_2d": (
                anomaly_result.get("pca_2d")
                if anomaly_result is not None
                else None
            ),

            "artifacts": {
                "clustering": clustering_artifacts,
                "anomaly_detection": anomaly_artifacts
            },

            "metadata": metadata,
            "paths": paths
        }

        # --------------------------------------------------------
        # 5. Historisation dans le Data Warehouse
        # --------------------------------------------------------

        try:
            warehouse_status = save_ml_results_to_warehouse(result)
            result["warehouse_status"] = warehouse_status

        except Exception as warehouse_error:
            result["warehouse_status"] = {
                "status": "failed",
                "error": str(warehouse_error)
            }

        # --------------------------------------------------------
        # 6. Logs console
        # --------------------------------------------------------

        print("Pipeline Machine Learning terminé avec succès.")
        print(f"Run ID : {run_id}")
        print(f"Input shape : {X_ready.shape}")

        if clustering_result is not None:
            print(f"Algorithme clustering : {clustering_result.get('algorithm_used')}")
            print(f"Nombre de clusters : {clustering_result.get('n_clusters')}")

        if anomaly_result is not None:
            print(f"Nombre d'anomalies : {anomaly_result['summary'].get('anomaly_count')}")
            print(f"Ratio anomalies : {anomaly_result['summary'].get('anomaly_ratio')}%")

        print(f"Rapports ML : {paths['ml_reports_dir']}")
        print(f"Data Warehouse : {result.get('warehouse_status')}")

        return result

    except Exception as e:
        end_time = datetime.now()
        duration_seconds = round((end_time - start_time).total_seconds(), 3)

        error_report = {
            "run_id": run_id,
            "created_at": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
            "status": "failed",
            "error": str(e),
            "models_dir": paths["ml_models_dir"],
            "reports_dir": paths["ml_reports_dir"]
        }

        save_json_report(
            report=error_report,
            output_path=os.path.join(
                paths["ml_reports_dir"],
                "ml_error_report.json"
            )
        )

        raise Exception(
            f"Erreur durant l'exécution du module Machine Learning : {e}"
        )


# ============================================================
# 5. RAPPORT CLUSTERING
# ============================================================

def build_clustering_report(
    clustering_result: Dict[str, Any],
    artifacts: Dict[str, str]
) -> Dict[str, Any]:
    """
    Construit un rapport synthétique de clustering.
    """

    hdbscan_result = clustering_result.get("hdbscan", {})
    metrics = clustering_result.get("metrics", {})

    report = {
        "algorithm_used": clustering_result.get("algorithm_used"),
        "is_large_dataset": clustering_result.get("is_large_dataset"),
        "n_clusters": clustering_result.get("n_clusters"),
        "metrics": metrics,
        "hdbscan": {
            "available": hdbscan_result.get("available"),
            "n_clusters": hdbscan_result.get("n_clusters"),
            "n_noise": hdbscan_result.get("n_noise"),
            "noise_ratio": hdbscan_result.get("noise_ratio"),
            "sample_size": hdbscan_result.get("sample_size"),
            "message": hdbscan_result.get("message")
        },
        "best_k_search": clustering_result.get("best_k_search"),
        "config": clustering_result.get("config"),
        "artifacts": artifacts
    }

    return report


# ============================================================
# 6. RAPPORT ANOMALIES
# ============================================================

def build_anomaly_report(
    anomaly_result: Dict[str, Any],
    artifacts: Dict[str, str]
) -> Dict[str, Any]:
    """
    Construit un rapport synthétique de détection d'anomalies.
    """

    report = {
        "summary": anomaly_result.get("summary"),
        "config": anomaly_result.get("config"),
        "artifacts": artifacts
    }

    return report


# ============================================================
# 7. AJOUTER LES LABELS AU DATASET
# ============================================================

def attach_cluster_labels(
    X_ready: pd.DataFrame,
    labels
) -> pd.DataFrame:
    """
    Ajoute les labels de clusters au dataset préparé.
    """

    if X_ready is None:
        raise ValueError("X_ready ne peut pas être None.")

    if labels is None:
        raise ValueError("Les labels de clusters ne peuvent pas être None.")

    if len(X_ready) != len(labels):
        raise ValueError(
            "La taille de X_ready doit correspondre au nombre de labels."
        )

    df_clustered = X_ready.copy()
    df_clustered["cluster"] = labels

    return df_clustered


def attach_anomaly_results(
    X_ready: pd.DataFrame,
    anomaly_labels,
    anomaly_scores
) -> pd.DataFrame:
    """
    Ajoute les résultats d'anomalies au dataset préparé.
    """

    if X_ready is None:
        raise ValueError("X_ready ne peut pas être None.")

    if anomaly_labels is None:
        raise ValueError("Les labels d'anomalies ne peuvent pas être None.")

    if anomaly_scores is None:
        raise ValueError("Les scores d'anomalies ne peuvent pas être None.")

    if len(X_ready) != len(anomaly_labels):
        raise ValueError(
            "La taille de X_ready doit correspondre au nombre de labels d'anomalies."
        )

    df_anomalies = X_ready.copy()
    df_anomalies["is_anomaly"] = anomaly_labels
    df_anomalies["anomaly_score"] = anomaly_scores

    return df_anomalies


# ============================================================
# 8. PREDICTIONS SUR NOUVELLES DONNEES
# ============================================================

def predict_new_clusters(
    X_new: pd.DataFrame,
    kmeans_model_path: str
):
    """
    Affecte de nouvelles données à des clusters existants.
    """

    model = load_kmeans_model(kmeans_model_path)

    labels = predict_clusters(
        X_new=X_new,
        kmeans_model=model
    )

    return labels


def predict_new_anomalies(
    X_new: pd.DataFrame,
    anomaly_model_path: str
) -> pd.DataFrame:
    """
    Prédit les anomalies sur de nouvelles données.
    """

    model = load_anomaly_model(anomaly_model_path)

    result_df = predict_anomalies(
        X_new=X_new,
        model=model
    )

    return result_df


# ============================================================
# 9. RESUME POUR DASHBOARD ANALYTIQUE
# ============================================================

def generate_ml_summary(
    ml_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Génère un résumé global du module ML pour le dashboard analytique.
    """

    if ml_result is None:
        return {}

    clustering = ml_result.get("clustering")
    anomaly = ml_result.get("anomaly_detection")
    metadata = ml_result.get("metadata", {})

    summary = {
        "run_id": ml_result.get("run_id"),
        "status": metadata.get("status"),
        "input_shape": metadata.get("input_shape"),
        "duration_seconds": metadata.get("duration_seconds"),
        "warehouse_status": ml_result.get("warehouse_status")
    }

    if clustering is not None:
        metrics = clustering.get("metrics", {})

        summary.update({
            "n_clusters": clustering.get("n_clusters"),
            "algorithm_used": clustering.get("algorithm_used"),
            "is_large_dataset": clustering.get("is_large_dataset"),
            "silhouette_score": metrics.get("silhouette_score"),
            "davies_bouldin_score": metrics.get("davies_bouldin_score"),
            "metrics_sample_size": metrics.get("sample_size")
        })

    if anomaly is not None:
        anomaly_summary = anomaly.get("summary", {})

        summary.update({
            "anomaly_count": anomaly_summary.get("anomaly_count"),
            "anomaly_ratio": anomaly_summary.get("anomaly_ratio"),
            "avg_anomaly_score": anomaly_summary.get("avg_anomaly_score"),
            "max_anomaly_score": anomaly_summary.get("max_anomaly_score"),
            "severity_counts": anomaly_summary.get("severity_counts")
        })

    return summary


# ============================================================
# 10. TEST LOCAL RAPIDE
# ============================================================

if __name__ == "__main__":

    import numpy as np

    np.random.seed(42)

    X_test = pd.DataFrame({
        "age": np.random.normal(30, 5, 30000),
        "amount_sum": np.random.normal(500, 100, 30000),
        "login_count": np.random.normal(10, 2, 30000)
    })

    result = run_ml_pipeline(
        X_ready=X_test,
        enable_clustering=True,
        enable_anomaly_detection=True,
        large_dataset_threshold=10000,
        hdbscan_sample_size=5000,
        pca_sample_size=3000,
        metrics_sample_size=3000,
        anomaly_pca_sample_size=3000,
        min_cluster_size=50,
        batch_size=2048,
        contamination=0.03
    )

    summary = generate_ml_summary(result)

    print("\n===== RESUME ML =====")
    print(summary)