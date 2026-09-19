# ============================================================
# clustering.py
# Module Machine Learning : segmentation utilisateurs optimisée
# ============================================================

import os

# Eviter certains warnings joblib/loky sous Windows
os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get("LOKY_MAX_CPU_COUNT", "4")

import joblib
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


# ============================================================
# IMPORT OPTIONNEL HDBSCAN
# ============================================================

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# ============================================================
# 1. VALIDATION DES DONNEES
# ============================================================

def validate_clustering_input(X: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie et prépare les données pour le clustering.

    Args:
        X (pd.DataFrame): Dataset préparé.

    Returns:
        pd.DataFrame: Dataset numérique prêt pour clustering.
    """

    if X is None:
        raise ValueError("X ne peut pas être None.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X doit être un DataFrame pandas.")

    if X.empty:
        raise ValueError("X est vide.")

    X_clean = X.copy()

    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("Aucune colonne numérique disponible pour le clustering.")

    X_clean = X_clean[numeric_cols]

    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(0)

    return X_clean


# ============================================================
# 2. ECHANTILLONNAGE
# ============================================================

def sample_dataframe(
    X: pd.DataFrame,
    sample_size: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Retourne un échantillon du dataset si nécessaire.
    """

    if len(X) <= sample_size:
        return X.copy()

    return X.sample(
        n=sample_size,
        random_state=random_state
    )


def sample_positions(
    n_rows: int,
    sample_size: int = 10000,
    random_state: int = 42
) -> np.ndarray:
    """
    Retourne des positions aléatoires pour échantillonnage.
    """

    if n_rows <= sample_size:
        return np.arange(n_rows)

    rng = np.random.default_rng(random_state)

    return rng.choice(
        np.arange(n_rows),
        size=sample_size,
        replace=False
    )


# ============================================================
# 3. HDBSCAN : DECOUVERTE AUTOMATIQUE DES CLUSTERS
# ============================================================

def run_hdbscan(
    X: pd.DataFrame,
    min_cluster_size: int = 50,
    min_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Exécute HDBSCAN afin de découvrir automatiquement la structure des données.

    Args:
        X (pd.DataFrame): Dataset numérique.
        min_cluster_size (int): Taille minimale d'un cluster.
        min_samples (int): Nombre minimal de voisins.

    Returns:
        dict: Résultats HDBSCAN.
    """

    X_clean = validate_clustering_input(X)

    if not HDBSCAN_AVAILABLE:
        return {
            "available": False,
            "model": None,
            "labels": None,
            "n_clusters": None,
            "n_noise": None,
            "noise_ratio": None,
            "message": "Le package hdbscan n'est pas installé."
        }

    if len(X_clean) < min_cluster_size:
        return {
            "available": True,
            "model": None,
            "labels": np.array([-1] * len(X_clean)),
            "n_clusters": 0,
            "n_noise": len(X_clean),
            "noise_ratio": 100.0,
            "message": "Nombre de lignes insuffisant pour HDBSCAN."
        }

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True
    )

    labels = model.fit_predict(X_clean)

    unique_labels = set(labels)
    clusters_without_noise = [
        label for label in unique_labels
        if label != -1
    ]

    n_clusters = len(clusters_without_noise)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = round((n_noise / len(labels)) * 100, 2)

    return {
        "available": True,
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "message": "HDBSCAN exécuté avec succès."
    }


def run_hdbscan_on_sample(
    X: pd.DataFrame,
    sample_size: int = 30000,
    min_cluster_size: int = 100,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Exécute HDBSCAN sur un échantillon pour optimiser les grands datasets.
    """

    X_clean = validate_clustering_input(X)

    X_sample = sample_dataframe(
        X=X_clean,
        sample_size=sample_size,
        random_state=random_state
    )

    result = run_hdbscan(
        X=X_sample,
        min_cluster_size=min_cluster_size
    )

    result["sample_size"] = len(X_sample)

    return result


# ============================================================
# 4. DETERMINATION DE K
# ============================================================

def determine_k_from_hdbscan(
    hdbscan_result: Dict[str, Any],
    default_k: int = 3,
    min_k: int = 2,
    max_k: int = 10
) -> int:
    """
    Détermine un K de départ (hypothèse, pas une valeur finale) à partir
    du nombre de clusters trouvés par HDBSCAN.

    Ce K est ensuite systématiquement affiné par recherche silhouette
    dans run_clustering_pipeline, car le nombre de clusters "naturel"
    trouvé par un algorithme density-based (HDBSCAN) ne correspond pas
    forcément au K optimal pour un algorithme à centroïdes (KMeans).
    """

    n_clusters = hdbscan_result.get("n_clusters")

    if n_clusters is None or n_clusters < min_k:
        return default_k

    if n_clusters > max_k:
        return max_k

    return int(n_clusters)


# ============================================================
# 5. METRIQUES SUR ECHANTILLON
# ============================================================

def compute_clustering_metrics_sampled(
    X: pd.DataFrame,
    labels,
    sample_size: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Calcule les métriques du clustering sur un échantillon.
    """

    X_clean = validate_clustering_input(X)

    labels = np.array(labels)

    if len(set(labels)) <= 1:
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "sample_size": 0,
            "n_clusters": 1,
            "message": "Une seule classe détectée."
        }

    positions = sample_positions(
        n_rows=len(X_clean),
        sample_size=sample_size,
        random_state=random_state
    )

    X_sample = X_clean.iloc[positions]
    labels_sample = labels[positions]

    metrics = {}

    try:
        metrics["silhouette_score"] = round(
            silhouette_score(X_sample, labels_sample),
            4
        )
    except Exception:
        metrics["silhouette_score"] = None

    try:
        metrics["davies_bouldin_score"] = round(
            davies_bouldin_score(X_sample, labels_sample),
            4
        )
    except Exception:
        metrics["davies_bouldin_score"] = None

    metrics["sample_size"] = len(X_sample)
    metrics["n_clusters"] = len(set(labels_sample))

    return metrics


# ============================================================
# 6. KMEANS CLASSIQUE
# ============================================================

def run_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Exécute K-Means classique.
    """

    X_clean = validate_clustering_input(X)

    if len(X_clean) < n_clusters:
        n_clusters = max(1, len(X_clean))

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    labels = model.fit_predict(X_clean)

    metrics = compute_clustering_metrics_sampled(
        X=X_clean,
        labels=labels,
        random_state=random_state
    )

    return {
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "cluster_centers": model.cluster_centers_,
        "metrics": metrics,
        "algorithm": "KMeans"
    }


# ============================================================
# 7. MINIBATCH KMEANS POUR GRANDS DATASETS
# ============================================================

def run_minibatch_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 3,
    batch_size: int = 4096,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Exécute MiniBatchKMeans pour les grands datasets.
    """

    X_clean = validate_clustering_input(X)

    if len(X_clean) < n_clusters:
        n_clusters = max(1, len(X_clean))

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init=10,
        max_iter=100
    )

    labels = model.fit_predict(X_clean)

    metrics = compute_clustering_metrics_sampled(
        X=X_clean,
        labels=labels,
        sample_size=10000,
        random_state=random_state
    )

    return {
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "cluster_centers": model.cluster_centers_,
        "metrics": metrics,
        "algorithm": "MiniBatchKMeans",
        "batch_size": batch_size
    }


# ============================================================
# 8. RECHERCHE AUTOMATIQUE DU MEILLEUR K (PAR SILHOUETTE)
# ============================================================

def find_best_k_by_silhouette(
    X: pd.DataFrame,
    min_k: int = 2,
    max_k: int = 10,
    sample_size: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Recherche un bon nombre de clusters avec le score silhouette
    sur un échantillon.

    min_k / max_k définissent la fenêtre de recherche : cette fonction
    est utilisée à la fois pour une recherche large (fallback complet)
    et pour un raffinement ciblé autour d'un K de départ (cf.
    run_clustering_pipeline).
    """

    X_clean = validate_clustering_input(X)

    if len(X_clean) < 3:
        return {
            "best_k": 1,
            "scores": {},
            "message": "Nombre de lignes insuffisant pour silhouette."
        }

    X_sample = sample_dataframe(
        X=X_clean,
        sample_size=sample_size,
        random_state=random_state
    )

    # Garantit une fenêtre de recherche valide même après le clipping.
    min_k = max(2, min_k)
    max_k = min(max_k, len(X_sample) - 1)

    if max_k < min_k:
        max_k = min_k

    scores = {}
    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):

        try:
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
                batch_size=2048,
                max_iter=100
            )

            labels = model.fit_predict(X_sample)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(X_sample, labels)
            scores[k] = round(score, 4)

            if score > best_score:
                best_score = score
                best_k = k

        except Exception:
            continue

    return {
        "best_k": best_k,
        "scores": scores,
        "best_score": round(best_score, 4) if best_score != -1 else None,
        "sample_size": len(X_sample)
    }


# ============================================================
# 9. PROFILAGE DES CLUSTERS
# ============================================================

def profile_clusters(
    X: pd.DataFrame,
    labels
) -> pd.DataFrame:
    """
    Génère un profil statistique simple pour chaque cluster.
    """

    X_clean = validate_clustering_input(X)

    labels = np.array(labels)

    df_profile = X_clean.copy()
    df_profile["cluster"] = labels

    profiles = []
    global_means = X_clean.mean()

    for cluster_id in sorted(df_profile["cluster"].unique()):

        cluster_data = df_profile[df_profile["cluster"] == cluster_id]
        cluster_size = len(cluster_data)

        cluster_means = cluster_data.drop(columns=["cluster"]).mean()

        dominant_features = (
            (cluster_means - global_means)
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .index
            .tolist()
        )

        profile = {
            "cluster": int(cluster_id),
            "size": int(cluster_size),
            "percentage": round((cluster_size / len(df_profile)) * 100, 2),
            "dominant_features": dominant_features
        }

        for feature in dominant_features:
            profile[f"{feature}_mean"] = round(cluster_means[feature], 4)

        profiles.append(profile)

    return pd.DataFrame(profiles)


# ============================================================
# 10. PCA POUR VISUALISATION
# ============================================================

def reduce_to_2d_pca(
    X: pd.DataFrame,
    labels=None
) -> pd.DataFrame:
    """
    Réduit les données en 2 dimensions avec PCA pour visualisation.
    """

    X_clean = validate_clustering_input(X)

    if X_clean.shape[1] < 2:
        result = pd.DataFrame({
            "PC1": X_clean.iloc[:, 0],
            "PC2": 0
        })
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_clean)

        result = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1]
        })

    if labels is not None:
        result["cluster"] = labels

    return result


# ============================================================
# 11. PIPELINE COMPLET DE CLUSTERING OPTIMISE
# ============================================================

def run_clustering_pipeline(
    X_ready: pd.DataFrame,
    use_hdbscan: bool = True,
    use_silhouette_fallback: bool = True,
    default_k: int = 3,
    min_cluster_size: int = 100,
    max_k: int = 10,
    random_state: int = 42,
    large_dataset_threshold: int = 100000,
    hdbscan_sample_size: int = 30000,
    pca_sample_size: int = 10000,
    metrics_sample_size: int = 10000,
    batch_size: int = 4096,
    high_noise_ratio_threshold: float = 40.0,
    k_refine_window: int = 1
) -> Dict[str, Any]:
    """
    Pipeline complet de segmentation optimisé pour petits et grands datasets.

    Stratégie :
        - HDBSCAN sur échantillon si dataset volumineux, pour obtenir une
          première hypothèse de structure (K de départ + taux de bruit)
        - Le K suggéré par HDBSCAN n'est JAMAIS utilisé tel quel pour
          KMeans : il sert de centre à une fenêtre de recherche silhouette
          (k_hdbscan - k_refine_window, k_hdbscan + k_refine_window),
          car un K "naturel" pour un algorithme density-based n'est pas
          nécessairement optimal pour un algorithme à centroïdes.
        - Si HDBSCAN échoue (0/1 cluster, indisponible, ou désactivé),
          une recherche silhouette complète (min_k -> max_k) est lancée.
        - MiniBatchKMeans sur tout le dataset si dataset volumineux,
          KMeans classique sinon.
        - PCA sur échantillon pour dashboard.
        - Métriques sur échantillon.
        - Le taux de bruit HDBSCAN est remonté au niveau racine du
          résultat, car un taux élevé (> high_noise_ratio_threshold)
          est un signal utile pour le technicien : il indique que les
          données sont peu structurées, indépendamment du K retenu.
    """

    X_clean = validate_clustering_input(X_ready)

    is_large_dataset = len(X_clean) >= large_dataset_threshold

    # --------------------------------------------------------
    # 1. HDBSCAN (hypothèse de structure, pas une décision finale)
    # --------------------------------------------------------

    if use_hdbscan:

        if is_large_dataset:
            hdbscan_result = run_hdbscan_on_sample(
                X=X_clean,
                sample_size=hdbscan_sample_size,
                min_cluster_size=min_cluster_size,
                random_state=random_state
            )
        else:
            hdbscan_result = run_hdbscan(
                X=X_clean,
                min_cluster_size=min_cluster_size
            )

    else:
        hdbscan_result = {
            "available": False,
            "model": None,
            "labels": None,
            "n_clusters": None,
            "n_noise": None,
            "noise_ratio": None,
            "message": "HDBSCAN désactivé."
        }

    noise_ratio = hdbscan_result.get("noise_ratio")
    high_noise_warning = (
        noise_ratio is not None
        and noise_ratio >= high_noise_ratio_threshold
    )

    # --------------------------------------------------------
    # 2. K de départ suggéré par HDBSCAN (simple hypothèse)
    # --------------------------------------------------------

    k_hdbscan = determine_k_from_hdbscan(
        hdbscan_result=hdbscan_result,
        default_k=default_k,
        min_k=2,
        max_k=max_k
    )

    hdbscan_usable = (
        hdbscan_result.get("n_clusters") is not None
        and hdbscan_result.get("n_clusters") >= 2
    )

    # --------------------------------------------------------
    # 3. Validation / affinement systématique par silhouette
    # --------------------------------------------------------
    #
    # Correction par rapport à la version initiale : on ne se contente
    # plus de réutiliser k_hdbscan tel quel dès qu'HDBSCAN renvoie au
    # moins 2 clusters. On confronte toujours ce K à une recherche
    # silhouette :
    #   - recherche ciblée autour de k_hdbscan si HDBSCAN est exploitable
    #   - recherche complète (min_k -> max_k) sinon
    # --------------------------------------------------------

    if use_silhouette_fallback and len(X_clean) >= 3:

        if hdbscan_usable:
            search_min_k = max(2, k_hdbscan - k_refine_window)
            search_max_k = min(max_k, k_hdbscan + k_refine_window)
        else:
            search_min_k = 2
            search_max_k = max_k

        best_k_result = find_best_k_by_silhouette(
            X=X_clean,
            min_k=search_min_k,
            max_k=search_max_k,
            sample_size=metrics_sample_size,
            random_state=random_state
        )

        k = best_k_result.get("best_k", k_hdbscan)

    else:
        best_k_result = None
        k = k_hdbscan

    # --------------------------------------------------------
    # 4. KMeans ou MiniBatchKMeans avec le K validé
    # --------------------------------------------------------

    if is_large_dataset:
        kmeans_result = run_minibatch_kmeans(
            X=X_clean,
            n_clusters=k,
            batch_size=batch_size,
            random_state=random_state
        )
    else:
        kmeans_result = run_kmeans(
            X=X_clean,
            n_clusters=k,
            random_state=random_state
        )

    labels = kmeans_result["labels"]

    # --------------------------------------------------------
    # 5. Métriques sur échantillon
    # --------------------------------------------------------

    metrics = compute_clustering_metrics_sampled(
        X=X_clean,
        labels=labels,
        sample_size=metrics_sample_size,
        random_state=random_state
    )

    kmeans_result["metrics"] = metrics

    # --------------------------------------------------------
    # 6. Profils sur tout le dataset
    # --------------------------------------------------------

    cluster_profiles = profile_clusters(
        X=X_clean,
        labels=labels
    )

    # --------------------------------------------------------
    # 7. PCA sur échantillon
    # --------------------------------------------------------

    positions = sample_positions(
        n_rows=len(X_clean),
        sample_size=pca_sample_size,
        random_state=random_state
    )

    X_pca_sample = X_clean.iloc[positions]
    labels_pca_sample = labels[positions]

    pca_2d = reduce_to_2d_pca(
        X=X_pca_sample,
        labels=labels_pca_sample
    )

    result = {
        "hdbscan": hdbscan_result,
        "noise_ratio": noise_ratio,
        "high_noise_warning": high_noise_warning,
        "k_hdbscan_suggestion": k_hdbscan,
        "best_k_search": best_k_result,
        "kmeans": kmeans_result,
        "labels": labels,
        "n_clusters": kmeans_result["n_clusters"],
        "cluster_profiles": cluster_profiles,
        "pca_2d": pca_2d,
        "metrics": metrics,
        "is_large_dataset": is_large_dataset,
        "algorithm_used": kmeans_result.get("algorithm"),
        "config": {
            "large_dataset_threshold": large_dataset_threshold,
            "hdbscan_sample_size": hdbscan_sample_size,
            "pca_sample_size": pca_sample_size,
            "metrics_sample_size": metrics_sample_size,
            "batch_size": batch_size,
            "min_cluster_size": min_cluster_size,
            "max_k": max_k,
            "high_noise_ratio_threshold": high_noise_ratio_threshold,
            "k_refine_window": k_refine_window
        }
    }

    return result


# ============================================================
# 12. SAUVEGARDE DES ARTEFACTS
# ============================================================

def save_clustering_artifacts(
    clustering_result: Dict[str, Any],
    output_dir: str = "models/clustering"
) -> Dict[str, str]:
    """
    Sauvegarde les artefacts de clustering.
    """

    os.makedirs(output_dir, exist_ok=True)

    kmeans_model_path = os.path.join(output_dir, "kmeans_model.pkl")
    clustering_summary_path = os.path.join(output_dir, "clustering_summary.pkl")

    joblib.dump(
        clustering_result["kmeans"]["model"],
        kmeans_model_path
    )

    summary = {
        "n_clusters": clustering_result["n_clusters"],
        "metrics": clustering_result["metrics"],
        "cluster_profiles": clustering_result["cluster_profiles"],
        "pca_2d": clustering_result["pca_2d"],
        "algorithm_used": clustering_result["algorithm_used"],
        "is_large_dataset": clustering_result["is_large_dataset"],
        "noise_ratio": clustering_result["noise_ratio"],
        "high_noise_warning": clustering_result["high_noise_warning"],
        "k_hdbscan_suggestion": clustering_result["k_hdbscan_suggestion"],
        "config": clustering_result["config"]
    }

    joblib.dump(summary, clustering_summary_path)

    return {
        "kmeans_model_path": kmeans_model_path,
        "clustering_summary_path": clustering_summary_path
    }


# ============================================================
# 13. CHARGEMENT MODELE KMEANS
# ============================================================

def load_kmeans_model(model_path: str):
    """
    Charge un modèle K-Means sauvegardé.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    return joblib.load(model_path)


# ============================================================
# 14. PREDICTION CLUSTERS POUR NOUVELLES DONNEES
# ============================================================

def predict_clusters(
    X_new: pd.DataFrame,
    kmeans_model
) -> np.ndarray:
    """
    Affecte de nouvelles données à des clusters existants.
    """

    X_clean = validate_clustering_input(X_new)

    return kmeans_model.predict(X_clean)


# ============================================================
# 15. TEST LOCAL RAPIDE
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)

    X_test = pd.DataFrame({
        "age": np.random.normal(30, 5, 30000),
        "amount_sum": np.random.normal(500, 100, 30000),
        "login_count": np.random.normal(10, 2, 30000)
    })

    result = run_clustering_pipeline(
        X_ready=X_test,
        use_hdbscan=True,
        default_k=3,
        large_dataset_threshold=10000,
        hdbscan_sample_size=5000,
        pca_sample_size=3000,
        metrics_sample_size=3000,
        min_cluster_size=50
    )

    print("===== RESULTAT CLUSTERING OPTIMISE =====")
    print("Algorithme utilisé :", result["algorithm_used"])
    print("Large dataset :", result["is_large_dataset"])
    print("Nombre de clusters :", result["n_clusters"])
    print("K suggéré par HDBSCAN :", result["k_hdbscan_suggestion"])
    print("Taux de bruit HDBSCAN :", result["noise_ratio"])
    print("Alerte bruit élevé :", result["high_noise_warning"])
    print("Métriques :", result["metrics"])

    print("\nProfils :")
    print(result["cluster_profiles"])

    print("\nPCA shape :")
    print(result["pca_2d"].shape)

    paths = save_clustering_artifacts(result)

    print("\nArtefacts sauvegardés :")
    print(paths)