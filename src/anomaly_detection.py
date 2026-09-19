# ============================================================
# anomaly_detection.py
# Module Machine Learning : détection d'anomalies
# ============================================================

import os
import joblib
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


# ============================================================
# 1. VALIDATION DES DONNEES
# ============================================================

def validate_anomaly_input(X: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie et prépare les données pour la détection d'anomalies.

    Args:
        X (pd.DataFrame): Dataset préparé.

    Returns:
        pd.DataFrame: Dataset numérique propre.
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
        raise ValueError("Aucune colonne numérique disponible pour la détection d'anomalies.")

    X_clean = X_clean[numeric_cols]

    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(0)

    return X_clean


# ============================================================
# 2. ISOLATION FOREST
# ============================================================

def run_isolation_forest(
    X: pd.DataFrame,
    contamination: str | float = "auto",
    n_estimators: int = 200,
    max_samples: str | int | float = "auto",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Exécute Isolation Forest pour détecter les anomalies.

    Args:
        X (pd.DataFrame): Dataset préparé.
        contamination: proportion attendue d'anomalies ou 'auto'.
        n_estimators (int): Nombre d'arbres.
        max_samples: nombre d'observations utilisées par arbre.
        random_state (int): Graine aléatoire.

    Returns:
        dict: Résultat complet du modèle.
    """

    X_clean = validate_anomaly_input(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1
    )

    raw_predictions = model.fit_predict(X_clean)

    # IsolationForest retourne :
    #  1  = normal
    # -1  = anomalie
    anomaly_labels = np.where(raw_predictions == -1, 1, 0)

    # decision_function : plus la valeur est faible, plus c'est anormal
    decision_scores = model.decision_function(X_clean)

    # score_samples : score brut du modèle
    raw_scores = model.score_samples(X_clean)

    anomaly_scores = normalize_anomaly_scores(decision_scores)

    result_df = build_anomaly_result_dataframe(
        X=X_clean,
        anomaly_labels=anomaly_labels,
        anomaly_scores=anomaly_scores,
        decision_scores=decision_scores,
        raw_scores=raw_scores
    )

    summary = generate_anomaly_summary(result_df)

    return {
        "model": model,
        "result_df": result_df,
        "anomaly_labels": anomaly_labels,
        "anomaly_scores": anomaly_scores,
        "decision_scores": decision_scores,
        "raw_scores": raw_scores,
        "summary": summary,
        "config": {
            "contamination": contamination,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "random_state": random_state
        }
    }


# ============================================================
# 3. NORMALISATION DU SCORE D'ANOMALIE
# ============================================================

def normalize_anomaly_scores(decision_scores: np.ndarray) -> np.ndarray:
    """
    Convertit les scores Isolation Forest en score d'anomalie normalisé entre 0 et 1.

    Plus le score est proche de 1, plus l'observation est anormale.
    """

    # decision_scores :
    # valeurs faibles -> anomalies
    inverted_scores = -decision_scores

    min_score = np.min(inverted_scores)
    max_score = np.max(inverted_scores)

    if max_score == min_score:
        return np.zeros_like(inverted_scores)

    normalized = (inverted_scores - min_score) / (max_score - min_score)

    return normalized


# ============================================================
# 4. NIVEAUX DE SEVERITE
# ============================================================

def assign_anomaly_severity(
    anomaly_score: float,
    is_anomaly: int,
    thresholds: Optional[Dict[str, float]] = None
) -> str:
    """
    Attribue un niveau de sévérité à une observation.

    Args:
        anomaly_score (float): Score entre 0 et 1.
        is_anomaly (int): 1 si anomalie, 0 sinon.
        thresholds (dict): seuils personnalisés.

    Returns:
        str: niveau de sévérité.
    """

    if is_anomaly == 0:
        return "normal"

    if thresholds is None:
        thresholds = {
            "faible": 0.50,
            "moyen": 0.70,
            "élevé": 0.85,
            "critique": 0.95
        }

    if anomaly_score >= thresholds["critique"]:
        return "critique"

    if anomaly_score >= thresholds["élevé"]:
        return "élevé"

    if anomaly_score >= thresholds["moyen"]:
        return "moyen"

    return "faible"


# ============================================================
# 5. CONSTRUCTION DU RESULTAT DETAILLE
# ============================================================

def build_anomaly_result_dataframe(
    X: pd.DataFrame,
    anomaly_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    decision_scores: np.ndarray,
    raw_scores: np.ndarray
) -> pd.DataFrame:
    """
    Construit un DataFrame contenant les résultats d'anomalies.
    """

    result_df = X.copy()

    result_df["is_anomaly"] = anomaly_labels
    result_df["anomaly_score"] = anomaly_scores
    result_df["decision_score"] = decision_scores
    result_df["raw_score"] = raw_scores

    result_df["severity"] = [
        assign_anomaly_severity(score, label)
        for score, label in zip(anomaly_scores, anomaly_labels)
    ]

    return result_df


# ============================================================
# 6. RESUME DES ANOMALIES
# ============================================================

def generate_anomaly_summary(result_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Génère un résumé global des anomalies détectées.
    """

    total_rows = len(result_df)
    anomaly_count = int(result_df["is_anomaly"].sum())

    anomaly_ratio = round((anomaly_count / total_rows) * 100, 2) if total_rows > 0 else 0

    severity_counts = (
        result_df["severity"]
        .value_counts()
        .to_dict()
    )

    avg_anomaly_score = round(
        result_df["anomaly_score"].mean(),
        4
    )

    max_anomaly_score = round(
        result_df["anomaly_score"].max(),
        4
    )

    summary = {
        "total_rows": total_rows,
        "anomaly_count": anomaly_count,
        "anomaly_ratio": anomaly_ratio,
        "severity_counts": severity_counts,
        "avg_anomaly_score": avg_anomaly_score,
        "max_anomaly_score": max_anomaly_score
    }

    return summary


# ============================================================
# 7. EXTRACTION DES ANOMALIES
# ============================================================

def get_anomalies_only(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne uniquement les observations détectées comme anomalies.
    """

    if result_df is None or result_df.empty:
        return pd.DataFrame()

    return result_df[result_df["is_anomaly"] == 1].copy()


def get_top_anomalies(
    result_df: pd.DataFrame,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Retourne les anomalies les plus fortes selon le score.
    """

    anomalies = get_anomalies_only(result_df)

    if anomalies.empty:
        return anomalies

    return (
        anomalies
        .sort_values(by="anomaly_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 8. PCA POUR VISUALISATION DES ANOMALIES
# ============================================================

def reduce_anomalies_to_2d_pca(
    X: pd.DataFrame,
    anomaly_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    sample_size: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Réduit les données en 2 dimensions avec PCA pour visualiser les anomalies.

    Pour les grands datasets, la visualisation est faite sur un échantillon.
    """

    X_clean = validate_anomaly_input(X)

    if len(X_clean) > sample_size:
        sample_idx = X_clean.sample(
            n=sample_size,
            random_state=random_state
        ).index
    else:
        sample_idx = X_clean.index

    X_sample = X_clean.loc[sample_idx]

    positions = X_clean.index.get_indexer(sample_idx)

    labels_sample = anomaly_labels[positions]
    scores_sample = anomaly_scores[positions]

    if X_sample.shape[1] < 2:
        pca_df = pd.DataFrame({
            "PC1": X_sample.iloc[:, 0],
            "PC2": 0
        })
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_sample)

        pca_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1]
        })

    pca_df["is_anomaly"] = labels_sample
    pca_df["anomaly_score"] = scores_sample
    pca_df["type"] = np.where(labels_sample == 1, "anomalie", "normal")

    return pca_df.reset_index(drop=True)


# ============================================================
# 9. PIPELINE COMPLET DE DETECTION D'ANOMALIES
# ============================================================

def run_anomaly_detection_pipeline(
    X_ready: pd.DataFrame,
    contamination: str | float = "auto",
    n_estimators: int = 200,
    max_samples: str | int | float = "auto",
    pca_sample_size: int = 10000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Exécute le pipeline complet de détection d'anomalies.

    Étapes :
        1. Validation des données.
        2. Isolation Forest.
        3. Scores d'anomalies.
        4. Niveaux de sévérité.
        5. PCA pour visualisation.
        6. Résumé global.

    Returns:
        dict: Résultats complets.
    """

    X_clean = validate_anomaly_input(X_ready)

    isolation_result = run_isolation_forest(
        X=X_clean,
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state
    )

    result_df = isolation_result["result_df"]

    pca_2d = reduce_anomalies_to_2d_pca(
        X=X_clean,
        anomaly_labels=isolation_result["anomaly_labels"],
        anomaly_scores=isolation_result["anomaly_scores"],
        sample_size=pca_sample_size,
        random_state=random_state
    )

    top_anomalies = get_top_anomalies(
        result_df=result_df,
        top_n=50
    )

    return {
        "model": isolation_result["model"],
        "result_df": result_df,
        "anomaly_labels": isolation_result["anomaly_labels"],
        "anomaly_scores": isolation_result["anomaly_scores"],
        "summary": isolation_result["summary"],
        "top_anomalies": top_anomalies,
        "pca_2d": pca_2d,
        "config": {
            "contamination": contamination,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "pca_sample_size": pca_sample_size,
            "random_state": random_state
        }
    }


# ============================================================
# 10. SAUVEGARDE DES ARTEFACTS
# ============================================================

def save_anomaly_artifacts(
    anomaly_result: Dict[str, Any],
    output_dir: str = "models/anomaly_detection"
) -> Dict[str, str]:
    """
    Sauvegarde les artefacts de détection d'anomalies.
    """

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "isolation_forest_model.pkl")
    summary_path = os.path.join(output_dir, "anomaly_summary.pkl")

    joblib.dump(
        anomaly_result["model"],
        model_path
    )

    summary = {
        "summary": anomaly_result["summary"],
        "config": anomaly_result["config"]
    }

    joblib.dump(
        summary,
        summary_path
    )

    return {
        "isolation_forest_model_path": model_path,
        "anomaly_summary_path": summary_path
    }


def load_anomaly_model(model_path: str):
    """
    Charge un modèle Isolation Forest sauvegardé.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    return joblib.load(model_path)


# ============================================================
# 11. PREDICTION SUR NOUVELLES DONNEES
# ============================================================

def predict_anomalies(
    X_new: pd.DataFrame,
    model
) -> pd.DataFrame:
    """
    Prédit les anomalies sur de nouvelles données avec un modèle sauvegardé.
    """

    X_clean = validate_anomaly_input(X_new)

    raw_predictions = model.predict(X_clean)

    anomaly_labels = np.where(raw_predictions == -1, 1, 0)

    decision_scores = model.decision_function(X_clean)
    raw_scores = model.score_samples(X_clean)
    anomaly_scores = normalize_anomaly_scores(decision_scores)

    result_df = build_anomaly_result_dataframe(
        X=X_clean,
        anomaly_labels=anomaly_labels,
        anomaly_scores=anomaly_scores,
        decision_scores=decision_scores,
        raw_scores=raw_scores
    )

    return result_df


# ============================================================
# 12. TEST LOCAL RAPIDE
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)

    normal_data = pd.DataFrame({
        "age": np.random.normal(30, 5, 500),
        "amount_sum": np.random.normal(500, 100, 500),
        "login_count": np.random.normal(10, 2, 500)
    })

    anomalous_data = pd.DataFrame({
        "age": np.random.normal(70, 3, 20),
        "amount_sum": np.random.normal(3000, 200, 20),
        "login_count": np.random.normal(1, 0.5, 20)
    })

    X_test = pd.concat(
        [normal_data, anomalous_data],
        ignore_index=True
    )

    result = run_anomaly_detection_pipeline(
        X_ready=X_test,
        contamination=0.05,
        n_estimators=200,
        pca_sample_size=500
    )

    print("===== RESULTAT DETECTION ANOMALIES =====")
    print("Résumé :")
    print(result["summary"])

    print("\nTop anomalies :")
    print(result["top_anomalies"].head())

    paths = save_anomaly_artifacts(result)

    print("\nArtefacts sauvegardés :")
    print(paths)