# ============================================================
# interpretation.py
# Couche d'interprétation automatique (clusters, anomalies, KPIs)
# Cible : administrateurs / managers non techniques.
#
# Principe : 100% rule-based (aucun appel externe), cohérent avec
# le reste du pipeline (assign_anomaly_severity dans
# anomaly_detection.py fonctionne déjà par seuils). Ce module ne
# fait qu'ajouter une couche de texte et d'estimation "valeurs
# réelles" au-dessus des résultats déjà calculés par clustering.py
# et anomaly_detection.py — il ne modifie aucune logique ML.
# ============================================================

import os
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, MiniBatchKMeans


# ============================================================
# 1. BANDES DE QUALITE (SEUILS)
# ============================================================

SILHOUETTE_BANDS = [
    (0.50, "excellente", "success"),
    (0.25, "correcte", "success"),
    (0.10, "faible", "warning"),
    (-1.01, "très faible / clusters non séparés", "danger"),
]

NOISE_BANDS = [
    (60.0, "critique", "danger"),
    (40.0, "élevé", "warning"),
    (15.0, "modéré", "neutral"),
    (-0.01, "faible", "success"),
]

ANOMALY_RATIO_BANDS = [
    (10.0, "critique", "danger"),
    (5.0, "élevé", "warning"),
    (1.0, "modéré", "neutral"),
    (-0.01, "faible", "success"),
]


def _band_lookup(value: Optional[float], bands: List[tuple]) -> tuple:
    """Retourne (label, tone) pour une valeur selon des bandes triées desc."""

    if value is None:
        return ("indisponible", "neutral")

    for threshold, label, tone in bands:
        if value >= threshold:
            return (label, tone)

    return (bands[-1][1], bands[-1][2])


# ============================================================
# 2. INTERPRETATION DU CLUSTERING
# ============================================================

def interpret_cluster_quality(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Traduit le score silhouette en appréciation qualitative.
    """

    metrics = metrics or {}
    score = metrics.get("silhouette_score")

    label, tone = _band_lookup(score, SILHOUETTE_BANDS)

    if score is None:
        text = (
            "Le score de séparation des segments n'a pas pu être calculé "
            "(volume de données insuffisant ou un seul segment détecté)."
        )
    else:
        text = (
            f"La séparation entre les segments utilisateurs est jugée {label} "
            f"(score silhouette = {score})."
        )

    return {"label": label, "tone": tone, "text": text, "score": score}


def interpret_noise_ratio(noise_ratio: Optional[float]) -> Dict[str, Any]:
    """
    Traduit le taux de bruit HDBSCAN en signal de structure des données.
    """

    label, tone = _band_lookup(noise_ratio, NOISE_BANDS)

    if noise_ratio is None:
        text = "L'estimation du bruit structurel n'est pas disponible."
    else:
        text = (
            f"Le taux de bruit détecté par HDBSCAN est {label} ({noise_ratio}%). "
        )
        if tone in ("warning", "danger"):
            text += (
                "Cela suggère que les données sont peu structurées : "
                "les segments proposés peuvent être moins fiables, et "
                "l'ajout de features plus discriminantes est recommandé."
            )
        else:
            text += "Les données présentent une structure exploitable pour la segmentation."

    return {"label": label, "tone": tone, "text": text, "noise_ratio": noise_ratio}


def _weighted_global_means(cluster_profiles_df: pd.DataFrame) -> Dict[str, float]:
    """
    Reconstitue une moyenne globale approximative par feature à partir
    des moyennes par cluster pondérées par la taille de chaque cluster.
    Utilisé pour donner une direction (au-dessus / en-dessous) aux
    features dominantes de chaque cluster, sans devoir modifier
    clustering.py.
    """

    if cluster_profiles_df is None or cluster_profiles_df.empty:
        return {}

    total_size = cluster_profiles_df["size"].sum()

    if total_size == 0:
        return {}

    mean_cols = [c for c in cluster_profiles_df.columns if c.endswith("_mean")]
    global_means = {}

    for col in mean_cols:
        weighted_sum = (cluster_profiles_df[col] * cluster_profiles_df["size"]).sum()
        global_means[col.replace("_mean", "")] = weighted_sum / total_size

    return global_means


def generate_cluster_narratives(cluster_profiles_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Génère un texte descriptif par cluster à partir de cluster_profiles.
    """

    if cluster_profiles_df is None or cluster_profiles_df.empty:
        return []

    global_means = _weighted_global_means(cluster_profiles_df)
    narratives = []

    for _, row in cluster_profiles_df.iterrows():

        dominant_features = row.get("dominant_features", []) or []
        traits = []

        for feature in dominant_features:
            mean_col = f"{feature}_mean"

            if mean_col not in row or pd.isna(row[mean_col]):
                continue

            cluster_value = row[mean_col]
            global_value = global_means.get(feature)

            if global_value is not None and global_value != 0:
                direction = "au-dessus" if cluster_value > global_value else "en-dessous"
                traits.append(f"**{feature}** ({direction} de la moyenne des segments)")
            else:
                traits.append(f"**{feature}**")

        if traits:
            traits_text = ", ".join(traits[:3])
            text = (
                f"Ce segment représente {row.get('percentage')}% des utilisateurs "
                f"({row.get('size')} lignes) et se distingue principalement par : {traits_text}."
            )
        else:
            text = (
                f"Ce segment représente {row.get('percentage')}% des utilisateurs "
                f"({row.get('size')} lignes). Aucune feature dominante claire n'a été identifiée."
            )

        narratives.append({
            "cluster": int(row.get("cluster")),
            "size": int(row.get("size")),
            "percentage": float(row.get("percentage")),
            "title": f"Segment {int(row.get('cluster'))}",
            "text": text,
            "dominant_features": dominant_features
        })

    return narratives


# ============================================================
# 3. VALEURS REELLES (DE-SCALING) POUR LES PROFILS DE CLUSTERS
# ============================================================

def _unscale_value(feature: str, scaled_value: float, artifacts: Dict[str, Any]) -> Optional[float]:
    """
    Inverse la mise à l'échelle (et la correction de skewness si
    applicable) pour une valeur scalaire d'une feature numérique,
    en utilisant les artefacts sauvegardés par fit_pipeline
    (preprocessing.py). Ne nécessite pas de recharger tout le
    dataset : StandardScaler / RobustScaler sont appliqués colonne
    par colonne, donc l'inversion peut se faire feature par feature.
    """

    scaler = artifacts.get("scaler")
    scale_cols = artifacts.get("numeric_cols_at_scaling", [])
    skew_transformers = artifacts.get("skew_transformers", {})

    if scaler is None or feature not in scale_cols:
        return None

    try:
        idx = scale_cols.index(feature)

        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_") and not hasattr(scaler, "center_"):
            value = scaled_value * scaler.scale_[idx] + scaler.mean_[idx]
        elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
            value = scaled_value * scaler.scale_[idx] + scaler.center_[idx]
        else:
            return None

        skew_info = skew_transformers.get(feature)

        if skew_info:
            method = skew_info.get("method")

            if method == "Log1p":
                value = np.expm1(value)

            elif method == "Yeo-Johnson":
                transformer = skew_info.get("transformer")
                if transformer is not None:
                    value = transformer.inverse_transform([[value]])[0][0]

        return round(float(value), 3)

    except Exception:
        return None


def enrich_cluster_profiles_with_real_values(
    cluster_profiles_df: pd.DataFrame,
    artifacts_path: Optional[str]
) -> pd.DataFrame:
    """
    Ajoute, quand c'est possible, une estimation en unités réelles
    (avant scaling / transformation de skewness) à côté des moyennes
    scalées pour les features dominantes de chaque cluster.

    C'est une ESTIMATION : le winsorizing des outliers n'est pas
    inversible, donc la valeur réelle peut légèrement différer de la
    donnée brute d'origine.
    """

    df = cluster_profiles_df.copy() if cluster_profiles_df is not None else pd.DataFrame()

    if df.empty or not artifacts_path or not os.path.exists(artifacts_path):
        return df

    try:
        artifacts = joblib.load(artifacts_path)
    except Exception:
        return df

    for idx, row in df.iterrows():
        for feature in (row.get("dominant_features") or []):
            mean_col = f"{feature}_mean"

            if mean_col not in df.columns or pd.isna(row[mean_col]):
                continue

            real_value = _unscale_value(feature, row[mean_col], artifacts)

            if real_value is not None:
                df.loc[idx, f"{feature}_estimation_reelle"] = real_value

    return df


# ============================================================
# 4. INTERPRETATION DES ANOMALIES
# ============================================================

def interpret_anomaly_ratio(anomaly_ratio: Optional[float]) -> Dict[str, Any]:
    """
    Traduit le ratio d'anomalies en appréciation qualitative.
    """

    label, tone = _band_lookup(anomaly_ratio, ANOMALY_RATIO_BANDS)

    if anomaly_ratio is None:
        text = "Le taux d'anomalies n'est pas disponible."
    else:
        text = f"Le taux d'anomalies détecté est {label} ({anomaly_ratio}%). "

        if tone == "danger":
            text += (
                "Un taux aussi élevé est inhabituel pour de la détection d'anomalies "
                "métier : il est probable qu'il reflète un problème de qualité de "
                "données (colonnes mal renseignées, doublons partiels) plutôt que de "
                "vraies anomalies. Vérifier la contamination configurée."
            )
        elif tone == "warning":
            text += "Ce niveau mérite une revue par un analyste avant action."
        else:
            text += "Ce niveau est cohérent avec un usage standard de détection d'anomalies."

    return {"label": label, "tone": tone, "text": text, "anomaly_ratio": anomaly_ratio}


def generate_anomaly_narrative(anomaly_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Génère un texte de synthèse sur les anomalies détectées.
    """

    anomaly_summary = anomaly_summary or {}
    severity_counts = anomaly_summary.get("severity_counts", {}) or {}

    critical = severity_counts.get("critique", 0)
    high = severity_counts.get("élevé", 0)

    ratio_interpretation = interpret_anomaly_ratio(anomaly_summary.get("anomaly_ratio"))

    if critical > 0:
        priority_text = (
            f"{critical} observation(s) de sévérité **critique** nécessitent une "
            f"vérification prioritaire."
        )
    elif high > 0:
        priority_text = (
            f"{high} observation(s) de sévérité **élevée** sont à surveiller."
        )
    else:
        priority_text = "Aucune anomalie de sévérité critique ou élevée n'a été détectée."

    return {
        "ratio_interpretation": ratio_interpretation,
        "priority_text": priority_text,
        "critical_count": critical,
        "high_count": high
    }


# ============================================================
# 5. ANALYSE CROISEE CLUSTERS x ANOMALIES
# ============================================================

def cross_analyze_clusters_anomalies(
    clustered_dataset: Optional[pd.DataFrame],
    anomaly_dataset: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """
    Croise les résultats de segmentation et de détection d'anomalies
    afin d'identifier si certains segments concentrent anormalement
    les anomalies. Les deux datasets proviennent du même X_ready
    (même ordre / même index), donc l'alignement par position est sûr.
    """

    if clustered_dataset is None or anomaly_dataset is None:
        return None

    if len(clustered_dataset) != len(anomaly_dataset):
        return None

    if "cluster" not in clustered_dataset.columns or "is_anomaly" not in anomaly_dataset.columns:
        return None

    merged = pd.DataFrame({
        "cluster": clustered_dataset["cluster"].values,
        "is_anomaly": anomaly_dataset["is_anomaly"].values
    })

    global_ratio = merged["is_anomaly"].mean() * 100 if len(merged) > 0 else 0

    grouped = (
        merged
        .groupby("cluster")
        .agg(size=("is_anomaly", "size"), anomaly_count=("is_anomaly", "sum"))
        .reset_index()
    )

    grouped["anomaly_ratio_pct"] = round((grouped["anomaly_count"] / grouped["size"]) * 100, 2)
    grouped["global_ratio_pct"] = round(global_ratio, 2)
    grouped["ecart_vs_global_pct"] = round(grouped["anomaly_ratio_pct"] - grouped["global_ratio_pct"], 2)
    grouped["a_surveiller"] = grouped["anomaly_ratio_pct"] > (grouped["global_ratio_pct"] * 1.5 + 1)

    return grouped.sort_values(by="anomaly_ratio_pct", ascending=False).reset_index(drop=True)


# ============================================================
# 6. RECOMMANDATIONS
# ============================================================

def generate_recommendations(
    clustering_result: Optional[Dict[str, Any]],
    anomaly_result: Optional[Dict[str, Any]],
    cross_analysis: Optional[pd.DataFrame] = None,
    surrogate_fidelity: Optional[Dict[str, Any]] = None,
    cluster_stability: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """
    Génère une liste de recommandations actionnables, par ordre de
    priorité décroissante.
    """

    recommendations = []

    if clustering_result is not None:
        metrics = clustering_result.get("metrics", {})
        score = metrics.get("silhouette_score")

        if score is not None and score < 0.25:
            recommendations.append({
                "tone": "warning",
                "text": (
                    "Le score de séparation des segments est faible : envisager "
                    "d'ajouter des features métier plus discriminantes ou de revoir "
                    "le nombre de segments (K)."
                )
            })

        if clustering_result.get("high_noise_warning"):
            recommendations.append({
                "tone": "warning",
                "text": (
                    "Taux de bruit HDBSCAN élevé : les données semblent peu "
                    "structurées. Les segments actuels doivent être interprétés "
                    "avec prudence."
                )
            })

        n_clusters = clustering_result.get("n_clusters")
        cluster_profiles = clustering_result.get("cluster_profiles")

        if isinstance(cluster_profiles, pd.DataFrame) and not cluster_profiles.empty:
            smallest = cluster_profiles["percentage"].min()

            if smallest < 2:
                recommendations.append({
                    "tone": "neutral",
                    "text": (
                        f"Au moins un segment représente moins de 2% des utilisateurs : "
                        f"vérifier qu'il ne s'agit pas d'un artefact plutôt que d'un "
                        f"segment métier réel."
                    )
                })

    if anomaly_result is not None:
        summary = anomaly_result.get("summary", {})
        ratio = summary.get("anomaly_ratio")

        if ratio is not None and ratio > 10:
            recommendations.append({
                "tone": "danger",
                "text": (
                    "Taux d'anomalies supérieur à 10% : probable problème de qualité "
                    "de données plutôt que de vraies anomalies métier. Vérifier le "
                    "paramètre de contamination et la qualité des colonnes sources."
                )
            })

        severity_counts = summary.get("severity_counts", {})

        if severity_counts.get("critique", 0) > 0:
            recommendations.append({
                "tone": "danger",
                "text": (
                    f"{severity_counts.get('critique')} anomalie(s) critique(s) "
                    f"détectée(s) : à faire vérifier en priorité par un analyste métier."
                )
            })

    if cross_analysis is not None and not cross_analysis.empty:
        risky_clusters = cross_analysis[cross_analysis["a_surveiller"]]

        for _, row in risky_clusters.iterrows():
            recommendations.append({
                "tone": "warning",
                "text": (
                    f"Le segment {int(row['cluster'])} concentre {row['anomaly_ratio_pct']}% "
                    f"d'anomalies contre {row['global_ratio_pct']}% en moyenne globale : "
                    f"à investiguer en priorité."
                )
            })

    if surrogate_fidelity is not None and surrogate_fidelity.get("tone") in ("warning", "danger"):
        recommendations.append({
            "tone": surrogate_fidelity["tone"],
            "text": (
                "Les règles d'explication des segments (modèle surrogate) ont une "
                "fidélité limitée : les segments détectés ne suivent pas de "
                "frontières simples sur les features actuelles. Considérer les "
                "règles affichées comme indicatives, pas comme une explication exacte."
            )
        })

    if cluster_stability is not None and cluster_stability.get("tone") in ("warning", "danger"):
        recommendations.append({
            "tone": cluster_stability["tone"],
            "text": (
                f"Stabilité de la segmentation {cluster_stability['label']} "
                f"(ARI moyen = {cluster_stability['mean_ari']}) : la partition change "
                "significativement selon l'initialisation. Envisager plus de données, "
                "d'autres features, ou un K différent avant de communiquer les segments "
                "comme définitifs."
            )
        })

    if not recommendations:
        recommendations.append({
            "tone": "success",
            "text": "Aucun signal préoccupant détecté sur ce run : segmentation et anomalies dans des plages attendues."
        })

    return recommendations


# ============================================================
# 6bis. EXPLICABILITE DES CLUSTERS PAR MODELE SURROGATE
# ============================================================
#
# Principe : le clustering (K-Means / MiniBatchKMeans) n'a pas de
# "raison" explicite pour ses partitions. On entraîne donc un arbre de
# décision peu profond à REPRODUIRE la partition déjà calculée
# (X -> cluster), pas à prédire un label métier. Cet arbre n'est
# jamais utilisé pour classer de nouvelles données : c'est un proxy
# interprétable dont le seul rôle est de traduire le clustering en
# règles IF/THEN lisibles, avec un score de fidélité qui indique à
# quel point on peut se fier à ces règles.
# ============================================================

def _extract_leaf_paths(tree_model: DecisionTreeClassifier, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Parcourt l'arbre entraîné et retourne, pour chaque feuille, la liste
    des conditions menant à cette feuille, la classe majoritaire, la
    pureté, ainsi que la répartition complète par classe (nécessaire
    pour calculer un vrai rappel par cluster quand une feuille n'est
    pas totalement pure).
    """

    tree_ = tree_model.tree_
    classes = tree_model.classes_

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else None
        for i in tree_.feature
    ]

    paths: List[Dict[str, Any]] = []

    def recurse(node: int, conditions: List[tuple]):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            recurse(tree_.children_left[node], conditions + [(name, "<=", threshold)])
            recurse(tree_.children_right[node], conditions + [(name, ">", threshold)])

        else:
            # tree_.value contient des PROPORTIONS par classe (pas des
            # comptes bruts) dans les versions récentes de scikit-learn :
            # value.sum() == 1 pour un noeud. Le vrai nombre
            # d'observations du noeud vient de tree_.n_node_samples.
            value = tree_.value[node][0]
            total = int(tree_.n_node_samples[node])
            class_idx = int(np.argmax(value))
            purity = float(value[class_idx])

            # Comptes reconstitués par classe (arrondis à l'entier le
            # plus proche) : permet de calculer, pour un cluster donné,
            # combien de ses observations sont réellement captées par
            # cette feuille — pas juste la taille totale de la feuille.
            class_counts = {
                int(cls): int(round(prop * total))
                for cls, prop in zip(classes, value)
            }

            paths.append({
                "conditions": conditions,
                "class_idx": class_idx,
                "n_samples": total,
                "purity": round(purity, 4),
                "class_counts": class_counts
            })

    recurse(0, [])

    return paths


def _format_rule_condition(
    feature: str,
    operator: str,
    threshold_scaled: float,
    artifacts: Optional[Dict[str, Any]]
) -> str:
    """
    Formate une condition de règle en tentant de remettre le seuil en
    unités réelles (réutilise le même mécanisme de de-scaling que les
    profils de clusters).
    """

    real_value = _unscale_value(feature, threshold_scaled, artifacts) if artifacts else None

    if real_value is not None:
        return f"{feature} {operator} {round(real_value, 2)}"

    return f"{feature} {operator} {round(threshold_scaled, 3)} (valeur normalisée)"


def train_cluster_surrogate(
    clustered_dataset: Optional[pd.DataFrame],
    max_depth: int = 4,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    cv_folds: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Entraîne le modèle surrogate (arbre de décision) sur les features
    déjà utilisées par le clustering (clustered_dataset = X_ready + une
    colonne 'cluster', produit par ml_controller.attach_cluster_labels).

    Retourne None si l'explication n'est pas possible (pas assez de
    diversité de clusters, dataset vide, etc.) plutôt que de lever une
    exception : l'absence d'explication ne doit jamais casser le
    dashboard.
    """

    if clustered_dataset is None or "cluster" not in clustered_dataset.columns:
        return None

    X = clustered_dataset.drop(columns=["cluster"]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = clustered_dataset["cluster"]

    if X.empty or y.nunique() < 2:
        return None

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    tree.fit(X, y)

    fidelity_train = round(float(tree.score(X, y)), 4)

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())

    cv_fidelity = None

    if min_class_count >= 2:
        safe_cv = max(2, min(cv_folds, min_class_count))

        try:
            scores = cross_val_score(
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                ),
                X, y, cv=safe_cv
            )
            cv_fidelity = round(float(scores.mean()), 4)

        except Exception:
            cv_fidelity = None

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": tree.feature_importances_
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    leaf_paths = _extract_leaf_paths(tree, list(X.columns))

    return {
        "tree": tree,
        "feature_names": list(X.columns),
        "classes": list(tree.classes_),
        "fidelity_train": fidelity_train,
        "cv_fidelity": cv_fidelity,
        "feature_importance": feature_importance,
        "leaf_paths": leaf_paths,
        "config": {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "cv_folds": cv_folds
        }
    }


def interpret_surrogate_fidelity(
    fidelity_train: Optional[float],
    cv_fidelity: Optional[float]
) -> Dict[str, Any]:
    """
    Traduit la fidélité du surrogate (à quel point ses règles
    reproduisent le vrai clustering) en niveau de confiance à afficher
    à côté des règles, pour éviter de présenter une approximation comme
    une vérité absolue.
    """

    reference = cv_fidelity if cv_fidelity is not None else fidelity_train

    if reference is None:
        return {
            "label": "indisponible",
            "tone": "neutral",
            "text": "La fiabilité des règles d'explication n'a pas pu être évaluée.",
            "fidelity": None
        }

    if reference >= 0.90:
        label, tone = "élevée", "success"
    elif reference >= 0.75:
        label, tone = "correcte", "success"
    elif reference >= 0.60:
        label, tone = "modérée", "warning"
    else:
        label, tone = "faible", "danger"

    validation_note = (
        " (validé par validation croisée)." if cv_fidelity is not None
        else " (mesurée sur les données d'entraînement, à interpréter avec prudence)."
    )

    text = (
        f"Confiance {label} dans les règles ci-dessous : le modèle d'explication "
        f"reproduit {round(reference * 100, 1)}% des segments détectés par le "
        f"clustering{validation_note}"
    )

    if tone in ("warning", "danger"):
        text += (
            " En dessous de 75%, considérez les règles comme une approximation "
            "indicative plutôt qu'une explication exacte du clustering."
        )

    return {"label": label, "tone": tone, "text": text, "fidelity": reference}


def generate_cluster_rules(
    surrogate_result: Optional[Dict[str, Any]],
    cluster_profiles_df: Optional[pd.DataFrame] = None,
    artifacts_path: Optional[str] = None,
    max_conditions: int = 3
) -> Dict[int, Dict[str, Any]]:
    """
    Pour chaque cluster, sélectionne la feuille la plus représentative
    de l'arbre surrogate (le plus grand nombre d'observations) et la
    traduit en règle lisible IF/THEN, avec les seuils remis en unités
    réelles quand c'est possible.
    """

    if surrogate_result is None:
        return {}

    artifacts = None

    if artifacts_path and os.path.exists(artifacts_path):
        try:
            artifacts = joblib.load(artifacts_path)
        except Exception:
            artifacts = None

    classes = surrogate_result["classes"]
    leaf_paths = surrogate_result["leaf_paths"]

    size_by_cluster = {}

    if cluster_profiles_df is not None and not cluster_profiles_df.empty:
        size_by_cluster = dict(zip(cluster_profiles_df["cluster"], cluster_profiles_df["size"]))

    rules_by_cluster: Dict[int, Dict[str, Any]] = {}

    for class_idx, cluster_label in enumerate(classes):

        cluster_label_int = int(cluster_label)

        candidate_leaves = [leaf for leaf in leaf_paths if leaf["class_idx"] == class_idx]

        if not candidate_leaves:
            continue

        # On sélectionne la feuille qui capture le plus grand nombre
        # RÉEL d'observations de ce cluster précis (recall), pas juste
        # la feuille la plus grande dans l'absolu — une feuille "class 0
        # majoritaire" peut être grande tout en contenant peu de vrais
        # points du cluster 0 si sa pureté est faible.
        best_leaf = max(
            candidate_leaves,
            key=lambda leaf: leaf["class_counts"].get(cluster_label_int, 0)
        )

        conditions_text = [
            _format_rule_condition(feature, operator, threshold, artifacts)
            for feature, operator, threshold in best_leaf["conditions"][:max_conditions]
        ]

        rule_text = " ET ".join(conditions_text) if conditions_text else "aucune condition (segment racine)"

        cluster_size = size_by_cluster.get(cluster_label)
        coverage_text = ""

        true_positive_count = best_leaf["class_counts"].get(cluster_label_int, 0)

        if cluster_size:
            coverage_pct = round((true_positive_count / cluster_size) * 100, 1)
            coverage_text = f" — règle valable pour ~{coverage_pct}% des observations de ce segment"

        rules_by_cluster[int(cluster_label)] = {
            "rule_text": rule_text,
            "purity": best_leaf["purity"],
            "n_samples": best_leaf["n_samples"],
            "coverage_text": coverage_text,
            "full_text": (
                f"Règle : SI {rule_text} ALORS segment {int(cluster_label)} "
                f"(pureté {round(best_leaf['purity'] * 100, 1)}%{coverage_text})."
            )
        }

    return rules_by_cluster


# ============================================================
# 6ter. EXPLICATION LOCALE DES ANOMALIES (SHAP + repli maison)
# ============================================================
#
# Deux méthodes, une seule sortie unifiée :
#   - SHAP (shap.TreeExplainer) si le package est installé : donne une
#     contribution par feature, par observation, basée sur la théorie
#     des jeux (valeurs de Shapley).
#   - Repli "maison" inspiré de DIFFI (Depth-based Isolation Forest
#     Feature Importance) si SHAP n'est pas disponible : pour chaque
#     arbre de l'Isolation Forest, on regarde quelles features ont
#     causé les splits les plus précoces sur le chemin de
#     l'observation (un split précoce = signal d'isolement plus fort).
#     Aucune dépendance externe, mais moins rigoureux que SHAP.
#
# Dans les deux cas, l'absence de résultat ne doit jamais interrompre
# le dashboard : ces fonctions renvoient None en cas d'échec plutôt
# que de lever une exception.
# ============================================================

_ANOMALY_EXTRA_COLUMNS = (
    "is_anomaly", "anomaly_score", "decision_score", "raw_score", "severity"
)


def _get_anomaly_feature_columns(result_df: pd.DataFrame) -> List[str]:
    return [col for col in result_df.columns if col not in _ANOMALY_EXTRA_COLUMNS]


def explain_anomalies_with_shap(
    anomaly_result: Optional[Dict[str, Any]],
    artifacts_path: Optional[str] = None,
    top_n: int = 15,
    max_features: int = 3
) -> Optional[pd.DataFrame]:
    """
    Explication locale des anomalies les plus fortes via SHAP
    (shap.TreeExplainer, compatible avec IsolationForest). Retourne
    None si le package `shap` n'est pas installé ou si le calcul
    échoue pour une autre raison — c'est une amélioration optionnelle,
    jamais un pré-requis du pipeline.
    """

    try:
        import shap
    except ImportError:
        return None

    if anomaly_result is None:
        return None

    model = anomaly_result.get("model")
    result_df = anomaly_result.get("result_df")

    if model is None or not isinstance(result_df, pd.DataFrame) or result_df.empty:
        return None

    feature_cols = _get_anomaly_feature_columns(result_df)

    if not feature_cols:
        return None

    anomalies_only = result_df[result_df["is_anomaly"] == 1]

    if anomalies_only.empty:
        return None

    top_subset = anomalies_only.sort_values(by="anomaly_score", ascending=False).head(top_n)
    X_subset = top_subset[feature_cols]

    try:
        explainer = shap.TreeExplainer(model)
        raw_shap_values = explainer.shap_values(X_subset)
    except Exception:
        return None

    shap_array = np.array(raw_shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[0]

    artifacts = None

    if artifacts_path and os.path.exists(artifacts_path):
        try:
            artifacts = joblib.load(artifacts_path)
        except Exception:
            artifacts = None

    rows = []

    for position, (row_index, row) in enumerate(top_subset.iterrows()):

        contributions = list(zip(feature_cols, shap_array[position]))
        # Les contributions les plus négatives poussent le plus vers
        # l'anomalie (score_samples plus faible = plus anormal).
        contributions.sort(key=lambda item: item[1])
        top_contributions = contributions[:max_features]

        parts = []

        for feature, _ in top_contributions:
            scaled_value = row[feature]
            real_value = _unscale_value(feature, scaled_value, artifacts) if artifacts else None
            direction = "au-dessus" if scaled_value > 0 else "en-dessous"

            if real_value is not None:
                parts.append(f"{feature} = {round(real_value, 2)} ({direction} de la moyenne)")
            else:
                parts.append(f"{feature} ({direction} de la moyenne)")

        rows.append({
            "index": row_index,
            "anomaly_score": row.get("anomaly_score"),
            "severity": row.get("severity"),
            "explanation": "Contributeurs principaux (SHAP) : " + " ; ".join(parts)
        })

    return pd.DataFrame(rows)


def compute_diffi_like_importance(
    model,
    X_row: pd.Series,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Approximation "maison", inspirée de DIFFI : pour un échantillon
    donné, additionne sur tous les arbres de l'Isolation Forest une
    contribution par feature, pondérée par l'inverse de la profondeur
    du split (split précoce = poids plus fort). Ne nécessite aucune
    dépendance externe.

    Limite connue : suppose que chaque arbre a été entraîné sur
    l'ensemble des features (max_features=1.0, réglage par défaut de
    IsolationForest et celui utilisé par anomaly_detection.py). Avec
    un sous-échantillonnage de features, les indices de tree_.feature
    ne correspondraient plus directement à feature_names.
    """

    X_array = X_row.values.reshape(1, -1)
    contributions = {name: 0.0 for name in feature_names}

    for estimator in model.estimators_:

        tree_ = estimator.tree_

        try:
            node_indicator = estimator.decision_path(X_array)
            node_index = node_indicator.indices[
                node_indicator.indptr[0]: node_indicator.indptr[1]
            ]
        except Exception:
            continue

        for depth, node_id in enumerate(node_index):

            feature_idx = tree_.feature[node_id]

            if feature_idx < 0 or feature_idx >= len(feature_names):
                continue

            feature_name = feature_names[feature_idx]
            contributions[feature_name] += 1.0 / (depth + 1)

    return contributions


def generate_diffi_like_explanations(
    anomaly_result: Optional[Dict[str, Any]],
    artifacts_path: Optional[str] = None,
    top_n: int = 15,
    max_features: int = 3
) -> Optional[pd.DataFrame]:
    """
    Génère des explications locales sans dépendance externe (repli
    utilisé quand `shap` n'est pas installé).
    """

    if anomaly_result is None:
        return None

    model = anomaly_result.get("model")
    result_df = anomaly_result.get("result_df")

    if model is None or not isinstance(result_df, pd.DataFrame) or result_df.empty:
        return None

    feature_cols = _get_anomaly_feature_columns(result_df)

    if not feature_cols:
        return None

    anomalies_only = result_df[result_df["is_anomaly"] == 1]

    if anomalies_only.empty:
        return None

    top_subset = anomalies_only.sort_values(by="anomaly_score", ascending=False).head(top_n)

    artifacts = None

    if artifacts_path and os.path.exists(artifacts_path):
        try:
            artifacts = joblib.load(artifacts_path)
        except Exception:
            artifacts = None

    rows = []

    for row_index, row in top_subset.iterrows():

        try:
            contributions = compute_diffi_like_importance(model, row[feature_cols], feature_cols)
        except Exception:
            continue

        sorted_contribs = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
        top_contribs = [item for item in sorted_contribs if item[1] > 0][:max_features]

        if not top_contribs:
            continue

        parts = []

        for feature, _ in top_contribs:
            scaled_value = row[feature]
            real_value = _unscale_value(feature, scaled_value, artifacts) if artifacts else None
            parts.append(f"{feature} = {round(real_value, 2)}" if real_value is not None else feature)

        rows.append({
            "index": row_index,
            "anomaly_score": row.get("anomaly_score"),
            "severity": row.get("severity"),
            "explanation": "Contributeurs principaux (méthode maison, sans SHAP) : " + ", ".join(parts)
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def generate_anomaly_local_explanations(
    anomaly_result: Optional[Dict[str, Any]],
    artifacts_path: Optional[str] = None,
    top_n: int = 15,
    max_features: int = 3
) -> Dict[str, Any]:
    """
    Point d'entrée unique pour les explications locales des anomalies :
    tente SHAP en premier, retombe automatiquement sur la méthode
    maison (DIFFI-like) si SHAP est indisponible.
    """

    explanations = explain_anomalies_with_shap(
        anomaly_result, artifacts_path=artifacts_path, top_n=top_n, max_features=max_features
    )

    if explanations is not None:
        return {"explanations": explanations, "method": "shap"}

    explanations = generate_diffi_like_explanations(
        anomaly_result, artifacts_path=artifacts_path, top_n=top_n, max_features=max_features
    )

    if explanations is not None:
        return {"explanations": explanations, "method": "diffi_maison"}

    return {"explanations": None, "method": None}


# ============================================================
# 6quater. EXPLICATION CONTREFACTUELLE (clusters x anomalies)
# ============================================================
#
# Combine deux résultats déjà calculés séparément (clustering et
# anomalies) pour produire un contrefactuel actionnable : pour une
# anomalie, quel écart avec le centroïde de son propre segment
# explique le plus son statut d'anomalie, et quelle valeur "typique"
# lui permettrait de ne plus être signalée.
# ============================================================

def generate_anomaly_counterfactuals(
    clustering_result: Optional[Dict[str, Any]],
    clustered_dataset: Optional[pd.DataFrame],
    anomaly_dataset: Optional[pd.DataFrame],
    artifacts_path: Optional[str] = None,
    top_n: int = 15,
    max_features: int = 2
) -> Optional[pd.DataFrame]:
    """
    Génère, pour les anomalies les plus fortes, un contrefactuel basé
    sur l'écart avec le centroïde de leur propre segment.
    """

    if (
        clustering_result is None
        or clustered_dataset is None
        or anomaly_dataset is None
        or "cluster" not in clustered_dataset.columns
        or "is_anomaly" not in anomaly_dataset.columns
        or len(clustered_dataset) != len(anomaly_dataset)
    ):
        return None

    cluster_centers = clustering_result.get("kmeans", {}).get("cluster_centers")

    if cluster_centers is None:
        return None

    feature_cols = [
        col for col in clustered_dataset.columns
        if col != "cluster" and pd.api.types.is_numeric_dtype(clustered_dataset[col])
    ]

    if not feature_cols or cluster_centers.shape[1] != len(feature_cols):
        # Défense contre un désalignement de colonnes plutôt que de
        # produire un contrefactuel silencieusement faux.
        return None

    centers_df = pd.DataFrame(cluster_centers, columns=feature_cols)

    anomalies_only = anomaly_dataset[anomaly_dataset["is_anomaly"] == 1]

    if anomalies_only.empty:
        return None

    anomalies_only = anomalies_only.sort_values(by="anomaly_score", ascending=False).head(top_n)

    artifacts = None

    if artifacts_path and os.path.exists(artifacts_path):
        try:
            artifacts = joblib.load(artifacts_path)
        except Exception:
            artifacts = None

    rows = []

    for idx in anomalies_only.index:

        if idx not in clustered_dataset.index:
            continue

        own_cluster = clustered_dataset.loc[idx, "cluster"]

        if own_cluster not in centers_df.index:
            continue

        feature_vector = clustered_dataset.loc[idx, feature_cols]
        centroid = centers_df.loc[own_cluster]

        deltas = (feature_vector - centroid).abs().sort_values(ascending=False)
        top_features = deltas.head(max_features).index.tolist()

        parts = []

        for feature in top_features:
            current_scaled = feature_vector[feature]
            target_scaled = centroid[feature]

            current_real = _unscale_value(feature, current_scaled, artifacts) if artifacts else None
            target_real = _unscale_value(feature, target_scaled, artifacts) if artifacts else None

            if current_real is not None and target_real is not None:
                parts.append(
                    f"{feature} passait de {round(current_real, 2)} à ~{round(target_real, 2)} "
                    f"(valeur typique du segment {int(own_cluster)})"
                )
            else:
                parts.append(
                    f"{feature} se rapprocherait de la valeur typique du segment {int(own_cluster)}"
                )

        counterfactual_text = (
            "Si " + " et ".join(parts) + ", cette observation ne serait "
            "probablement plus signalée comme anomalie."
        )

        rows.append({
            "index": idx,
            "cluster": int(own_cluster),
            "anomaly_score": anomalies_only.loc[idx, "anomaly_score"],
            "counterfactual_text": counterfactual_text
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


# ============================================================
# 6quinquies. META-EXPLICABILITE DU PIPELINE
# ============================================================
#
# Explique les décisions AUTOMATIQUES du pipeline lui-même (choix du
# scaler, choix du K final), pas seulement les résultats du modèle.
# ============================================================

def explain_scaler_choice(artifacts_path: Optional[str]) -> Optional[str]:
    """
    Reconstitue, à partir des artefacts de preprocessing, une
    explication textuelle du choix automatique de scaler
    (preprocessing.choose_scaler).
    """

    if not artifacts_path or not os.path.exists(artifacts_path):
        return None

    try:
        artifacts = joblib.load(artifacts_path)
    except Exception:
        return None

    scaler_name = artifacts.get("scaler_name")
    iqr_bounds = artifacts.get("iqr_bounds", {})

    if not scaler_name:
        return None

    if not iqr_bounds:
        return f"Scaler retenu automatiquement : {scaler_name} (aucune information sur les outliers disponible)."

    flagged_or_winsorized = sum(
        1 for info in iqr_bounds.values()
        if info.get("action") in ("Winsorized", "Flagged only - delegated to Isolation Forest")
    )
    total_cols = len(iqr_bounds)
    ratio_pct = round((flagged_or_winsorized / total_cols) * 100, 1) if total_cols else 0

    if scaler_name == "RobustScaler":
        return (
            f"RobustScaler retenu automatiquement : {ratio_pct}% des colonnes numériques "
            f"({flagged_or_winsorized}/{total_cols}) présentaient des outliers significatifs. "
            f"RobustScaler s'appuie sur la médiane et l'IQR, moins sensibles aux valeurs "
            f"extrêmes que la moyenne et l'écart-type utilisés par StandardScaler."
        )

    return (
        f"StandardScaler retenu automatiquement : seulement {ratio_pct}% des colonnes "
        f"numériques ({flagged_or_winsorized}/{total_cols}) présentaient des outliers "
        f"significatifs, ce qui ne justifiait pas un scaler robuste aux valeurs extrêmes."
    )


def explain_k_choice(clustering_result: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Explique pourquoi le K final a été retenu, en s'appuyant sur la
    recherche par score silhouette déjà effectuée par
    clustering.find_best_k_by_silhouette, par rapport à l'hypothèse
    initiale suggérée par HDBSCAN.
    """

    if clustering_result is None:
        return None

    k_suggested = clustering_result.get("k_hdbscan_suggestion")
    k_final = clustering_result.get("n_clusters")
    best_k_search = clustering_result.get("best_k_search")

    if k_final is None:
        return None

    if not best_k_search or not best_k_search.get("scores"):
        return f"K={k_final} retenu directement (pas de recherche silhouette disponible pour comparer d'autres valeurs)."

    scores = best_k_search["scores"]
    best_score = best_k_search.get("best_score")

    if k_suggested is not None and k_suggested != k_final:

        alt_score = scores.get(k_suggested)
        text = (
            f"K={k_final} retenu au lieu de K={k_suggested} suggéré par HDBSCAN : "
            f"le score silhouette était meilleur pour K={k_final} ({best_score}"
        )

        if alt_score is not None:
            text += f" contre {alt_score} pour K={k_suggested})."
        else:
            text += ")."

        return text

    return (
        f"K={k_final} confirmé par la recherche silhouette (score = {best_score}), "
        f"cohérent avec l'hypothèse de structure suggérée par HDBSCAN."
    )


def generate_pipeline_meta_explanation(
    clustering_result: Optional[Dict[str, Any]],
    artifacts_path: Optional[str]
) -> Dict[str, Optional[str]]:
    """
    Assemble les explications des décisions internes du pipeline
    (scaler, K), pour montrer que le pipeline n'est pas une boîte
    noire même dans ses choix automatiques.
    """

    return {
        "scaler_explanation": explain_scaler_choice(artifacts_path),
        "k_choice_explanation": explain_k_choice(clustering_result)
    }


# ============================================================
# 6sexies. STABILITE DES CLUSTERS (ARI multi-graines)
# ============================================================

def compute_cluster_stability(
    clustered_dataset: Optional[pd.DataFrame],
    n_clusters: Optional[int],
    algorithm_used: Optional[str] = "KMeans",
    n_runs: int = 5,
    sample_size: int = 5000,
    random_state: int = 42
) -> Optional[Dict[str, Any]]:
    """
    Mesure la stabilité de la segmentation : relance le clustering
    plusieurs fois avec des graines différentes (sur un échantillon si
    le dataset est grand) et calcule l'Adjusted Rand Index (ARI) moyen
    entre chaque nouvelle partition et la partition de référence.

    Un ARI proche de 1 indique une segmentation stable, donc une
    explication (narratives, règles surrogate) plus digne de
    confiance. Un ARI faible indique une forte sensibilité à
    l'initialisation : les règles doivent être prises avec prudence.
    """

    if (
        clustered_dataset is None
        or "cluster" not in clustered_dataset.columns
        or not n_clusters
        or n_clusters < 2
    ):
        return None

    X = clustered_dataset.drop(columns=["cluster"]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X.empty:
        return None

    if len(X) > sample_size:
        sample_idx = X.sample(n=sample_size, random_state=random_state).index
        X_sample = X.loc[sample_idx]
        reference_sample = clustered_dataset.loc[sample_idx, "cluster"].values
    else:
        X_sample = X
        reference_sample = clustered_dataset["cluster"].values

    ModelClass = MiniBatchKMeans if algorithm_used == "MiniBatchKMeans" else KMeans

    ari_scores = []

    for run in range(n_runs):
        try:
            model = ModelClass(
                n_clusters=n_clusters,
                random_state=random_state + run + 1,
                n_init=5
            )
            labels_run = model.fit_predict(X_sample)
            ari_scores.append(adjusted_rand_score(reference_sample, labels_run))
        except Exception:
            continue

    if not ari_scores:
        return None

    mean_ari = round(float(np.mean(ari_scores)), 4)

    if mean_ari >= 0.75:
        label, tone = "élevée", "success"
    elif mean_ari >= 0.5:
        label, tone = "modérée", "warning"
    else:
        label, tone = "faible", "danger"

    text = (
        f"Stabilité {label} de la segmentation (ARI moyen = {mean_ari} sur "
        f"{len(ari_scores)} relances avec des graines différentes)."
    )

    if tone in ("warning", "danger"):
        text += (
            " Une stabilité faible signifie que la partition change "
            "significativement selon l'initialisation : les narratives et "
            "règles par segment doivent être interprétées avec prudence."
        )

    return {
        "mean_ari": mean_ari,
        "scores": [round(s, 4) for s in ari_scores],
        "label": label,
        "tone": tone,
        "text": text
    }


# ============================================================
# 7. POINT D'ENTREE GLOBAL
# ============================================================

def generate_full_interpretation(
    ml_result: Dict[str, Any],
    clustered_dataset: Optional[pd.DataFrame] = None,
    anomaly_dataset: Optional[pd.DataFrame] = None,
    artifacts_path: Optional[str] = None,
    surrogate_max_depth: int = 4,
    compute_stability: bool = True,
    stability_n_runs: int = 5,
    stability_sample_size: int = 5000,
    anomaly_explanation_top_n: int = 15
) -> Dict[str, Any]:
    """
    Assemble l'ensemble des interprétations pour un run ML donné.
    Point d'entrée unique utilisé par le dashboard analytique.
    """

    clustering_result = ml_result.get("clustering")
    anomaly_result = ml_result.get("anomaly_detection")

    result: Dict[str, Any] = {
        "cluster_quality": None,
        "noise_interpretation": None,
        "cluster_narratives": [],
        "cluster_profiles_enriched": None,
        "cluster_surrogate": None,
        "surrogate_fidelity": None,
        "cluster_rules": {},
        "cluster_stability": None,
        "pipeline_meta_explanation": None,
        "anomaly_narrative": None,
        "anomaly_local_explanations": None,
        "anomaly_local_explanations_method": None,
        "anomaly_counterfactuals": None,
        "cross_analysis": None,
        "recommendations": []
    }

    if clustering_result is not None:
        result["cluster_quality"] = interpret_cluster_quality(clustering_result.get("metrics"))
        result["noise_interpretation"] = interpret_noise_ratio(clustering_result.get("noise_ratio"))

        cluster_profiles = clustering_result.get("cluster_profiles")

        if isinstance(cluster_profiles, pd.DataFrame) and not cluster_profiles.empty:
            result["cluster_narratives"] = generate_cluster_narratives(cluster_profiles)
            result["cluster_profiles_enriched"] = enrich_cluster_profiles_with_real_values(
                cluster_profiles,
                artifacts_path
            )

        # ----------------------------------------------------
        # Explicabilité par modèle surrogate (règles IF/THEN)
        # ----------------------------------------------------

        surrogate_result = train_cluster_surrogate(
            clustered_dataset,
            max_depth=surrogate_max_depth
        ) if clustered_dataset is not None else None

        if surrogate_result is not None:

            result["cluster_surrogate"] = surrogate_result
            result["surrogate_fidelity"] = interpret_surrogate_fidelity(
                surrogate_result["fidelity_train"],
                surrogate_result["cv_fidelity"]
            )
            result["cluster_rules"] = generate_cluster_rules(
                surrogate_result,
                cluster_profiles_df=cluster_profiles,
                artifacts_path=artifacts_path
            )

            # Enrichit les narratives déjà générées avec la règle
            # correspondante quand elle existe, pour affichage unifié.
            for narrative in result["cluster_narratives"]:
                rule = result["cluster_rules"].get(narrative["cluster"])
                if rule:
                    narrative["rule_text"] = rule["full_text"]

        # ----------------------------------------------------
        # Stabilité de la segmentation (ARI multi-graines)
        # ----------------------------------------------------

        if compute_stability:
            result["cluster_stability"] = compute_cluster_stability(
                clustered_dataset,
                n_clusters=clustering_result.get("n_clusters"),
                algorithm_used=clustering_result.get("algorithm_used"),
                n_runs=stability_n_runs,
                sample_size=stability_sample_size
            )

        # ----------------------------------------------------
        # Méta-explicabilité du pipeline (choix du scaler et du K)
        # ----------------------------------------------------

        result["pipeline_meta_explanation"] = generate_pipeline_meta_explanation(
            clustering_result,
            artifacts_path
        )

    if anomaly_result is not None:
        result["anomaly_narrative"] = generate_anomaly_narrative(anomaly_result.get("summary"))

        local_explanations = generate_anomaly_local_explanations(
            anomaly_result,
            artifacts_path=artifacts_path,
            top_n=anomaly_explanation_top_n
        )
        result["anomaly_local_explanations"] = local_explanations["explanations"]
        result["anomaly_local_explanations_method"] = local_explanations["method"]

        result["anomaly_counterfactuals"] = generate_anomaly_counterfactuals(
            clustering_result,
            clustered_dataset,
            anomaly_dataset,
            artifacts_path=artifacts_path,
            top_n=anomaly_explanation_top_n
        )

    cross_analysis = cross_analyze_clusters_anomalies(clustered_dataset, anomaly_dataset)
    result["cross_analysis"] = cross_analysis

    result["recommendations"] = generate_recommendations(
        clustering_result=clustering_result,
        anomaly_result=anomaly_result,
        cross_analysis=cross_analysis,
        surrogate_fidelity=result["surrogate_fidelity"],
        cluster_stability=result["cluster_stability"]
    )

    return result