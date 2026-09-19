# ============================================================
# chart_interpretation.py
# Interprétation automatique des visualisations analytiques
# ============================================================

from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd


# ============================================================
# 1. FORMAT STANDARD D'UNE INTERPRETATION
# ============================================================

def build_chart_interpretation(
    observation: str,
    interpretation: str,
    recommendation: Optional[str] = None,
    limitation: Optional[str] = None,
    tone: str = "neutral",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Construit une interprétation standardisée pour un graphique.

    Args:
        observation:
            Information factuelle directement calculée.
        interpretation:
            Conclusion analytique fondée sur une règle.
        recommendation:
            Action ou vérification proposée.
        limitation:
            Limite méthodologique de l'interprétation.
        tone:
            neutral, success, warning ou danger.
        metadata:
            Informations supplémentaires.

    Returns:
        dict: Interprétation standardisée.
    """

    return {
        "observation": observation,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "limitation": limitation,
        "tone": tone,
        "metadata": metadata or {}
    }


def empty_chart_interpretation(
    message: str = "Interprétation non disponible."
) -> Dict[str, Any]:
    """
    Retourne une interprétation vide standardisée.
    """

    return build_chart_interpretation(
        observation=message,
        interpretation=(
            "Les données disponibles ne permettent pas de produire "
            "une interprétation suffisamment fiable."
        ),
        recommendation="Vérifier la disponibilité et la qualité des données.",
        limitation=None,
        tone="neutral"
    )


# ============================================================
# 2. DISTRIBUTION DES CLUSTERS
# ============================================================

def interpret_cluster_distribution(
    clustered_dataset: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """
    Interprète le bar chart et le pie chart de distribution des clusters.
    """

    if (
        clustered_dataset is None
        or clustered_dataset.empty
        or "cluster" not in clustered_dataset.columns
    ):
        return empty_chart_interpretation(
            "Distribution des clusters non disponible."
        )

    counts = clustered_dataset["cluster"].value_counts().sort_index()

    if counts.empty:
        return empty_chart_interpretation(
            "Aucun cluster n'a été détecté."
        )

    percentages = counts / counts.sum() * 100

    largest_cluster = percentages.idxmax()
    smallest_cluster = percentages.idxmin()

    largest_percentage = float(percentages.max())
    smallest_percentage = float(percentages.min())
    imbalance_gap = largest_percentage - smallest_percentage

    observation = (
        f"Le cluster {largest_cluster} est le plus représenté avec "
        f"{largest_percentage:.2f}% des observations. "
        f"Le cluster {smallest_cluster} est le moins représenté avec "
        f"{smallest_percentage:.2f}%."
    )

    if largest_percentage >= 80:
        tone = "danger"
        interpretation = (
            "La segmentation est fortement déséquilibrée. Un seul cluster "
            "regroupe la grande majorité des observations, ce qui peut indiquer "
            "une structure peu discriminante ou un nombre de clusters inadapté."
        )
        recommendation = (
            "Réexaminer le nombre de clusters, les features utilisées et les "
            "paramètres HDBSCAN ou K-Means."
        )

    elif largest_percentage >= 60:
        tone = "warning"
        interpretation = (
            "La segmentation présente un cluster dominant. Les petits clusters "
            "peuvent correspondre à des profils spécialisés, mais leur stabilité "
            "et leur pertinence métier doivent être vérifiées."
        )
        recommendation = (
            "Analyser les profils, les règles du surrogate et la stabilité "
            "des petits clusters."
        )

    else:
        tone = "success"
        interpretation = (
            "La distribution des observations entre les clusters est "
            "relativement équilibrée. Aucun groupe ne domine excessivement "
            "la segmentation."
        )
        recommendation = (
            "Comparer les profils et les features dominantes afin d'attribuer "
            "une signification métier à chaque segment."
        )

    return build_chart_interpretation(
        observation=observation,
        interpretation=interpretation,
        recommendation=recommendation,
        limitation=(
            "Une distribution équilibrée ne garantit pas à elle seule que "
            "les clusters sont bien séparés ou pertinents."
        ),
        tone=tone,
        metadata={
            "largest_cluster": int(largest_cluster),
            "largest_percentage": round(largest_percentage, 2),
            "smallest_cluster": int(smallest_cluster),
            "smallest_percentage": round(smallest_percentage, 2),
            "imbalance_gap": round(imbalance_gap, 2)
        }
    )


# ============================================================
# 3. PCA DES CLUSTERS
# ============================================================

def interpret_cluster_pca(
    clustering_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Interprète la projection PCA en utilisant les métriques du clustering.
    """

    if not clustering_result:
        return empty_chart_interpretation(
            "Projection PCA clustering non disponible."
        )

    metrics = clustering_result.get("metrics", {})

    silhouette = metrics.get("silhouette_score")
    davies_bouldin = metrics.get("davies_bouldin_score")

    if silhouette is None:
        return build_chart_interpretation(
            observation=(
                "La projection PCA permet d'observer la position relative "
                "des observations dans un espace à deux dimensions."
            ),
            interpretation=(
                "Aucune métrique de séparation n'est disponible pour confirmer "
                "la qualité des groupes visibles."
            ),
            recommendation=(
                "Comparer la PCA avec les profils de clusters et la heatmap."
            ),
            limitation=(
                "La PCA est une projection en deux dimensions et peut masquer "
                "des séparations présentes dans l'espace complet."
            ),
            tone="neutral"
        )

    if silhouette >= 0.50:
        tone = "success"
        interpretation = (
            "La segmentation présente une séparation globalement nette. "
            "Les observations appartenant au même cluster sont relativement "
            "cohérentes et les groupes sont suffisamment distincts."
        )

    elif silhouette >= 0.25:
        tone = "neutral"
        interpretation = (
            "La séparation entre les clusters est modérée. Certains groupes "
            "peuvent se chevaucher tout en conservant des profils distincts."
        )

    elif silhouette >= 0.10:
        tone = "warning"
        interpretation = (
            "La séparation entre les clusters est faible. Un chevauchement "
            "important peut être visible dans la projection."
        )

    else:
        tone = "danger"
        interpretation = (
            "Les clusters sont très peu séparés. La structure détectée peut "
            "être instable ou insuffisamment discriminante."
        )

    observation = f"Le score silhouette obtenu est de {silhouette:.4f}."

    if davies_bouldin is not None:
        observation += (
            f" Le score Davies-Bouldin est de {davies_bouldin:.4f}, "
            "où une valeur plus faible est préférable."
        )

    return build_chart_interpretation(
        observation=observation,
        interpretation=interpretation,
        recommendation=(
            "Croiser cette visualisation avec les profils des clusters, "
            "la heatmap, le surrogate et le score de stabilité."
        ),
        limitation=(
            "La PCA ne conserve qu'une partie de l'information initiale. "
            "Elle ne doit pas être utilisée seule pour valider les clusters."
        ),
        tone=tone,
        metadata={
            "silhouette_score": silhouette,
            "davies_bouldin_score": davies_bouldin
        }
    )


# ============================================================
# 4. HEATMAP DES CLUSTERS
# ============================================================

def interpret_cluster_heatmap(
    cluster_means: Optional[pd.DataFrame],
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Identifie les features qui différencient le plus les clusters.
    """

    if cluster_means is None or cluster_means.empty:
        return empty_chart_interpretation(
            "Heatmap des profils moyens non disponible."
        )

    feature_ranges = (
        cluster_means.max(axis=0)
        - cluster_means.min(axis=0)
    ).sort_values(ascending=False)

    top_features = feature_ranges.head(top_n).index.tolist()

    if not top_features:
        return empty_chart_interpretation(
            "Aucune feature distinctive n'a été identifiée."
        )

    strongest_feature = top_features[0]

    highest_cluster = cluster_means[strongest_feature].idxmax()
    lowest_cluster = cluster_means[strongest_feature].idxmin()

    observation = (
        "Les features qui varient le plus entre les clusters sont : "
        + ", ".join(f"`{feature}`" for feature in top_features)
        + "."
    )

    interpretation = (
        f"La feature `{strongest_feature}` est particulièrement distinctive. "
        f"Sa moyenne est la plus élevée dans le cluster {highest_cluster} "
        f"et la plus faible dans le cluster {lowest_cluster}."
    )

    return build_chart_interpretation(
        observation=observation,
        interpretation=interpretation,
        recommendation=(
            "Utiliser ces variables pour nommer les segments et définir "
            "des actions métier différenciées."
        ),
        limitation=(
            "Les valeurs correspondent au dataset préparé et peuvent être "
            "standardisées. Une valeur positive ne correspond pas toujours "
            "directement à une unité métier."
        ),
        tone="success",
        metadata={
            "top_features": top_features,
            "strongest_feature": strongest_feature,
            "highest_cluster": int(highest_cluster),
            "lowest_cluster": int(lowest_cluster)
        }
    )


# ============================================================
# 5. BOXPLOT D'UNE FEATURE PAR CLUSTER
# ============================================================

def interpret_feature_by_cluster(
    clustered_dataset: Optional[pd.DataFrame],
    feature: str
) -> Dict[str, Any]:
    """
    Interprète un box plot comparant une feature entre les clusters.
    """

    if (
        clustered_dataset is None
        or clustered_dataset.empty
        or "cluster" not in clustered_dataset.columns
        or feature not in clustered_dataset.columns
    ):
        return empty_chart_interpretation(
            f"Distribution de la feature {feature} non disponible."
        )

    grouped = clustered_dataset.groupby("cluster")[feature].agg(
        median="median",
        mean="mean",
        std="std",
        minimum="min",
        maximum="max",
        count="count"
    )

    if grouped.empty:
        return empty_chart_interpretation(
            f"Aucune statistique disponible pour la feature {feature}."
        )

    highest_cluster = grouped["median"].idxmax()
    lowest_cluster = grouped["median"].idxmin()

    highest_median = float(grouped.loc[highest_cluster, "median"])
    lowest_median = float(grouped.loc[lowest_cluster, "median"])

    valid_std = grouped["std"].dropna()

    if not valid_std.empty:
        most_variable_cluster = valid_std.idxmax()
        highest_std = float(valid_std.max())
    else:
        most_variable_cluster = None
        highest_std = None

    observation = (
        f"Pour `{feature}`, le cluster {highest_cluster} possède la médiane "
        f"la plus élevée ({highest_median:.4f}), tandis que le cluster "
        f"{lowest_cluster} possède la médiane la plus faible "
        f"({lowest_median:.4f})."
    )

    if most_variable_cluster is not None:
        observation += (
            f" Le cluster {most_variable_cluster} présente la dispersion "
            f"interne la plus importante, avec un écart-type de "
            f"{highest_std:.4f}."
        )

    return build_chart_interpretation(
        observation=observation,
        interpretation=(
            f"La feature `{feature}` contribue à différencier certains "
            "segments. L'écart entre les médianes permet d'identifier les "
            "clusters ayant les profils les plus opposés sur cette variable."
        ),
        recommendation=(
            "Comparer cette feature avec les autres variables dominantes "
            "avant de définir une stratégie métier."
        ),
        limitation=(
            "Les valeurs peuvent être standardisées. La dispersion interne "
            "peut également être influencée par les outliers."
        ),
        tone="neutral",
        metadata={
            "feature": feature,
            "highest_cluster": int(highest_cluster),
            "lowest_cluster": int(lowest_cluster),
            "highest_median": highest_median,
            "lowest_median": lowest_median
        }
    )


# ============================================================
# 6. SCATTER PLOT ENTRE DEUX FEATURES
# ============================================================

def interpret_feature_scatter(
    clustered_dataset: Optional[pd.DataFrame],
    x_feature: str,
    y_feature: str
) -> Dict[str, Any]:
    """
    Interprète la relation entre deux features numériques.
    """

    if (
        clustered_dataset is None
        or clustered_dataset.empty
        or x_feature not in clustered_dataset.columns
        or y_feature not in clustered_dataset.columns
    ):
        return empty_chart_interpretation(
            "Analyse croisée des features non disponible."
        )

    clean_data = clustered_dataset[
        [x_feature, y_feature]
    ].dropna()

    if len(clean_data) < 2:
        return empty_chart_interpretation(
            "Nombre de valeurs insuffisant pour calculer la corrélation."
        )

    correlation = float(
        clean_data[x_feature].corr(clean_data[y_feature])
    )

    absolute_correlation = abs(correlation)

    if absolute_correlation >= 0.80:
        strength = "très forte"
        tone = "warning"

    elif absolute_correlation >= 0.60:
        strength = "forte"
        tone = "neutral"

    elif absolute_correlation >= 0.30:
        strength = "modérée"
        tone = "neutral"

    else:
        strength = "faible"
        tone = "success"

    direction = "positive" if correlation >= 0 else "négative"

    return build_chart_interpretation(
        observation=(
            f"La corrélation linéaire entre `{x_feature}` et `{y_feature}` "
            f"est de {correlation:.4f}."
        ),
        interpretation=(
            f"La relation linéaire observée est {strength} et {direction}. "
            "La couleur des points permet également de vérifier si cette "
            "relation varie selon les clusters."
        ),
        recommendation=(
            "Si la corrélation est très forte, vérifier si les deux features "
            "apportent réellement des informations différentes au modèle."
        ),
        limitation=(
            "Une corrélation ne prouve pas une relation causale. Des relations "
            "non linéaires peuvent également exister."
        ),
        tone=tone,
        metadata={
            "x_feature": x_feature,
            "y_feature": y_feature,
            "correlation": round(correlation, 4)
        }
    )


# ============================================================
# 7. PROPORTION DES ANOMALIES
# ============================================================

def interpret_anomaly_ratio_chart(
    anomaly_summary: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Interprète le pie chart des observations normales et anormales.
    """

    if not anomaly_summary:
        return empty_chart_interpretation(
            "Résumé des anomalies non disponible."
        )

    total_rows = anomaly_summary.get("total_rows", 0) or 0
    anomaly_count = anomaly_summary.get("anomaly_count", 0) or 0
    anomaly_ratio = anomaly_summary.get("anomaly_ratio")

    if anomaly_ratio is None and total_rows > 0:
        anomaly_ratio = anomaly_count / total_rows * 100

    if anomaly_ratio is None:
        return empty_chart_interpretation(
            "Le taux d'anomalies ne peut pas être calculé."
        )

    observation = (
        f"{anomaly_count} anomalies ont été détectées sur "
        f"{total_rows} observations, soit {anomaly_ratio:.2f}%."
    )

    if anomaly_ratio >= 10:
        tone = "danger"
        interpretation = (
            "Le taux d'anomalies est très élevé. Cela peut correspondre à "
            "un changement important dans les données, à une mauvaise qualité "
            "des données ou à un paramétrage trop sensible."
        )
        recommendation = (
            "Examiner immédiatement les anomalies critiques, la contamination "
            "configurée et la qualité du preprocessing."
        )

    elif anomaly_ratio >= 5:
        tone = "warning"
        interpretation = (
            "Le taux d'anomalies est élevé et mérite une investigation."
        )
        recommendation = (
            "Analyser les explications locales SHAP ou DIFFI-like et identifier "
            "les segments qui concentrent le plus d'anomalies."
        )

    elif anomaly_ratio >= 1:
        tone = "neutral"
        interpretation = (
            "Le taux d'anomalies est modéré. Un nombre limité de comportements "
            "atypiques a été identifié."
        )
        recommendation = (
            "Prioriser les anomalies de sévérité élevée ou critique."
        )

    else:
        tone = "success"
        interpretation = (
            "Le taux d'anomalies est faible. La grande majorité des "
            "observations suit le comportement général du dataset."
        )
        recommendation = (
            "Maintenir la surveillance et comparer ce taux entre les runs."
        )

    return build_chart_interpretation(
        observation=observation,
        interpretation=interpretation,
        recommendation=recommendation,
        limitation=(
            "Le taux dépend du paramètre de contamination et ne correspond "
            "pas automatiquement à un taux réel de fraude ou d'erreur."
        ),
        tone=tone,
        metadata={
            "total_rows": total_rows,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": anomaly_ratio
        }
    )


# ============================================================
# 8. DISTRIBUTION DES SEVERITES
# ============================================================

def interpret_severity_distribution(
    severity_counts: Optional[Dict[str, int]]
) -> Dict[str, Any]:
    """
    Interprète la répartition des niveaux de sévérité.
    """

    if not severity_counts:
        return empty_chart_interpretation(
            "Répartition des sévérités non disponible."
        )

    total = sum(severity_counts.values())

    critical_count = int(severity_counts.get("critique", 0))
    high_count = int(severity_counts.get("élevé", 0))
    medium_count = int(severity_counts.get("moyen", 0))
    low_count = int(severity_counts.get("faible", 0))
    normal_count = int(severity_counts.get("normal", 0))

    priority_count = critical_count + high_count
    priority_ratio = priority_count / total * 100 if total else 0

    observation = (
        f"La distribution contient {critical_count} cas critiques, "
        f"{high_count} cas élevés, {medium_count} cas moyens, "
        f"{low_count} cas faibles et {normal_count} cas normaux."
    )

    if critical_count > 0:
        tone = "danger"
        interpretation = (
            f"{priority_ratio:.2f}% des observations appartiennent aux niveaux "
            "élevé ou critique. Ces cas nécessitent une analyse prioritaire."
        )
        recommendation = (
            "Examiner les explications locales et les contrefactuels des "
            "observations critiques."
        )

    elif high_count > 0:
        tone = "warning"
        interpretation = (
            "Des anomalies de niveau élevé sont présentes, même si aucun cas "
            "critique n'a été détecté."
        )
        recommendation = (
            "Analyser les principales features responsables de ces anomalies."
        )

    else:
        tone = "success"
        interpretation = (
            "Aucune anomalie critique ou élevée n'a été identifiée."
        )
        recommendation = (
            "Poursuivre la surveillance de l'évolution des scores."
        )

    return build_chart_interpretation(
        observation=observation,
        interpretation=interpretation,
        recommendation=recommendation,
        limitation=(
            "Les niveaux de sévérité reposent sur des seuils définis dans "
            "le projet et doivent être adaptés au contexte métier."
        ),
        tone=tone,
        metadata={
            "priority_count": priority_count,
            "priority_ratio": round(priority_ratio, 2)
        }
    )


# ============================================================
# 9. HISTOGRAMME DES SCORES D'ANOMALIE
# ============================================================

def interpret_anomaly_score_distribution(
    anomaly_result_df: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """
    Interprète l'histogramme des scores d'anomalie.
    """

    if (
        anomaly_result_df is None
        or anomaly_result_df.empty
        or "anomaly_score" not in anomaly_result_df.columns
    ):
        return empty_chart_interpretation(
            "Distribution des scores d'anomalie non disponible."
        )

    scores = anomaly_result_df["anomaly_score"].dropna()

    if scores.empty:
        return empty_chart_interpretation(
            "Aucun score d'anomalie exploitable."
        )

    mean_score = float(scores.mean())
    median_score = float(scores.median())

    high_ratio = float((scores >= 0.85).mean() * 100)
    critical_ratio = float((scores >= 0.95).mean() * 100)

    observation = (
        f"Le score moyen est de {mean_score:.4f} et le score médian "
        f"de {median_score:.4f}. {high_ratio:.2f}% des observations "
        "ont un score supérieur ou égal à 0,85."
    )

    if critical_ratio >= 5:
        tone = "danger"
        interpretation = (
            f"{critical_ratio:.2f}% des observations possèdent un score "
            "supérieur ou égal à 0,95. La distribution contient une proportion "
            "notable de comportements extrêmement atypiques."
        )
        recommendation = (
            "Analyser immédiatement les observations critiques et leurs "
            "principales contributions SHAP ou DIFFI-like."
        )

    elif high_ratio >= 5:
        tone = "warning"
        interpretation = (
            "Une proportion significative des observations possède un score "
            "d'anomalie élevé."
        )
        recommendation = (
            "Comparer les anomalies élevées avec le profil typique de leur cluster."
        )

    else:
        tone = "success"
        interpretation = (
            "La majorité des observations possède un score d'anomalie faible "
            "ou modéré."
        )
        recommendation = (
            "Maintenir la surveillance et suivre l'évolution historique."
        )

    return build_chart_interpretation(
        observation=observation,
        interpretation=interpretation,
        recommendation=recommendation,
        limitation=(
            "Le score est normalisé relativement au dataset analysé. "
            "Il ne représente pas une probabilité de fraude."
        ),
        tone=tone,
        metadata={
            "mean_score": round(mean_score, 4),
            "median_score": round(median_score, 4),
            "high_ratio": round(high_ratio, 2),
            "critical_ratio": round(critical_ratio, 2)
        }
    )


# ============================================================
# 10. PCA DES ANOMALIES
# ============================================================

def interpret_anomaly_pca(
    anomaly_pca_df: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """
    Interprète la projection PCA des anomalies.
    """

    if anomaly_pca_df is None or anomaly_pca_df.empty:
        return empty_chart_interpretation(
            "Projection PCA des anomalies non disponible."
        )

    if "type" not in anomaly_pca_df.columns:
        return build_chart_interpretation(
            observation=(
                "La projection PCA présente les observations dans un espace 2D."
            ),
            interpretation=(
                "Les types normal et anomalie ne sont pas disponibles pour "
                "mesurer leur séparation."
            ),
            recommendation="Vérifier la génération de la colonne type.",
            limitation=(
                "La PCA ne conserve qu'une partie de l'information initiale."
            ),
            tone="neutral"
        )

    anomaly_mask = (
        anomaly_pca_df["type"]
        .astype(str)
        .str.lower()
        .isin(["anomalie", "anomaly", "1"])
    )

    normal_mask = ~anomaly_mask

    anomaly_count = int(anomaly_mask.sum())
    normal_count = int(normal_mask.sum())

    observation = (
        f"La projection contient {anomaly_count} anomalies et "
        f"{normal_count} observations normales."
    )

    if anomaly_count == 0:
        interpretation = (
            "Aucune anomalie n'est visible dans l'échantillon PCA."
        )
        tone = "neutral"

    elif normal_count == 0:
        interpretation = (
            "L'échantillon PCA ne contient que des anomalies et ne permet "
            "pas de comparaison avec les observations normales."
        )
        tone = "warning"

    else:
        anomaly_center = anomaly_pca_df.loc[
            anomaly_mask,
            ["PC1", "PC2"]
        ].mean()

        normal_center = anomaly_pca_df.loc[
            normal_mask,
            ["PC1", "PC2"]
        ].mean()

        center_distance = float(
            np.linalg.norm(
                anomaly_center.to_numpy()
                - normal_center.to_numpy()
            )
        )

        interpretation = (
            f"La distance entre le centre des anomalies et celui des "
            f"observations normales dans la projection vaut "
            f"{center_distance:.4f}."
        )

        if center_distance >= 2:
            interpretation += (
                " Les anomalies semblent globalement éloignées du comportement "
                "normal dans l'espace projeté."
            )
            tone = "success"
        else:
            interpretation += (
                " Les anomalies ne forment pas nécessairement un groupe "
                "nettement séparé dans la projection."
            )
            tone = "neutral"

    return build_chart_interpretation(
        observation=observation,
        interpretation=interpretation,
        recommendation=(
            "Utiliser les explications locales pour identifier les features "
            "responsables de chaque anomalie."
        ),
        limitation=(
            "La distance dans la PCA est uniquement descriptive. "
            "Isolation Forest travaille dans l'espace complet des features."
        ),
        tone=tone
    )


# ============================================================
# 11. ANALYSE CROISEE CLUSTERS ET ANOMALIES
# ============================================================

def interpret_cross_analysis(
    cross_analysis: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """
    Interprète le taux d'anomalies par cluster.
    """

    if cross_analysis is None or cross_analysis.empty:
        return empty_chart_interpretation(
            "Analyse croisée segments et anomalies non disponible."
        )

    required_columns = {
        "cluster",
        "anomaly_ratio_pct",
        "global_ratio_pct"
    }

    if not required_columns.issubset(cross_analysis.columns):
        return empty_chart_interpretation(
            "Colonnes nécessaires à l'analyse croisée manquantes."
        )

    highest_row = cross_analysis.loc[
        cross_analysis["anomaly_ratio_pct"].idxmax()
    ]

    cluster_id = highest_row["cluster"]
    cluster_ratio = float(highest_row["anomaly_ratio_pct"])
    global_ratio = float(highest_row["global_ratio_pct"])

    difference = cluster_ratio - global_ratio

    observation = (
        f"Le cluster {cluster_id} possède le taux d'anomalies le plus élevé "
        f"avec {cluster_ratio:.2f}%, contre {global_ratio:.2f}% "
        "pour l'ensemble du dataset."
    )

    if difference >= 5:
        tone = "danger"
        interpretation = (
            f"Ce segment dépasse la moyenne globale de {difference:.2f} "
            "points. Il concentre anormalement les observations atypiques."
        )
        recommendation = (
            f"Prioriser l'analyse du cluster {cluster_id} et examiner ses "
            "features dominantes, anomalies critiques et contrefactuels."
        )

    elif difference >= 2:
        tone = "warning"
        interpretation = (
            f"Ce segment dépasse la moyenne globale de {difference:.2f} "
            "points et mérite une surveillance particulière."
        )
        recommendation = (
            "Comparer les anomalies de ce segment à son profil typique."
        )

    else:
        tone = "success"
        interpretation = (
            "Aucun cluster ne semble concentrer les anomalies de manière "
            "fortement disproportionnée."
        )
        recommendation = (
            "Maintenir une surveillance uniforme des différents segments."
        )

    return build_chart_interpretation(
        observation=observation,
        interpretation=interpretation,
        recommendation=recommendation,
        limitation=(
            "Une concentration d'anomalies dans un segment ne signifie pas "
            "automatiquement que ce segment est problématique."
        ),
        tone=tone,
        metadata={
            "highest_risk_cluster": cluster_id,
            "cluster_ratio": cluster_ratio,
            "global_ratio": global_ratio,
            "difference": round(difference, 2)
        }
    )


# ============================================================
# 12. EVOLUTION HISTORIQUE D'UNE METRIQUE
# ============================================================

def interpret_metric_trend(
    history_df: Optional[pd.DataFrame],
    metric_column: str,
    metric_label: str,
    higher_is_better: bool = True,
    significant_change: float = 0.01
) -> Dict[str, Any]:
    """
    Analyse l'évolution d'une métrique entre les runs.
    """

    if (
        history_df is None
        or history_df.empty
        or metric_column not in history_df.columns
    ):
        return empty_chart_interpretation(
            f"Historique de {metric_label} non disponible."
        )

    valid_history = history_df.dropna(
        subset=[metric_column]
    ).copy()

    if "created_at" in valid_history.columns:
        valid_history["created_at"] = pd.to_datetime(
            valid_history["created_at"],
            errors="coerce"
        )

        valid_history = valid_history.sort_values("created_at")

    values = valid_history[metric_column]

    if len(values) < 2:
        return build_chart_interpretation(
            observation=(
                f"Une seule valeur de {metric_label} est disponible."
            ),
            interpretation=(
                "Au moins deux exécutions comparables sont nécessaires "
                "pour analyser une tendance."
            ),
            recommendation=(
                "Exécuter le pipeline sur plusieurs périodes ou versions."
            ),
            limitation=None,
            tone="neutral"
        )

    first_value = float(values.iloc[0])
    last_value = float(values.iloc[-1])
    variation = last_value - first_value

    if abs(variation) < significant_change:
        tone = "neutral"
        interpretation = (
            f"Le {metric_label} reste globalement stable."
        )

    else:
        improved = (
            variation > 0
            if higher_is_better
            else variation < 0
        )

        if improved:
            tone = "success"
            interpretation = (
                f"Le {metric_label} évolue dans une direction favorable."
            )
        else:
            tone = "warning"
            interpretation = (
                f"Le {metric_label} évolue dans une direction qui nécessite "
                "une investigation."
            )

    return build_chart_interpretation(
        observation=(
            f"Le {metric_label} passe de {first_value:.4f} à "
            f"{last_value:.4f}, soit une variation de {variation:+.4f}."
        ),
        interpretation=interpretation,
        recommendation=(
            "Comparer les datasets, les paramètres, les features et les "
            "algorithmes utilisés entre les runs."
        ),
        limitation=(
            "La comparaison n'est pertinente que si les datasets et les "
            "conditions d'exécution sont comparables."
        ),
        tone=tone,
        metadata={
            "first_value": first_value,
            "last_value": last_value,
            "variation": variation
        }
    )


# ============================================================
# 13. GENERATION GLOBALE DES INTERPRETATIONS DE GRAPHIQUES
# ============================================================

def generate_chart_interpretations(
    clustered_dataset: Optional[pd.DataFrame] = None,
    clustering_result: Optional[Dict[str, Any]] = None,
    anomaly_result: Optional[Dict[str, Any]] = None,
    anomaly_pca_df: Optional[pd.DataFrame] = None,
    cross_analysis: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Génère les principales interprétations statiques du dashboard.
    """

    anomaly_summary = (
        anomaly_result.get("summary", {})
        if anomaly_result
        else None
    )

    anomaly_result_df = (
        anomaly_result.get("result_df")
        if anomaly_result
        else None
    )

    severity_counts = (
        anomaly_summary.get("severity_counts")
        if anomaly_summary
        else None
    )

    return {
        "cluster_distribution": interpret_cluster_distribution(
            clustered_dataset
        ),
        "cluster_pca": interpret_cluster_pca(
            clustering_result
        ),
        "anomaly_ratio": interpret_anomaly_ratio_chart(
            anomaly_summary
        ),
        "severity_distribution": interpret_severity_distribution(
            severity_counts
        ),
        "anomaly_score_distribution": (
            interpret_anomaly_score_distribution(
                anomaly_result_df
            )
        ),
        "anomaly_pca": interpret_anomaly_pca(
            anomaly_pca_df
        ),
        "cross_analysis": interpret_cross_analysis(
            cross_analysis
        )
    }