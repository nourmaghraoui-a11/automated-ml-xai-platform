# ============================================================
# 1_Dashboard_Analytique.py
# Dashboard : clustering, anomalies, XAI et interprétations
# ============================================================

import os
import sys
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time
from concurrent.futures import ThreadPoolExecutor


# ============================================================
# AJOUT DU CHEMIN RACINE DU PROJET
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(APP_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from src.ml_controller import (
    run_ml_pipeline,
    attach_cluster_labels,
    attach_anomaly_results,
    generate_ml_summary
)

from src.interpretation import generate_full_interpretation

from src.chart_interpretation import (
    generate_chart_interpretations,
    interpret_cluster_heatmap,
    interpret_feature_by_cluster,
    interpret_feature_scatter,
    interpret_metric_trend
)

from src.warehouse import (
    get_ml_runs,
    get_cluster_history,
    get_anomaly_history
)

from src.ui_theme import (
    apply_theme,
    hero,
    stage_header,
    pill,
    card
)

from src.session_store import save_full_session
from src.live_sync import (
    sync_analytical_page,
    set_active_run_id,
    get_active_run_id
)
from src.scheduler_ui import render_live_sync_controls, render_scheduler_panel


# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="Dashboard Analytique",
    page_icon="📊",
    layout="wide"
)

apply_theme()

hero(
    eyebrow="Espace analytique",
    title="Dashboard analytique : segmentation et anomalies",
    subtitle=(
        "Analyse Machine Learning, explicabilité, interprétations "
        "automatiques, recommandations et suivi historique."
    )
)

TOTAL_STEPS = 11


# ============================================================
# OUTILS D'AFFICHAGE
# ============================================================

def apply_plot_theme(figure):
    """
    Applique le thème du projet aux graphiques Plotly.
    """

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#EDEEF0",
        title_font_color="#EDEEF0",
        legend_title_font_color="#EDEEF0",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return figure


def sample_for_visualization(
    dataframe,
    sample_size=10000,
    random_state=42
):
    """
    Échantillonne un DataFrame pour garder les graphiques rapides.
    """

    if dataframe is None or dataframe.empty:
        return dataframe

    if len(dataframe) <= sample_size:
        return dataframe.copy()

    return dataframe.sample(
        n=sample_size,
        random_state=random_state
    )


def get_numeric_features(
    dataframe,
    excluded_columns=None
):
    """
    Retourne les features numériques disponibles.
    """

    if dataframe is None or dataframe.empty:
        return []

    excluded_columns = excluded_columns or []

    return [
        column
        for column in dataframe.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def safe_metric_value(value, default="-"):
    """
    Retourne une valeur adaptée à st.metric.
    """

    if value is None:
        return default

    return value

def format_elapsed_time(seconds):
    """
    Convertit une durée en secondes au format HH:MM:SS.
    """

    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_with_live_timer(
    target_function,
    timer_placeholder,
    status_placeholder=None,
    refresh_interval=0.5,
    **function_kwargs
):
    """
    Exécute une fonction dans un thread séparé et met à jour
    un chronomètre Streamlit pendant son exécution.

    Returns:
        tuple:
            - résultat de la fonction
            - durée réelle en secondes
    """

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as executor:

        future = executor.submit(
            target_function,
            **function_kwargs
        )

        while not future.done():

            elapsed_seconds = (
                time.perf_counter() - start_time
            )

            timer_placeholder.metric(
                label="Temps d'exécution en cours",
                value=format_elapsed_time(elapsed_seconds)
            )

            if status_placeholder is not None:
                status_placeholder.info(
                    "Le pipeline Machine Learning et XAI est en cours "
                    "d'exécution. Ne fermez pas cette page."
                )

            time.sleep(refresh_interval)

        result = future.result()

    duration_seconds = (
        time.perf_counter() - start_time
    )

    timer_placeholder.metric(
        label="Temps d'exécution final",
        value=format_elapsed_time(duration_seconds)
    )

    if status_placeholder is not None:
        status_placeholder.success(
            "Exécution terminée avec succès."
        )

    return result, round(duration_seconds, 3)

def display_chart_interpretation(
    title,
    insight,
    expanded=True
):
    """
    Affiche une interprétation automatique sous un graphique.
    """

    if not insight:
        return

    tone = insight.get("tone", "neutral")

    tone_icons = {
        "success": "✅",
        "warning": "⚠️",
        "danger": "🚨",
        "neutral": "ℹ️"
    }

    icon = tone_icons.get(tone, "ℹ️")

    with st.expander(
        f"{icon} Interprétation automatique : {title}",
        expanded=expanded
    ):
        observation = insight.get("observation")
        interpretation_text = insight.get("interpretation")
        recommendation = insight.get("recommendation")
        limitation = insight.get("limitation")

        if observation:
            st.markdown("**Ce que montre le graphique**")
            st.write(observation)

        if interpretation_text:
            st.markdown("**Interprétation**")
            st.write(interpretation_text)

        if recommendation:
            st.markdown("**Recommandation**")
            st.write(recommendation)

        if limitation:
            st.markdown("**Limite de l'interprétation**")
            st.caption(limitation)


# ============================================================
# INFORMATIONS DU MODULE
# ============================================================

with st.expander("À propos de ce module"):
    st.markdown("""
    Le module analytique intègre :

    - segmentation avec HDBSCAN et K-Means ou MiniBatchKMeans ;
    - détection d'anomalies avec Isolation Forest ;
    - interprétation globale et locale ;
    - surrogate model pour expliquer les clusters ;
    - SHAP avec méthode DIFFI-like en repli ;
    - explications contrefactuelles ;
    - interprétation automatique de chaque visualisation ;
    - analyse croisée entre clusters et anomalies ;
    - suivi historique des performances ;
    - export des résultats.
    """)


# ============================================================
# SESSION STATE
# ============================================================

session_defaults = {
    "analytical_ml_result": None,
    "clustered_dataset": None,
    "anomaly_dataset": None,
    "interpretation": None
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ------------------------------------------------------------
# Synchronisation avec le planificateur (processus indépendant).
#
# Même logique que sur le Dashboard Technique : le planificateur ne
# peut pas écrire directement dans le st.session_state de cette page,
# donc c'est cette page qui vérifie elle-même, à chaque exécution du
# script, si un run plus récent (X_ready, ml_result, datasets enrichis,
# XAI) existe sur disque — et se recharge alors automatiquement.
#
# Un seul mécanisme couvre à la fois le premier chargement (session
# vide -> restaure le dernier run disponible) et les mises à jour en
# direct pendant que la page reste ouverte (via l'auto-rafraîchissement
# configuré plus bas, qui ré-exécute ce script périodiquement).
# ------------------------------------------------------------

if get_active_run_id() is None and st.session_state.analytical_ml_result is not None:
    # Résultat déjà en mémoire (rerun Streamlit normal) mais run_id pas
    # encore suivi : on le déduit sans recharger depuis le disque.
    set_active_run_id(st.session_state.analytical_ml_result.get("run_id"))

_sync_result = sync_analytical_page()

with st.expander("🗓️ Planificateur & synchronisation en direct", expanded=bool(_sync_result)):
    render_live_sync_controls(
        _sync_result,
        label="la segmentation, les anomalies et les explications XAI",
        key_prefix="analytics_live_sync"
    )
    st.divider()
    render_scheduler_panel()


# ============================================================
# 1. CHARGEMENT DE X_READY
# ============================================================

stage_header(
    1,
    TOTAL_STEPS,
    "Chargement du dataset préparé"
)

st.info("""
Exécutez d'abord la préparation des données dans le Dashboard Technique.
Le dataset X_ready sera ensuite utilisé pour la segmentation,
la détection d'anomalies et l'explicabilité.
""")

X_ready = None
artifacts_path = None

pipeline_result = st.session_state.get("pipeline_result")

if pipeline_result is not None:

    X_ready = pipeline_result.get("X_ready")

    artifacts_path = (
        pipeline_result
        .get("preprocessing_info", {})
        .get("artifacts_saved_at")
    )

    if isinstance(X_ready, pd.DataFrame) and not X_ready.empty:

        st.markdown(
            pill(
                "Dataset préparé trouvé dans la session",
                tone="success"
            ),
            unsafe_allow_html=True
        )

        st.write("")

        numeric_feature_count = (
            X_ready
            .select_dtypes(include="number")
            .shape[1]
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Nombre de lignes",
                X_ready.shape[0]
            )

        with col2:
            st.metric(
                "Nombre de features",
                X_ready.shape[1]
            )

        with col3:
            st.metric(
                "Features numériques",
                numeric_feature_count
            )

        st.subheader("Aperçu de X_ready")

        st.dataframe(
            X_ready.head(20),
            width="stretch"
        )

        if X_ready.shape[0] < 10:
            st.warning(
                "Le dataset contient moins de 10 lignes. "
                "Les résultats ML peuvent être peu fiables."
            )

        if numeric_feature_count == 0:
            st.error(
                "Aucune feature numérique n'est disponible dans X_ready."
            )

    else:
        X_ready = None

        st.warning(
            "Le résultat du pipeline ne contient pas de dataset exploitable."
        )

else:
    st.warning(
        "Aucun dataset préparé n'est présent dans la session."
    )


# ============================================================
# 2. PARAMETRES MACHINE LEARNING
# ============================================================

stage_header(
    2,
    TOTAL_STEPS,
    "Paramètres Machine Learning"
)

(
    tab_clustering_parameters,
    tab_anomaly_parameters,
    tab_xai_parameters
) = st.tabs([
    "Segmentation",
    "Détection d'anomalies",
    "Explicabilité"
])


# ============================================================
# PARAMETRES CLUSTERING
# ============================================================

with tab_clustering_parameters:

    enable_clustering = st.checkbox(
        "Activer la segmentation",
        value=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        default_k = st.number_input(
            "Nombre de clusters par défaut",
            min_value=2,
            max_value=20,
            value=3
        )

    with col2:
        max_k = st.number_input(
            "Nombre maximal de clusters",
            min_value=2,
            max_value=30,
            value=10
        )

    with col3:
        min_cluster_size = st.number_input(
            "Taille minimale HDBSCAN",
            min_value=2,
            max_value=10000,
            value=100
        )

    if default_k > max_k:
        st.error(
            "Le nombre de clusters par défaut doit être inférieur "
            "ou égal au nombre maximal de clusters."
        )

    col4, col5 = st.columns(2)

    with col4:
        use_hdbscan = st.checkbox(
            "Utiliser HDBSCAN",
            value=True
        )

    with col5:
        use_silhouette_fallback = st.checkbox(
            "Utiliser Silhouette comme fallback",
            value=True
        )

    st.subheader("Optimisation grands volumes")

    col6, col7, col8 = st.columns(3)

    with col6:
        large_dataset_threshold = st.number_input(
            "Seuil dataset volumineux",
            min_value=10000,
            max_value=1000000,
            value=100000,
            step=10000
        )

    with col7:
        hdbscan_sample_size = st.number_input(
            "Échantillon HDBSCAN",
            min_value=1000,
            max_value=200000,
            value=30000,
            step=1000
        )

    with col8:
        pca_sample_size = st.number_input(
            "Échantillon PCA clustering",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000
        )

    col9, col10 = st.columns(2)

    with col9:
        metrics_sample_size = st.number_input(
            "Échantillon métriques",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000
        )

    with col10:
        batch_size = st.number_input(
            "Batch size MiniBatchKMeans",
            min_value=512,
            max_value=20000,
            value=4096,
            step=512
        )


# ============================================================
# PARAMETRES ANOMALIES
# ============================================================

with tab_anomaly_parameters:

    enable_anomaly_detection = st.checkbox(
        "Activer la détection d'anomalies",
        value=True
    )

    contamination_mode = st.radio(
        "Mode contamination",
        options=["auto", "personnalisé"],
        horizontal=True
    )

    if contamination_mode == "auto":
        contamination = "auto"
    else:
        contamination = st.number_input(
            "Taux attendu d'anomalies",
            min_value=0.001,
            max_value=0.5,
            value=0.03,
            step=0.001,
            format="%.3f"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        n_estimators = st.number_input(
            "Nombre d'arbres",
            min_value=50,
            max_value=1000,
            value=200,
            step=50
        )

    with col2:
        max_samples_mode = st.radio(
            "Max samples",
            options=["auto", "personnalisé"]
        )

    with col3:
        anomaly_pca_sample_size = st.number_input(
            "Échantillon PCA anomalies",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000
        )

    if max_samples_mode == "auto":
        max_samples = "auto"
    else:
        max_samples = st.number_input(
            "Nombre maximal d'échantillons",
            min_value=1000,
            max_value=200000,
            value=10000,
            step=1000
        )


# ============================================================
# PARAMETRES XAI
# ============================================================

with tab_xai_parameters:

    st.markdown("""
    Les paramètres suivants contrôlent la complexité des explications.
    """)

    surrogate_max_depth = st.slider(
        "Profondeur maximale du surrogate",
        min_value=2,
        max_value=10,
        value=4
    )

    compute_stability = st.checkbox(
        "Calculer la stabilité des clusters",
        value=True
    )

    stability_n_runs = st.slider(
        "Nombre de relances pour la stabilité",
        min_value=2,
        max_value=10,
        value=5
    )

    stability_sample_size = st.number_input(
        "Échantillon stabilité",
        min_value=1000,
        max_value=50000,
        value=5000,
        step=1000
    )

    anomaly_explanation_top_n = st.slider(
        "Nombre d'anomalies à expliquer",
        min_value=5,
        max_value=50,
        value=15
    )


# ============================================================
# 3. EXECUTION DU MODULE ML
# ============================================================

stage_header(
    3,
    TOTAL_STEPS,
    "Exécution du module Machine Learning"
)

has_numeric_features = (
    isinstance(X_ready, pd.DataFrame)
    and not X_ready.empty
    and X_ready.select_dtypes(include="number").shape[1] > 0
)

run_disabled = (
    X_ready is None
    or not has_numeric_features
    or default_k > max_k
    or (
        not enable_clustering
        and not enable_anomaly_detection
    )
)

if X_ready is not None:

    if not enable_clustering and not enable_anomaly_detection:
        st.warning(
            "Activez au moins un module Machine Learning."
        )

    if st.button(
        "Lancer l'analyse Machine Learning",
        type="primary",
        disabled=run_disabled
    ):

        try:
            with st.spinner(
                "Analyse Machine Learning et XAI en cours..."
            ):

                ml_result = run_ml_pipeline(
                    X_ready=X_ready,

                    enable_clustering=enable_clustering,
                    use_hdbscan=use_hdbscan,
                    use_silhouette_fallback=use_silhouette_fallback,
                    default_k=default_k,
                    min_cluster_size=min_cluster_size,
                    max_k=max_k,
                    large_dataset_threshold=large_dataset_threshold,
                    hdbscan_sample_size=hdbscan_sample_size,
                    pca_sample_size=pca_sample_size,
                    metrics_sample_size=metrics_sample_size,
                    batch_size=batch_size,

                    enable_anomaly_detection=enable_anomaly_detection,
                    contamination=contamination,
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    anomaly_pca_sample_size=anomaly_pca_sample_size
                )

                if (
                    enable_clustering
                    and ml_result.get("cluster_labels") is not None
                ):
                    clustered_dataset = attach_cluster_labels(
                        X_ready=X_ready,
                        labels=ml_result["cluster_labels"]
                    )
                else:
                    clustered_dataset = None

                if (
                    enable_anomaly_detection
                    and ml_result.get("anomaly_labels") is not None
                ):
                    anomaly_dataset = attach_anomaly_results(
                        X_ready=X_ready,
                        anomaly_labels=ml_result["anomaly_labels"],
                        anomaly_scores=ml_result["anomaly_scores"]
                    )
                else:
                    anomaly_dataset = None

                interpretation = generate_full_interpretation(
                    ml_result=ml_result,
                    clustered_dataset=clustered_dataset,
                    anomaly_dataset=anomaly_dataset,
                    artifacts_path=artifacts_path,
                    surrogate_max_depth=surrogate_max_depth,
                    compute_stability=compute_stability,
                    stability_n_runs=stability_n_runs,
                    stability_sample_size=stability_sample_size,
                    anomaly_explanation_top_n=(
                        anomaly_explanation_top_n
                    )
                )

                st.session_state.analytical_ml_result = ml_result
                st.session_state.clustered_dataset = clustered_dataset
                st.session_state.anomaly_dataset = anomaly_dataset
                st.session_state.interpretation = interpretation

                # Persistance sur disque : survit à un redémarrage de
                # Streamlit, du terminal, ou à une nouvelle session.
                # On utilise le run_id du module ML pour garder un
                # instantané cohérent (préparation + ML + XAI ensemble).
                save_full_session(
                    run_id=ml_result["run_id"],
                    prep_result=pipeline_result,
                    ml_result=ml_result,
                    clustered_dataset=clustered_dataset,
                    anomaly_dataset=anomaly_dataset,
                    interpretation=interpretation
                )
                # Marque ce run comme "déjà affiché" par CETTE session :
                # évite qu'un tick de synchronisation en direct le
                # recharge inutilement juste après l'avoir calculé.
                set_active_run_id(ml_result["run_id"])

            st.success(
                "Analyse Machine Learning et XAI terminée avec succès."
            )

            st.rerun()

        except Exception as error:
            st.error(
                f"Erreur lors de l'analyse : {error}"
            )

    if run_disabled:
        st.caption(
            "Le bouton est désactivé. Vérifiez le dataset, "
            "les paramètres et l'activation des modules."
        )

else:
    st.warning(
        "Préparez les données avant de lancer l'analyse."
    )


# ============================================================
# 4. RECUPERATION DES RESULTATS
# ============================================================

ml_result = st.session_state.get("analytical_ml_result")
clustered_dataset = st.session_state.get("clustered_dataset")
anomaly_dataset = st.session_state.get("anomaly_dataset")
interpretation = st.session_state.get("interpretation")

filtered_anomalies = None


if ml_result is not None:

    st.divider()

    summary = generate_ml_summary(ml_result)
    metadata = ml_result.get("metadata", {})

    clustering_result = ml_result.get("clustering")
    anomaly_result = ml_result.get("anomaly_detection")

    cross_analysis_result = (
        interpretation.get("cross_analysis")
        if interpretation
        else None
    )

    chart_interpretations = generate_chart_interpretations(
        clustered_dataset=clustered_dataset,
        clustering_result=clustering_result,
        anomaly_result=anomaly_result,
        anomaly_pca_df=ml_result.get("anomaly_pca_2d"),
        cross_analysis=cross_analysis_result
    )


    # ========================================================
    # 4. RESUME GLOBAL
    # ========================================================

    stage_header(
        4,
        TOTAL_STEPS,
        "Résumé global"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Run ID",
            safe_metric_value(summary.get("run_id"))
        )

    with col2:
        st.metric(
            "Statut",
            safe_metric_value(summary.get("status"))
        )

    with col3:
        st.metric(
            "Durée en secondes",
            safe_metric_value(
                summary.get("duration_seconds")
            )
        )

    with col4:
        st.metric(
            "Taille input",
            str(summary.get("input_shape"))
        )

    with st.expander("Résumé technique"):
        st.json(summary)


    # ========================================================
    # 5. INTERPRETATION AUTOMATIQUE GLOBALE
    # ========================================================

    stage_header(
        5,
        TOTAL_STEPS,
        "Interprétation automatique",
        "Lecture métier et recommandations globales"
    )

    if interpretation:

        col1, col2 = st.columns(2)

        cluster_quality = interpretation.get("cluster_quality")
        anomaly_narrative = interpretation.get("anomaly_narrative")

        with col1:
            st.markdown("##### Qualité de la segmentation")

            if cluster_quality:

                st.markdown(
                    pill(
                        cluster_quality["label"].capitalize(),
                        tone=cluster_quality["tone"]
                    ),
                    unsafe_allow_html=True
                )

                st.write("")
                st.write(cluster_quality["text"])

            else:
                st.info("Segmentation non disponible.")

        with col2:
            st.markdown("##### Résumé des anomalies")

            if anomaly_narrative:

                ratio_info = anomaly_narrative[
                    "ratio_interpretation"
                ]

                st.markdown(
                    pill(
                        ratio_info["label"].capitalize(),
                        tone=ratio_info["tone"]
                    ),
                    unsafe_allow_html=True
                )

                st.write("")
                st.write(ratio_info["text"])
                st.write(anomaly_narrative["priority_text"])

            else:
                st.info(
                    "Détection d'anomalies non disponible."
                )

        st.markdown("##### Recommandations")

        recommendations = interpretation.get(
            "recommendations",
            []
        )

        if recommendations:

            for recommendation in recommendations:

                st.markdown(
                    (
                        f"{pill('●', tone=recommendation['tone'])} "
                        f"{recommendation['text']}"
                    ),
                    unsafe_allow_html=True
                )

        else:
            st.info(
                "Aucune recommandation disponible."
            )

    else:
        st.info(
            "Interprétation globale non disponible."
        )


    # ========================================================
    # ONGLETS ANALYTIQUES
    # ========================================================

    (
        tab_overview,
        tab_clusters,
        tab_anomalies,
        tab_features,
        tab_xai,
        tab_cross
    ) = st.tabs([
        "Vue globale",
        "Segmentation",
        "Anomalies",
        "Analyse des features",
        "Explicabilité XAI",
        "Segments et anomalies"
    ])


    # ========================================================
    # VUE GLOBALE
    # ========================================================

    with tab_overview:

        st.subheader("Vue synthétique")

        col1, col2 = st.columns(2)

        with col1:

            if (
                clustered_dataset is not None
                and "cluster" in clustered_dataset.columns
            ):

                cluster_distribution = (
                    clustered_dataset["cluster"]
                    .value_counts()
                    .reset_index()
                )

                cluster_distribution.columns = [
                    "cluster",
                    "count"
                ]

                cluster_distribution["cluster"] = (
                    "Cluster "
                    + cluster_distribution[
                        "cluster"
                    ].astype(str)
                )

                fig_cluster_overview = px.pie(
                    cluster_distribution,
                    names="cluster",
                    values="count",
                    title="Répartition des clusters",
                    hole=0.45
                )

                fig_cluster_overview.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )

                apply_plot_theme(fig_cluster_overview)

                st.plotly_chart(
                    fig_cluster_overview,
                    width="stretch"
                )

                display_chart_interpretation(
                    title="Répartition des clusters",
                    insight=chart_interpretations.get(
                        "cluster_distribution"
                    )
                )

            else:
                st.info(
                    "Segmentation non disponible."
                )

        with col2:

            if anomaly_result is not None:

                anomaly_summary = anomaly_result.get(
                    "summary",
                    {}
                )

                total_rows = (
                    anomaly_summary.get("total_rows", 0)
                    or 0
                )

                anomaly_count = (
                    anomaly_summary.get("anomaly_count", 0)
                    or 0
                )

                normal_count = max(
                    total_rows - anomaly_count,
                    0
                )

                anomaly_distribution = pd.DataFrame({
                    "Type": ["Normales", "Anomalies"],
                    "Nombre": [
                        normal_count,
                        anomaly_count
                    ]
                })

                fig_anomaly_overview = px.pie(
                    anomaly_distribution,
                    names="Type",
                    values="Nombre",
                    title="Proportion des anomalies",
                    hole=0.45,
                    color="Type",
                    color_discrete_map={
                        "Normales": "#4FA57C",
                        "Anomalies": "#C25B5B"
                    }
                )

                fig_anomaly_overview.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )

                apply_plot_theme(fig_anomaly_overview)

                st.plotly_chart(
                    fig_anomaly_overview,
                    width="stretch"
                )

                display_chart_interpretation(
                    title="Proportion des anomalies",
                    insight=chart_interpretations.get(
                        "anomaly_ratio"
                    )
                )

            else:
                st.info(
                    "Détection d'anomalies non disponible."
                )


    # ========================================================
    # SEGMENTATION
    # ========================================================

    with tab_clusters:

        if clustering_result is not None:

            stage_header(
                6,
                TOTAL_STEPS,
                "Résultats de segmentation"
            )

            metrics = clustering_result.get("metrics", {})

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Nombre de clusters",
                    safe_metric_value(
                        clustering_result.get("n_clusters")
                    )
                )

            with col2:
                st.metric(
                    "Silhouette",
                    safe_metric_value(
                        metrics.get("silhouette_score")
                    )
                )

            with col3:
                st.metric(
                    "Davies-Bouldin",
                    safe_metric_value(
                        metrics.get(
                            "davies_bouldin_score"
                        )
                    )
                )

            with col4:
                st.metric(
                    "Algorithme",
                    safe_metric_value(
                        clustering_result.get(
                            "algorithm_used"
                        )
                    )
                )

            st.subheader("Distribution des clusters")

            if clustered_dataset is not None:

                cluster_counts = (
                    clustered_dataset["cluster"]
                    .value_counts()
                    .reset_index()
                )

                cluster_counts.columns = [
                    "cluster",
                    "count"
                ]

                cluster_counts["cluster"] = (
                    cluster_counts["cluster"].astype(str)
                )

                col_bar, col_pie = st.columns(2)

                with col_bar:

                    fig_cluster_bar = px.bar(
                        cluster_counts,
                        x="cluster",
                        y="count",
                        text="count",
                        title="Effectif par cluster"
                    )

                    fig_cluster_bar.update_traces(
                        marker_color="#F2A93C"
                    )

                    apply_plot_theme(fig_cluster_bar)

                    st.plotly_chart(
                        fig_cluster_bar,
                        width="stretch"
                    )

                with col_pie:

                    cluster_pie = cluster_counts.copy()

                    cluster_pie["cluster"] = (
                        "Cluster "
                        + cluster_pie["cluster"]
                    )

                    fig_cluster_pie = px.pie(
                        cluster_pie,
                        names="cluster",
                        values="count",
                        title="Proportion par cluster",
                        hole=0.4
                    )

                    fig_cluster_pie.update_traces(
                        textposition="inside",
                        textinfo="percent+label"
                    )

                    apply_plot_theme(fig_cluster_pie)

                    st.plotly_chart(
                        fig_cluster_pie,
                        width="stretch"
                    )

                display_chart_interpretation(
                    title="Distribution des clusters",
                    insight=chart_interpretations.get(
                        "cluster_distribution"
                    )
                )

            st.subheader("Projection PCA des clusters")

            cluster_pca = ml_result.get("cluster_pca_2d")

            if (
                isinstance(cluster_pca, pd.DataFrame)
                and not cluster_pca.empty
            ):

                pca_display = cluster_pca.copy()

                pca_display["cluster"] = (
                    pca_display["cluster"].astype(str)
                )

                fig_cluster_pca = px.scatter(
                    pca_display,
                    x="PC1",
                    y="PC2",
                    color="cluster",
                    title="Projection PCA des clusters",
                    opacity=0.7
                )

                apply_plot_theme(fig_cluster_pca)

                st.plotly_chart(
                    fig_cluster_pca,
                    width="stretch"
                )

                display_chart_interpretation(
                    title="Projection PCA des clusters",
                    insight=chart_interpretations.get(
                        "cluster_pca"
                    )
                )

            else:
                st.info("PCA clustering non disponible.")

            st.subheader("Profils des clusters")

            cluster_profiles = ml_result.get(
                "cluster_profiles"
            )

            if (
                isinstance(cluster_profiles, pd.DataFrame)
                and not cluster_profiles.empty
            ):

                enriched = (
                    interpretation.get(
                        "cluster_profiles_enriched"
                    )
                    if interpretation
                    else None
                )

                if (
                    isinstance(enriched, pd.DataFrame)
                    and not enriched.empty
                ):
                    st.dataframe(
                        enriched,
                        width="stretch"
                    )
                else:
                    st.dataframe(
                        cluster_profiles,
                        width="stretch"
                    )

                narratives = (
                    interpretation.get(
                        "cluster_narratives",
                        []
                    )
                    if interpretation
                    else []
                )

                for narrative in narratives:

                    card(
                        (
                            f"{narrative['title']} : "
                            f"{narrative['percentage']}%"
                        ),
                        narrative["text"]
                    )

                    st.write("")

            else:
                st.info(
                    "Profils des clusters non disponibles."
                )

        else:
            st.info("Segmentation désactivée.")


    # ========================================================
    # ANOMALIES
    # ========================================================

    with tab_anomalies:

        if anomaly_result is not None:

            stage_header(
                7,
                TOTAL_STEPS,
                "Détection d'anomalies"
            )

            anomaly_summary = anomaly_result.get(
                "summary",
                {}
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Anomalies",
                    safe_metric_value(
                        anomaly_summary.get(
                            "anomaly_count"
                        )
                    )
                )

            with col2:
                st.metric(
                    "Ratio (%)",
                    safe_metric_value(
                        anomaly_summary.get(
                            "anomaly_ratio"
                        )
                    )
                )

            with col3:
                st.metric(
                    "Score moyen",
                    safe_metric_value(
                        anomaly_summary.get(
                            "avg_anomaly_score"
                        )
                    )
                )

            with col4:
                st.metric(
                    "Score maximal",
                    safe_metric_value(
                        anomaly_summary.get(
                            "max_anomaly_score"
                        )
                    )
                )

            severity_counts = anomaly_summary.get(
                "severity_counts",
                {}
            )

            st.subheader("Répartition des sévérités")

            if severity_counts:

                severity_df = pd.DataFrame({
                    "severity": list(
                        severity_counts.keys()
                    ),
                    "count": list(
                        severity_counts.values()
                    )
                })

                fig_severity = px.bar(
                    severity_df,
                    x="severity",
                    y="count",
                    text="count",
                    color="severity",
                    title="Niveaux de sévérité",
                    color_discrete_map={
                        "normal": "#4FA57C",
                        "faible": "#F1C40F",
                        "moyen": "#E67E22",
                        "élevé": "#E74C3C",
                        "critique": "#8E44AD"
                    }
                )

                apply_plot_theme(fig_severity)

                st.plotly_chart(
                    fig_severity,
                    width="stretch"
                )

                display_chart_interpretation(
                    title="Répartition des sévérités",
                    insight=chart_interpretations.get(
                        "severity_distribution"
                    )
                )

            st.subheader(
                "Distribution des scores d'anomalie"
            )

            anomaly_result_df = ml_result.get(
                "anomaly_result_df"
            )

            if (
                isinstance(anomaly_result_df, pd.DataFrame)
                and not anomaly_result_df.empty
                and "anomaly_score"
                in anomaly_result_df.columns
            ):

                histogram_data = sample_for_visualization(
                    anomaly_result_df,
                    sample_size=50000
                )

                color_column = (
                    "severity"
                    if "severity" in histogram_data.columns
                    else None
                )

                fig_histogram = px.histogram(
                    histogram_data,
                    x="anomaly_score",
                    color=color_column,
                    nbins=50,
                    title="Distribution des scores"
                )

                apply_plot_theme(fig_histogram)

                st.plotly_chart(
                    fig_histogram,
                    width="stretch"
                )

                display_chart_interpretation(
                    title="Distribution des scores d'anomalie",
                    insight=chart_interpretations.get(
                        "anomaly_score_distribution"
                    )
                )

            st.subheader("PCA des anomalies")

            anomaly_pca = ml_result.get(
                "anomaly_pca_2d"
            )

            if (
                isinstance(anomaly_pca, pd.DataFrame)
                and not anomaly_pca.empty
            ):

                fig_anomaly_pca = px.scatter(
                    anomaly_pca,
                    x="PC1",
                    y="PC2",
                    color="type",
                    size="anomaly_score",
                    title="Projection PCA des anomalies",
                    opacity=0.7,
                    color_discrete_map={
                        "normal": "#4FA57C",
                        "anomalie": "#C25B5B"
                    }
                )

                apply_plot_theme(fig_anomaly_pca)

                st.plotly_chart(
                    fig_anomaly_pca,
                    width="stretch"
                )

                display_chart_interpretation(
                    title="Projection PCA des anomalies",
                    insight=chart_interpretations.get(
                        "anomaly_pca"
                    )
                )

            st.subheader("Top anomalies")

            top_anomalies = ml_result.get(
                "top_anomalies"
            )

            if (
                isinstance(top_anomalies, pd.DataFrame)
                and not top_anomalies.empty
            ):

                severity_options = (
                    sorted(
                        top_anomalies["severity"]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    if "severity" in top_anomalies.columns
                    else []
                )

                col_filter1, col_filter2 = st.columns(2)

                with col_filter1:

                    selected_severities = st.multiselect(
                        "Sévérités",
                        options=severity_options,
                        default=severity_options
                    )

                with col_filter2:

                    min_score = st.slider(
                        "Score minimum",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05
                    )

                filtered_anomalies = top_anomalies.copy()

                if (
                    selected_severities
                    and "severity"
                    in filtered_anomalies.columns
                ):
                    filtered_anomalies = (
                        filtered_anomalies[
                            filtered_anomalies[
                                "severity"
                            ].isin(selected_severities)
                        ]
                    )

                if (
                    "anomaly_score"
                    in filtered_anomalies.columns
                ):
                    filtered_anomalies = (
                        filtered_anomalies[
                            filtered_anomalies[
                                "anomaly_score"
                            ] >= min_score
                        ]
                    )

                st.dataframe(
                    filtered_anomalies,
                    width="stretch"
                )

            else:
                st.info("Aucune anomalie disponible.")

        else:
            st.info(
                "Détection d'anomalies désactivée."
            )


    # ========================================================
    # ANALYSE DES FEATURES
    # ========================================================

    with tab_features:

        stage_header(
            8,
            TOTAL_STEPS,
            "Analyse des features"
        )

        if clustered_dataset is not None:

            numeric_features = get_numeric_features(
                clustered_dataset,
                excluded_columns=["cluster"]
            )

            if numeric_features:

                cluster_means = (
                    clustered_dataset
                    .groupby("cluster")[numeric_features]
                    .mean()
                )

                st.subheader("Heatmap des profils moyens")

                if len(numeric_features) > 20:

                    selected_heatmap_features = (
                        cluster_means
                        .var()
                        .sort_values(ascending=False)
                        .head(20)
                        .index
                        .tolist()
                    )

                    heatmap_data = cluster_means[
                        selected_heatmap_features
                    ]

                else:
                    heatmap_data = cluster_means

                fig_heatmap = px.imshow(
                    heatmap_data,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Moyennes normalisées par cluster"
                )

                apply_plot_theme(fig_heatmap)

                st.plotly_chart(
                    fig_heatmap,
                    width="stretch"
                )

                heatmap_insight = interpret_cluster_heatmap(
                    cluster_means,
                    top_n=5
                )

                display_chart_interpretation(
                    title="Heatmap des clusters",
                    insight=heatmap_insight
                )

                st.subheader(
                    "Box plot configurable"
                )

                selected_box_feature = st.selectbox(
                    "Feature à comparer",
                    options=numeric_features
                )

                box_data = clustered_dataset[
                    [
                        selected_box_feature,
                        "cluster"
                    ]
                ].copy()

                box_data = sample_for_visualization(
                    box_data,
                    sample_size=50000
                )

                box_data["cluster"] = (
                    "Cluster "
                    + box_data["cluster"].astype(str)
                )

                fig_boxplot = px.box(
                    box_data,
                    x="cluster",
                    y=selected_box_feature,
                    color="cluster",
                    points=False,
                    title=(
                        f"Distribution de "
                        f"{selected_box_feature}"
                    )
                )

                apply_plot_theme(fig_boxplot)

                st.plotly_chart(
                    fig_boxplot,
                    width="stretch"
                )

                box_insight = interpret_feature_by_cluster(
                    clustered_dataset,
                    selected_box_feature
                )

                display_chart_interpretation(
                    title=(
                        f"Distribution de "
                        f"{selected_box_feature}"
                    ),
                    insight=box_insight
                )

                if len(numeric_features) >= 2:

                    st.subheader(
                        "Scatter plot configurable"
                    )

                    col_x, col_y = st.columns(2)

                    with col_x:
                        x_feature = st.selectbox(
                            "Feature X",
                            options=numeric_features,
                            index=0,
                            key="scatter_x"
                        )

                    with col_y:
                        y_feature = st.selectbox(
                            "Feature Y",
                            options=numeric_features,
                            index=1,
                            key="scatter_y"
                        )

                    scatter_data = clustered_dataset[
                        [
                            x_feature,
                            y_feature,
                            "cluster"
                        ]
                    ].copy()

                    scatter_data = sample_for_visualization(
                        scatter_data,
                        sample_size=10000
                    )

                    scatter_data["cluster"] = (
                        scatter_data["cluster"].astype(str)
                    )

                    fig_scatter = px.scatter(
                        scatter_data,
                        x=x_feature,
                        y=y_feature,
                        color="cluster",
                        opacity=0.6,
                        title=(
                            f"{x_feature} en fonction "
                            f"de {y_feature}"
                        )
                    )

                    apply_plot_theme(fig_scatter)

                    st.plotly_chart(
                        fig_scatter,
                        width="stretch"
                    )

                    scatter_insight = (
                        interpret_feature_scatter(
                            clustered_dataset,
                            x_feature,
                            y_feature
                        )
                    )

                    display_chart_interpretation(
                        title=(
                            f"Relation entre {x_feature} "
                            f"et {y_feature}"
                        ),
                        insight=scatter_insight
                    )

                st.subheader("Radar des clusters")

                radar_features = (
                    cluster_means
                    .var()
                    .sort_values(ascending=False)
                    .head(min(6, len(numeric_features)))
                    .index
                    .tolist()
                )

                radar_figure = go.Figure()

                for cluster_id, row in cluster_means.iterrows():

                    values = row[radar_features].tolist()

                    radar_figure.add_trace(
                        go.Scatterpolar(
                            r=values + [values[0]],
                            theta=(
                                radar_features
                                + [radar_features[0]]
                            ),
                            fill="toself",
                            name=f"Cluster {cluster_id}"
                        )
                    )

                radar_figure.update_layout(
                    title="Comparaison des profils",
                    polar={
                        "radialaxis": {
                            "visible": True
                        }
                    }
                )

                apply_plot_theme(radar_figure)

                st.plotly_chart(
                    radar_figure,
                    width="stretch"
                )

                display_chart_interpretation(
                    title="Radar des clusters",
                    insight=heatmap_insight
                )

                st.subheader("Features dominantes")

                cluster_profiles = ml_result.get(
                    "cluster_profiles"
                )

                if (
                    isinstance(cluster_profiles, pd.DataFrame)
                    and "dominant_features"
                    in cluster_profiles.columns
                ):

                    dominant_features = []

                    for features in cluster_profiles[
                        "dominant_features"
                    ]:

                        if isinstance(features, list):
                            dominant_features.extend(features)

                        elif isinstance(features, str):

                            cleaned = (
                                features
                                .replace("[", "")
                                .replace("]", "")
                                .replace("'", "")
                                .replace('"', "")
                                .split(",")
                            )

                            dominant_features.extend([
                                value.strip()
                                for value in cleaned
                                if value.strip()
                            ])

                    feature_counts = Counter(
                        dominant_features
                    )

                    if feature_counts:

                        dominant_df = pd.DataFrame(
                            feature_counts.items(),
                            columns=[
                                "feature",
                                "frequency"
                            ]
                        ).sort_values(
                            "frequency",
                            ascending=False
                        )

                        fig_dominant = px.bar(
                            dominant_df,
                            x="frequency",
                            y="feature",
                            orientation="h",
                            text="frequency",
                            title=(
                                "Fréquence des features "
                                "dominantes"
                            )
                        )

                        fig_dominant.update_traces(
                            marker_color="#F2A93C"
                        )

                        fig_dominant.update_layout(
                            yaxis={
                                "categoryorder":
                                "total ascending"
                            }
                        )

                        apply_plot_theme(fig_dominant)

                        st.plotly_chart(
                            fig_dominant,
                            width="stretch"
                        )

                        display_chart_interpretation(
                            title="Features dominantes",
                            insight=heatmap_insight
                        )

            else:
                st.info(
                    "Aucune feature numérique disponible."
                )

        else:
            st.info(
                "Exécutez la segmentation pour afficher "
                "l'analyse des features."
            )


    # ========================================================
    # EXPLICABILITE XAI
    # ========================================================

    with tab_xai:

        stage_header(
            9,
            TOTAL_STEPS,
            "Explicabilité XAI"
        )

        if interpretation:

            surrogate_result = interpretation.get(
                "surrogate_result"
            )

            surrogate_fidelity = interpretation.get(
                "surrogate_fidelity"
            )

            cluster_rules = interpretation.get(
                "cluster_rules",
                {}
            )

            st.subheader(
                "Surrogate model des clusters"
            )

            if surrogate_fidelity:

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Fidélité entraînement",
                        safe_metric_value(
                            surrogate_fidelity.get(
                                "fidelity_train"
                            )
                        )
                    )

                with col2:
                    st.metric(
                        "Fidélité validation",
                        safe_metric_value(
                            surrogate_fidelity.get(
                                "cv_fidelity"
                            )
                        )
                    )

                st.write(
                    surrogate_fidelity.get("text")
                )

            if surrogate_result:

                importance_df = surrogate_result.get(
                    "feature_importance"
                )

                if (
                    isinstance(importance_df, pd.DataFrame)
                    and not importance_df.empty
                ):

                    fig_surrogate_importance = px.bar(
                        importance_df.head(15),
                        x="importance",
                        y="feature",
                        orientation="h",
                        title=(
                            "Importance des features "
                            "du surrogate"
                        )
                    )

                    fig_surrogate_importance.update_layout(
                        yaxis={
                            "categoryorder":
                            "total ascending"
                        }
                    )

                    apply_plot_theme(
                        fig_surrogate_importance
                    )

                    st.plotly_chart(
                        fig_surrogate_importance,
                        width="stretch"
                    )

                    st.caption(
                        "Cette importance explique le surrogate. "
                        "Sa fiabilité dépend de son score de fidélité."
                    )

            if cluster_rules:

                st.subheader(
                    "Règles explicatives des clusters"
                )

                for cluster_id, rule_info in (
                    cluster_rules.items()
                ):

                    rule_text = rule_info.get(
                        "rule_text",
                        rule_info.get(
                            "text",
                            str(rule_info)
                        )
                    )

                    card(
                        f"Règle du cluster {cluster_id}",
                        rule_text
                    )

                    st.write("")

            st.subheader(
                "Explications locales des anomalies"
            )

            local_explanations = interpretation.get(
                "anomaly_local_explanations"
            )

            if isinstance(local_explanations, dict):

                method = local_explanations.get(
                    "method",
                    "non précisée"
                )

                explanation_df = local_explanations.get(
                    "explanations"
                )

                st.info(
                    f"Méthode utilisée : {method}"
                )

                if (
                    isinstance(explanation_df, pd.DataFrame)
                    and not explanation_df.empty
                ):

                    st.dataframe(
                        explanation_df,
                        width="stretch"
                    )

                    if {
                        "feature",
                        "importance"
                    }.issubset(explanation_df.columns):

                        global_xai_importance = (
                            explanation_df
                            .groupby("feature")["importance"]
                            .mean()
                            .sort_values(ascending=False)
                            .reset_index()
                        )

                        fig_xai_importance = px.bar(
                            global_xai_importance.head(15),
                            x="importance",
                            y="feature",
                            orientation="h",
                            title=(
                                "Importance moyenne pour "
                                "les anomalies expliquées"
                            )
                        )

                        fig_xai_importance.update_layout(
                            yaxis={
                                "categoryorder":
                                "total ascending"
                            }
                        )

                        apply_plot_theme(
                            fig_xai_importance
                        )

                        st.plotly_chart(
                            fig_xai_importance,
                            width="stretch"
                        )

            else:
                st.info(
                    "Explications locales non disponibles."
                )

            st.subheader(
                "Explications contrefactuelles"
            )

            counterfactuals = interpretation.get(
                "counterfactuals"
            )

            if (
                isinstance(counterfactuals, pd.DataFrame)
                and not counterfactuals.empty
            ):

                st.dataframe(
                    counterfactuals,
                    width="stretch"
                )

                st.caption(
                    "Les contrefactuels décrivent les modifications "
                    "qui rapprochent une anomalie du profil typique "
                    "de son segment."
                )

            else:
                st.info(
                    "Contrefactuels non disponibles."
                )

            st.subheader(
                "Décisions internes du pipeline"
            )

            meta_explanation = interpretation.get(
                "pipeline_meta_explanation",
                {}
            )

            if meta_explanation:

                for decision_name, decision_text in (
                    meta_explanation.items()
                ):

                    if decision_text:

                        card(
                            decision_name.replace(
                                "_",
                                " "
                            ).capitalize(),
                            decision_text
                        )

                        st.write("")

            stability = interpretation.get(
                "cluster_stability"
            )

            if stability:

                st.subheader(
                    "Stabilité des clusters"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "ARI moyen",
                        safe_metric_value(
                            stability.get("mean_ari")
                        )
                    )

                with col2:
                    st.metric(
                        "Nombre de relances",
                        safe_metric_value(
                            stability.get("n_runs")
                        )
                    )

                st.write(
                    stability.get(
                        "text",
                        "Stabilité calculée avec plusieurs graines."
                    )
                )

        else:
            st.info(
                "Résultats XAI non disponibles."
            )


    # ========================================================
    # SEGMENTS ET ANOMALIES
    # ========================================================

    with tab_cross:

        stage_header(
            10,
            TOTAL_STEPS,
            "Analyse croisée segments et anomalies"
        )

        if (
            isinstance(cross_analysis_result, pd.DataFrame)
            and not cross_analysis_result.empty
        ):

            cross_display = cross_analysis_result.copy()

            cross_display["cluster"] = (
                cross_display["cluster"].astype(str)
            )

            fig_cross = px.bar(
                cross_display,
                x="cluster",
                y="anomaly_ratio_pct",
                text="anomaly_ratio_pct",
                color="a_surveiller",
                title="Taux d'anomalies par segment",
                color_discrete_map={
                    True: "#C25B5B",
                    False: "#4FA57C"
                }
            )

            if "global_ratio_pct" in cross_display.columns:

                global_ratio = float(
                    cross_display[
                        "global_ratio_pct"
                    ].iloc[0]
                )

                fig_cross.add_hline(
                    y=global_ratio,
                    line_dash="dash",
                    line_color="#9299A3",
                    annotation_text="Moyenne globale"
                )

            apply_plot_theme(fig_cross)

            st.plotly_chart(
                fig_cross,
                width="stretch"
            )

            display_chart_interpretation(
                title="Taux d'anomalies par segment",
                insight=chart_interpretations.get(
                    "cross_analysis"
                )
            )

            st.dataframe(
                cross_analysis_result,
                width="stretch"
            )

        else:
            st.info(
                "Analyse croisée non disponible."
            )


    # ========================================================
    # ARTEFACTS ET EXPORTS
    # ========================================================

    st.divider()

    stage_header(
        11,
        TOTAL_STEPS,
        "Artefacts et exports"
    )

    with st.expander("Artefacts et rapports"):

        st.json({
            "models_dir": metadata.get("models_dir"),
            "reports_dir": metadata.get("reports_dir"),
            "paths": ml_result.get("paths"),
            "artifacts": ml_result.get("artifacts"),
            "warehouse_status": ml_result.get(
                "warehouse_status"
            )
        })

    col1, col2 = st.columns(2)

    with col1:

        if clustered_dataset is not None:

            st.download_button(
                label="Télécharger le dataset avec clusters",
                data=(
                    clustered_dataset
                    .to_csv(index=False)
                    .encode("utf-8")
                ),
                file_name="dataset_with_clusters.csv",
                mime="text/csv"
            )

        cluster_profiles = ml_result.get(
            "cluster_profiles"
        )

        if (
            isinstance(cluster_profiles, pd.DataFrame)
            and not cluster_profiles.empty
        ):

            st.download_button(
                label="Télécharger les profils de clusters",
                data=(
                    cluster_profiles
                    .to_csv(index=False)
                    .encode("utf-8")
                ),
                file_name="cluster_profiles.csv",
                mime="text/csv"
            )

    with col2:

        if anomaly_dataset is not None:

            st.download_button(
                label="Télécharger le dataset avec anomalies",
                data=(
                    anomaly_dataset
                    .to_csv(index=False)
                    .encode("utf-8")
                ),
                file_name="dataset_with_anomalies.csv",
                mime="text/csv"
            )

        if (
            isinstance(filtered_anomalies, pd.DataFrame)
            and not filtered_anomalies.empty
        ):

            st.download_button(
                label="Télécharger les anomalies filtrées",
                data=(
                    filtered_anomalies
                    .to_csv(index=False)
                    .encode("utf-8")
                ),
                file_name="top_anomalies.csv",
                mime="text/csv"
            )

    if (
        isinstance(cross_analysis_result, pd.DataFrame)
        and not cross_analysis_result.empty
    ):

        st.download_button(
            label="Télécharger l'analyse croisée",
            data=(
                cross_analysis_result
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name="cross_analysis.csv",
            mime="text/csv"
        )

else:
    st.info(
        "Lancez l'analyse Machine Learning pour afficher "
        "les résultats."
    )


# ============================================================
# HISTORIQUE DES EXECUTIONS ML
# ============================================================

st.divider()

st.header("Historique des exécutions Machine Learning")

try:
    ml_runs_history = get_ml_runs()

    if not ml_runs_history.empty:

        st.dataframe(
            ml_runs_history,
            width="stretch"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Nombre de runs",
                ml_runs_history.shape[0]
            )

        with col2:
            st.metric(
                "Dernier statut",
                ml_runs_history.iloc[0]["status"]
            )

        with col3:
            st.metric(
                "Dernier run",
                ml_runs_history.iloc[0]["run_id"]
            )

        cluster_history = get_cluster_history()
        anomaly_history = get_anomaly_history()

        col_trend1, col_trend2 = st.columns(2)

        with col_trend1:

            if not cluster_history.empty:

                cluster_history = cluster_history.copy()

                cluster_history["created_at"] = pd.to_datetime(
                    cluster_history["created_at"],
                    errors="coerce"
                )

                cluster_history = (
                    cluster_history
                    .dropna(subset=["created_at"])
                    .sort_values("created_at")
                )

                fig_silhouette_trend = px.line(
                    cluster_history,
                    x="created_at",
                    y="silhouette_score",
                    markers=True,
                    color="algorithm_used",
                    title="Évolution du score silhouette"
                )

                apply_plot_theme(
                    fig_silhouette_trend
                )

                st.plotly_chart(
                    fig_silhouette_trend,
                    width="stretch"
                )

                silhouette_insight = interpret_metric_trend(
                    history_df=cluster_history,
                    metric_column="silhouette_score",
                    metric_label="score silhouette",
                    higher_is_better=True,
                    significant_change=0.01
                )

                display_chart_interpretation(
                    title="Évolution du score silhouette",
                    insight=silhouette_insight,
                    expanded=False
                )

            else:
                st.info(
                    "Historique clustering non disponible."
                )

        with col_trend2:

            if not anomaly_history.empty:

                anomaly_history = anomaly_history.copy()

                anomaly_history["created_at"] = pd.to_datetime(
                    anomaly_history["created_at"],
                    errors="coerce"
                )

                anomaly_history = (
                    anomaly_history
                    .dropna(subset=["created_at"])
                    .sort_values("created_at")
                )

                fig_anomaly_trend = px.line(
                    anomaly_history,
                    x="created_at",
                    y="anomaly_ratio",
                    markers=True,
                    title="Évolution du taux d'anomalies"
                )

                fig_anomaly_trend.update_traces(
                    line_color="#C25B5B"
                )

                apply_plot_theme(
                    fig_anomaly_trend
                )

                st.plotly_chart(
                    fig_anomaly_trend,
                    width="stretch"
                )

                anomaly_trend_insight = interpret_metric_trend(
                    history_df=anomaly_history,
                    metric_column="anomaly_ratio",
                    metric_label="taux d'anomalies",
                    higher_is_better=False,
                    significant_change=0.5
                )

                display_chart_interpretation(
                    title="Évolution du taux d'anomalies",
                    insight=anomaly_trend_insight,
                    expanded=False
                )

            else:
                st.info(
                    "Historique anomalies non disponible."
                )

    else:
        st.info(
            "Aucune exécution ML historisée."
        )

except Exception as error:
    st.error(
        f"Erreur lors du chargement de l'historique : {error}"
    )