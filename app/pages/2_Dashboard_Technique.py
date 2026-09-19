import os
import sys
import json
import sqlite3

import pandas as pd
import streamlit as st


# ============================================================
# AJOUT DU CHEMIN RACINE DU PROJET
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(APP_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from src.warehouse import (
    get_pipeline_runs,
    get_features_history,
    get_reports_history
)

from src.pipeline_controller import start_pipeline

from src.feature_manager import (
    build_feature_registry,
    summarize_feature_status,
    get_exclusion_reasons
)

from src.ui_theme import apply_theme, hero, stage_header, pill, card

from src.session_store import save_full_session
from src.live_sync import (
    sync_technique_page,
    set_active_run_id,
    get_active_run_id
)
from src.scheduler_ui import render_live_sync_controls, render_scheduler_panel

# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="Dashboard Technique",
    page_icon="🛠️",
    layout="wide"
)

apply_theme()

hero(
    eyebrow="Espace technique",
    title="Dashboard technique — Préparation des données",
    subtitle="Ingestion, intégration, détection de structure et preprocessing automatique.",
)

TOTAL_STEPS = 11


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "manual_features" not in st.session_state:
    st.session_state.manual_features = []

if "integration_config" not in st.session_state:
    st.session_state.integration_config = None

if "data_mode" not in st.session_state:
    st.session_state.data_mode = None


# ------------------------------------------------------------
# Synchronisation avec le planificateur (processus indépendant).
#
# Le planificateur et ce dashboard sont deux processus Python séparés :
# le planificateur ne peut pas écrire directement dans le
# st.session_state d'une session Streamlit déjà ouverte. La solution
# est donc que CE dashboard vérifie lui-même, à chaque exécution du
# script, si un run plus récent existe sur disque (écrit par
# session_store.py, que ce soit par ce dashboard ou par le
# planificateur) — et se recharge alors tout seul.
#
# Cet appel gère à la fois :
#   - le premier chargement (session vide -> restaure le dernier run
#     disponible, y compris après un redémarrage du terminal) ;
#   - la mise à jour en direct pendant que la page reste ouverte,
#     à condition que le script soit ré-exécuté — ce que fait
#     automatiquement l'auto-rafraîchissement configuré plus bas.
# ------------------------------------------------------------

if get_active_run_id() is None and st.session_state.pipeline_result is not None:
    # Un résultat existe déjà en mémoire (rerun Streamlit normal) mais
    # son run_id n'était pas encore suivi : on le déduit sans recharger.
    set_active_run_id(st.session_state.pipeline_result.get("run_id"))

_sync_result = sync_technique_page()

with st.expander("🗓️ Planificateur & synchronisation en direct", expanded=bool(_sync_result)):
    render_live_sync_controls(
        _sync_result,
        label="le dataset préparé",
        key_prefix="tech_live_sync"
    )
    st.divider()
    render_scheduler_panel()


# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================

stage_header(1, TOTAL_STEPS, "Chargement des données", "CSV, base SQL, ou sources multiples à fusionner")

source_choice = st.radio(
    "Choisir le type de source",
    options=[
        "Fichier CSV unique",
        "Base SQLite / SQL",
        "Sources multiples CSV"
    ],
    horizontal=True
)


# ------------------------------------------------------------
# OPTION 1 : CSV UNIQUE
# ------------------------------------------------------------

if source_choice == "Fichier CSV unique":

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"],
        key="single_csv"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.session_state.df = df
            st.session_state.data_mode = "single"
            st.session_state.integration_config = None
            st.session_state.pipeline_result = None

            st.success("Fichier CSV chargé avec succès.")

            st.subheader("Aperçu des données")
            st.dataframe(df.head(), use_container_width=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Nombre de lignes", df.shape[0])

            with col2:
                st.metric("Nombre de colonnes", df.shape[1])

            with col3:
                st.metric("Valeurs manquantes", int(df.isnull().sum().sum()))

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier CSV : {e}")


# ------------------------------------------------------------
# OPTION 2 : BASE SQLITE / SQL
# ------------------------------------------------------------

elif source_choice == "Base SQLite / SQL":

    st.markdown("### Connexion à une base SQLite")

    database_file = st.file_uploader(
        "Importer une base SQLite (.db, .sqlite)",
        type=["db", "sqlite", "sqlite3"],
        key="sqlite_db"
    )

    query = st.text_area(
        "Requête SQL",
        value="SELECT * FROM users;",
        height=120
    )

    if database_file is not None:

        temp_db_path = os.path.join("data", "raw", database_file.name)
        os.makedirs(os.path.dirname(temp_db_path), exist_ok=True)

        with open(temp_db_path, "wb") as f:
            f.write(database_file.getbuffer())

        st.info(f"Base chargée temporairement : {temp_db_path}")

        if st.button("Exécuter la requête SQL"):

            try:
                conn = sqlite3.connect(temp_db_path)
                df = pd.read_sql_query(query, conn)
                conn.close()

                st.session_state.df = df
                st.session_state.data_mode = "single"
                st.session_state.integration_config = None
                st.session_state.pipeline_result = None

                st.success("Données SQL chargées avec succès.")

                st.subheader("Aperçu des données SQL")
                st.dataframe(df.head(), use_container_width=True)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Nombre de lignes", df.shape[0])

                with col2:
                    st.metric("Nombre de colonnes", df.shape[1])

                with col3:
                    st.metric("Valeurs manquantes", int(df.isnull().sum().sum()))

            except Exception as e:
                st.error(f"Erreur lors de l'exécution SQL : {e}")

    else:
        st.warning("Veuillez importer une base SQLite avant d'exécuter une requête.")


# ------------------------------------------------------------
# OPTION 3 : SOURCES MULTIPLES CSV
# ------------------------------------------------------------

elif source_choice == "Sources multiples CSV":

    st.markdown("### Importer plusieurs fichiers CSV")

    st.info(
        "Exemple : users.csv, transactions.csv, logins.csv. "
        "Le pipeline va harmoniser et fusionner les sources autour d'une clé commune."
    )

    uploaded_files = st.file_uploader(
        "Importer plusieurs fichiers CSV",
        type=["csv"],
        accept_multiple_files=True,
        key="multiple_csv"
    )

    join_key = st.text_input(
        "Clé de jointure principale",
        value="user_id"
    )

    st.markdown("### Configuration des tables événementielles")

    st.write(
        "Si une source contient plusieurs lignes par utilisateur, "
        "comme transactions ou logins, elle doit être agrégée."
    )

    enable_default_aggregations = st.checkbox(
        "Utiliser des agrégations automatiques simples",
        value=True
    )

    if uploaded_files:

        dataframes = {}

        for uploaded_file in uploaded_files:
            try:
                source_name = (
                    uploaded_file.name
                    .replace(".csv", "")
                    .replace(" ", "_")
                    .lower()
                )

                df_source = pd.read_csv(uploaded_file)
                dataframes[source_name] = df_source

            except Exception as e:
                st.error(f"Erreur lors du chargement de {uploaded_file.name} : {e}")

        if dataframes:

            st.session_state.df = dataframes
            st.session_state.data_mode = "multiple"
            st.session_state.pipeline_result = None

            st.success(f"{len(dataframes)} sources chargées avec succès.")

            st.subheader("Sources chargées")

            for name, df_source in dataframes.items():

                with st.expander(
                    f"Source : {name} — {df_source.shape[0]} lignes, {df_source.shape[1]} colonnes"
                ):
                    st.dataframe(df_source.head(), use_container_width=True)
                    st.write("Colonnes :", list(df_source.columns))

            integration_config = {
                "join_key": join_key,
                "event_aggregations": {}
            }

            if enable_default_aggregations:

                for name, df_source in dataframes.items():

                    columns_lower = [col.lower() for col in df_source.columns]

                    if "transaction" in name or "amount" in columns_lower:

                        aggregations = {}

                        if "amount" in df_source.columns:
                            aggregations["amount"] = ["sum", "mean", "count"]

                        if "transaction_id" in df_source.columns:
                            aggregations["transaction_id"] = ["count"]

                        if aggregations:
                            integration_config["event_aggregations"][name] = {
                                "group_key": join_key,
                                "aggregations": aggregations
                            }

                    if "login" in name:

                        aggregations = {}

                        if "login_id" in df_source.columns:
                            aggregations["login_id"] = ["count"]

                        if aggregations:
                            integration_config["event_aggregations"][name] = {
                                "group_key": join_key,
                                "aggregations": aggregations
                            }

            st.session_state.integration_config = integration_config

            st.subheader("Configuration d'intégration générée")
            st.json(integration_config)


# ============================================================
# 2. EXECUTION DU MODULE DE PREPARATION
# ============================================================

stage_header(2, TOTAL_STEPS, "Exécution du module de préparation des données")

if st.session_state.df is not None:

    if st.button("Lancer la préparation des données", type="primary"):

        try:
            with st.spinner("Exécution du pipeline de préparation en cours..."):

                result = start_pipeline(
                    data=st.session_state.df,
                    integration_config=st.session_state.get("integration_config"),
                    manual_features_to_keep=None
                )

                st.session_state.pipeline_result = result

                # Persistance sur disque : survit à un redémarrage de
                # Streamlit, du terminal, ou à une nouvelle session.
                save_full_session(
                    run_id=result["run_id"],
                    prep_result=result
                )
                # Marque ce run comme "déjà affiché" par CETTE session :
                # sans cela, le prochain tick de synchronisation en direct
                # le détecterait comme "nouveau" et le rechargerait pour
                # rien (le résultat est déjà en mémoire, on vient de le
                # produire nous-mêmes).
                set_active_run_id(result["run_id"])

            st.success("Préparation des données exécutée avec succès.")

        except Exception as e:
            st.error(f"Erreur pendant l'exécution du pipeline : {e}")

else:
    st.warning("Veuillez charger une source de données avant de lancer le pipeline.")


# ============================================================
# 3. AFFICHAGE DES RESULTATS
# ============================================================

result = st.session_state.pipeline_result

if result is not None:

    structure = result["structure"]
    structure_summary = result["structure_summary"]
    preprocessing_info = result["preprocessing_info"]
    preprocessing_reports = result["preprocessing_reports"]
    X_ready = result["X_ready"]
    y = result["y"]
    paths = result["paths"]

    st.divider()

    # ========================================================
    # STRUCTURE DETECTEE
    # ========================================================

    stage_header(3, TOTAL_STEPS, "Structure détectée du dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Lignes", structure_summary["nombre_lignes"])

    with col2:
        st.metric("Colonnes", structure_summary["nombre_colonnes"])

    with col3:
        st.metric("Features exploitables", structure_summary["nb_features_exploitables"])

    with col4:
        st.metric("Colonnes exclues", structure_summary["nb_colonnes_exclues"])

    st.subheader("Types de colonnes détectés")

    tab_num, tab_cat, tab_date, tab_id, tab_label = st.tabs([
        "Numériques",
        "Catégorielles",
        "Dates",
        "Identifiants",
        "Labels potentiels"
    ])

    with tab_num:
        st.write(structure.get("numeric_columns", []))

    with tab_cat:
        st.write(structure.get("categorical_columns", []))

    with tab_date:
        st.write(structure.get("date_columns", []))

    with tab_id:
        st.write(structure.get("id_columns", []))

    with tab_label:
        st.write(structure.get("possible_label_columns", []))


    # ========================================================
    # COLONNES EXCLUES
    # ========================================================

    stage_header(4, TOTAL_STEPS, "Colonnes exclues automatiquement")

    excluded_columns = structure.get("excluded_columns", [])
    exclusion_reasons = get_exclusion_reasons(structure, excluded_columns)

    if excluded_columns:
        st.markdown(pill(f"{len(excluded_columns)} colonne(s) exclue(s)", tone="warning"), unsafe_allow_html=True)
        st.write("")

        excluded_df = pd.DataFrame({
            "Colonne": excluded_columns,
            "Raison": [exclusion_reasons.get(col, "") for col in excluded_columns]
        })

        st.dataframe(excluded_df, width="stretch")

    else:
        st.info("Aucune colonne exclue automatiquement.")


    # ========================================================
    # REGISTRE DES FEATURES
    # ========================================================

    stage_header(5, TOTAL_STEPS, "Registre des features")

    feature_registry = build_feature_registry(
        structure=structure,
        preprocessing_info=preprocessing_info
    )

    feature_summary = summarize_feature_status(feature_registry)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Colonnes totales", feature_summary["total_columns"])

    with col2:
        st.metric("Features", feature_summary["total_features"])

    with col3:
        st.metric("Exclues", feature_summary["total_excluded"])

    with col4:
        st.metric("Réintégrées", feature_summary["total_reintegrated"])

    registry_df = pd.DataFrame(feature_registry)

    if not registry_df.empty:
        display_registry = registry_df[
            [
                "column",
                "detected_type",
                "is_excluded",
                "is_feature",
                "is_reintegrated",
                "exclusion_reason"
            ]
        ]

        st.dataframe(display_registry, width="stretch")


    # ========================================================
    # RAPPORTS DE PREPROCESSING
    # ========================================================

    stage_header(6, TOTAL_STEPS, "Rapports de preprocessing")

    st.subheader("Informations générales du preprocessing")
    st.json(preprocessing_info)

    st.subheader("Valeurs manquantes")

    missing_report = preprocessing_reports.get("missing_values")

    if missing_report is not None and not missing_report.empty:
        st.dataframe(missing_report, width="stretch")
    else:
        st.info("Aucune valeur manquante détectée après traitement.")

    st.subheader("Outliers détectés")

    outliers_report = preprocessing_reports.get("outliers_detected")

    if outliers_report is not None and not outliers_report.empty:
        st.dataframe(outliers_report, width="stretch")
    else:
        st.info("Aucun rapport d'outliers disponible.")

    st.subheader("Transformations de skewness")

    skew_report = preprocessing_reports.get("skew_transformations")

    if skew_report is not None and not skew_report.empty:
        st.dataframe(skew_report, width="stretch")
    else:
        st.info("Aucune transformation de skewness appliquée.")


    # ========================================================
    # DATASET PREPARE
    # ========================================================

    stage_header(7, TOTAL_STEPS, "Dataset préparé pour Machine Learning")

    st.write("Aperçu de X_ready :")
    st.dataframe(X_ready.head(), width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Nombre de lignes X_ready", X_ready.shape[0])

    with col2:
        st.metric("Nombre de features X_ready", X_ready.shape[1])

    if y is not None:
        st.markdown(pill(f"Label détecté : {preprocessing_info.get('label_column')}", tone="success"), unsafe_allow_html=True)
        st.write("")
        st.write("Aperçu du label y :")
        st.dataframe(y.head())
    else:
        st.info("Aucun label détecté.")


    # ========================================================
    # ARTEFACTS
    # ========================================================

    stage_header(8, TOTAL_STEPS, "Artefacts et rapports sauvegardés")

    st.write("Chemins générés :")

    st.json({
        "artifacts_path": preprocessing_info.get("artifacts_saved_at"),
        "reports_run_dir": paths.get("reports_run_dir"),
        "structure_report_path": paths.get("structure_report_path"),
        "preprocessing_report_path": paths.get("preprocessing_report_path"),
        "data_preparation_report_path": paths.get("data_preparation_report_path"),
    })

    warehouse_status = result.get("warehouse_status")

    if warehouse_status is not None:
        st.subheader("Statut Data Warehouse")
        st.json(warehouse_status)


    # ========================================================
    # REINTEGRATION MANUELLE
    # ========================================================

    stage_header(9, TOTAL_STEPS, "Réintégration contrôlée des features exclues")

    if excluded_columns:

        selected_features = st.multiselect(
            "Sélectionner les colonnes exclues à réintégrer",
            options=excluded_columns
        )

        st.warning(
            "Attention : la réintégration d'un identifiant ou d'un label potentiel "
            "peut provoquer une fuite d'information ou fausser les modèles."
        )

        if st.button("Relancer le pipeline avec les features sélectionnées"):

            try:
                with st.spinner("Relance du pipeline avec réintégration..."):

                    new_result = start_pipeline(
                        data=st.session_state.df,
                        integration_config=st.session_state.get("integration_config"),
                        manual_features_to_keep=selected_features
                    )

                    st.session_state.pipeline_result = new_result
                    st.session_state.manual_features = selected_features

                    save_full_session(
                        run_id=new_result["run_id"],
                        prep_result=new_result
                    )
                    set_active_run_id(new_result["run_id"])

                st.success("Pipeline relancé avec succès.")
                st.rerun()

            except Exception as e:
                st.error(f"Erreur lors de la relance : {e}")

    else:
        st.info("Aucune colonne exclue disponible pour réintégration.")


    # ========================================================
    # EXPORTS
    # ========================================================

    stage_header(10, TOTAL_STEPS, "Export des résultats de préparation")

    csv_x_ready = X_ready.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Télécharger X_ready en CSV",
        data=csv_x_ready,
        file_name="X_ready_data_preparation.csv",
        mime="text/csv"
    )

    if y is not None:
        y_df = pd.DataFrame({
            preprocessing_info.get("label_column", "label"): y
        })

        csv_y = y_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Télécharger y en CSV",
            data=csv_y,
            file_name="y_label_data_preparation.csv",
            mime="text/csv"
        )

    preprocessing_info_json = json.dumps(
        preprocessing_info,
        indent=4,
        ensure_ascii=False,
        default=str
    )

    st.download_button(
        label="Télécharger preprocessing_info en JSON",
        data=preprocessing_info_json,
        file_name="preprocessing_info_data_preparation.json",
        mime="application/json"
    )

    structure_json = json.dumps(
        structure,
        indent=4,
        ensure_ascii=False,
        default=str
    )

    st.download_button(
        label="Télécharger structure détectée en JSON",
        data=structure_json,
        file_name="structure_detectee_data_preparation.json",
        mime="application/json"
    )

else:
    st.info("Lancez le pipeline pour afficher les résultats techniques et les exports.")

# ========================================================
# HISTORIQUE DATA WAREHOUSE
# ========================================================

stage_header(11, TOTAL_STEPS, "Historique du Data Warehouse", "Runs, features et rapports historisés")

st.markdown("""
Cette section affiche les exécutions historisées du module de préparation des données.
Elle permet au technicien de consulter les anciens runs, les features exclues,
les features réintégrées, les features finales et les rapports techniques.
""")

try:
    runs_history = get_pipeline_runs()

    if not runs_history.empty:

        st.subheader("Historique des exécutions")

        st.dataframe(
            runs_history,
            width="stretch"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Nombre d'exécutions", runs_history.shape[0])

        with col2:
            last_status = runs_history.iloc[0]["status"]
            st.metric("Dernier statut", last_status)

        with col3:
            last_run_id = runs_history.iloc[0]["run_id"]
            st.metric("Dernier run", last_run_id)

    else:
        st.info("Aucune exécution historisée pour le moment.")

except Exception as e:
    st.error(f"Erreur lors du chargement de l'historique des exécutions : {e}")


# ========================================================
# HISTORIQUE DES FEATURES
# ========================================================

st.subheader("Historique des features")

feature_history_choice = st.selectbox(
    "Choisir le type d'historique des features",
    options=[
        "Features exclues",
        "Features réintégrées",
        "Features finales"
    ]
)

table_mapping = {
    "Features exclues": "excluded_features",
    "Features réintégrées": "reintegrated_features",
    "Features finales": "final_features"
}

try:
    selected_table = table_mapping[feature_history_choice]

    features_history = get_features_history(selected_table)

    if not features_history.empty:
        st.dataframe(
            features_history,
            width="stretch"
        )
    else:
        st.info("Aucune donnée disponible pour cet historique.")

except Exception as e:
    st.error(f"Erreur lors du chargement de l'historique des features : {e}")


# ========================================================
# HISTORIQUE DES RAPPORTS TECHNIQUES
# ========================================================

st.subheader("Historique des rapports techniques")

try:
    reports_history = get_reports_history()

    if not reports_history.empty:
        st.dataframe(
            reports_history,
            width="stretch"
        )
    else:
        st.info("Aucun rapport technique historisé pour le moment.")

except Exception as e:
    st.error(f"Erreur lors du chargement de l'historique des rapports : {e}")