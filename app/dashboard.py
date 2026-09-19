import streamlit as st

from src.ui_theme import apply_theme, hero, pill, card


# ============================================================
# CONFIGURATION GENERALE DE L'APPLICATION
# ============================================================

st.set_page_config(
    page_title="Pipeline ML Générique",
    page_icon="🤖",
    layout="wide"
)

apply_theme()


# ============================================================
# PAGE D'ACCUEIL
# ============================================================

hero(
    eyebrow="Plateforme Machine Learning",
    title="Pipeline ML générique avec interprétation automatique",
    subtitle=(
        "Ingestion, intégration, détection de structure, preprocessing, "
        "segmentation utilisateurs et détection d'anomalies — dans une "
        "seule plateforme, avec explicabilité intégrée."
    ),
)

st.write("")

st.markdown(
    pill("Module de préparation disponible", tone="success"),
    unsafe_allow_html=True,
)

st.write("")
st.write("")

capabilities = [
    ("ti-database-import", "Ingestion multi-sources", "CSV, SQL, ou fusion de plusieurs sources en un dataset unifié."),
    ("ti-list-search", "Détection de structure", "Types, dates, identifiants et labels potentiels détectés automatiquement."),
    ("ti-adjustments-horizontal", "Preprocessing automatique", "Nettoyage, imputation, encodage et mise à l'échelle sans configuration manuelle."),
    ("ti-shield-check", "Gestion contrôlée des features", "Colonnes exclues consultables, réintégration validée avec alerte de risque."),
    ("ti-report", "Rapports techniques", "Génération et historisation des rapports à chaque exécution."),
    ("ti-archive", "Artefacts versionnés", "Sauvegarde des encodeurs, du scaler et des métadonnées du pipeline."),
]

cols = st.columns(3)

for i, (icon, title, desc) in enumerate(capabilities):
    with cols[i % 3]:
        card(title, f"<div style='margin-bottom:0.5rem;font-size:1.1rem;'>{title}</div>{desc}")
        st.write("")


# ============================================================
# NAVIGATION
# ============================================================

st.divider()

st.markdown("### Navigation")

col1, col2 = st.columns(2)

with col1:
    card(
        "Dashboard analytique",
        """
        Destiné aux managers et décideurs.
        <ul>
            <li>Indicateurs clés</li>
            <li>Segments utilisateurs</li>
            <li>Anomalies détectées</li>
            <li>Interprétations automatiques</li>
            <li>Recommandations</li>
        </ul>
        """,
    )

with col2:
    card(
        "Dashboard technique",
        """
        Destiné aux utilisateurs techniques.
        <ul>
            <li>Structure détectée du dataset</li>
            <li>Colonnes exclues et raisons</li>
            <li>Rapports de preprocessing</li>
            <li>Artefacts générés</li>
            <li>Réintégration contrôlée des features</li>
        </ul>
        """,
    )

st.write("")
st.info("Utilisez le menu latéral pour accéder aux différents espaces de l'application.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Projet : Pipeline ML générique avec interprétation automatique — "
    "Préparation des données, segmentation utilisateurs et détection d'anomalies."
)