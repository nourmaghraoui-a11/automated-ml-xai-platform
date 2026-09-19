"""
ui_theme.py
============
Système de design partagé pour l'application "Pipeline ML Générique".

Direction visuelle : "Signal" — fond anthracite neutre, quasiment
monochrome, avec une seule couleur vive (un ambre/or chaud) réservée
aux éléments qui portent une vraie information : la donnée active,
la sévérité d'une anomalie, l'action principale, l'onglet sélectionné.
Le reste de l'interface reste volontairement sourd (gris, bordures
fines) pour que cette couleur agisse comme un signal au sens propre :
elle attire l'œil uniquement là où quelque chose mérite l'attention
du technicien ou du manager — cohérent avec le rôle du pipeline
(détection de structure, d'anomalies, de risques de réintégration).

Typographie : Sora pour les titres (display géométrique, caractère
mais lisible), Manrope pour le texte courant, et JetBrains Mono
réservé aux éléments techniques/donnée (métriques, valeurs, code,
identifiants de colonnes) — pour marquer visuellement la frontière
entre "propos" et "donnée brute".

Ce module n'altère aucune logique métier : il ne fait qu'injecter du
CSS et fournir de petits composants d'habillage (en-têtes, badges
d'étape, puces de statut, cartes) utilisés par les pages. L'API est
volontairement identique à la version précédente (apply_theme, hero,
stage_header, pill, card) pour rester un remplacement direct.
"""

import streamlit as st


# ------------------------------------------------------------------
# TOKENS
# ------------------------------------------------------------------

COLORS = {
    # Fond / surfaces — anthracite neutre, jamais noir pur
    "bg": "#131417",
    "bg_alt": "#0E0F11",
    "surface": "#1B1D21",
    "surface_alt": "#232529",
    "border": "#2E3136",
    "border_soft": "#25272C",

    # Texte
    "text": "#EDEEF0",
    "text_muted": "#9299A3",
    "text_faint": "#5B6068",

    # Accent unique — "Signal" (ambre/or chaud, utilisé avec parcimonie)
    "accent": "#F2A93C",
    "accent_soft": "rgba(242, 169, 60, 0.14)",
    "accent_border": "rgba(242, 169, 60, 0.45)",
    "on_accent": "#181205",

    # Statuts sémantiques — volontairement sourds, ne rivalisent pas
    # avec l'accent (utilisés uniquement pour une info fonctionnelle :
    # succès / avertissement / erreur, pas comme décoration)
    "success": "#4FA57C",
    "success_soft": "rgba(79, 165, 124, 0.14)",
    "warning": "#C98A3E",
    "warning_soft": "rgba(201, 138, 62, 0.14)",
    "danger": "#C25B5B",
    "danger_soft": "rgba(194, 91, 91, 0.14)",
}

FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Sora:wght@500;600;700&"
    "family=Manrope:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@400;500;600&display=swap"
)


def _css() -> str:
    c = COLORS
    return f"""
    @import url('{FONT_IMPORT}');

    html, body, [class*="css"] {{
        font-family: 'Manrope', sans-serif;
    }}

    .stApp {{
        background-color: {c['bg']};
        background-image: radial-gradient(
            circle at 15% -10%,
            rgba(242, 169, 60, 0.06),
            transparent 45%
        );
        color: {c['text']};
    }}

    [data-testid="stHeader"] {{
        background-color: transparent;
    }}

    [data-testid="stSidebar"] {{
        background-color: {c['bg_alt']};
        border-right: 1px solid {c['border']};
    }}

    [data-testid="stSidebar"] * {{
        color: {c['text']} !important;
    }}

    [data-testid="stSidebarNav"] a {{
        font-family: 'Manrope', sans-serif;
        font-size: 0.88rem;
        border-radius: 6px;
        margin: 1px 0;
    }}

    [data-testid="stSidebarNav"] a:hover {{
        background-color: {c['surface_alt']};
    }}

    [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background-color: {c['accent_soft']};
        box-shadow: inset 2px 0 0 {c['accent']};
    }}

    h1, h2, h3 {{
        font-family: 'Sora', sans-serif !important;
        color: {c['text']} !important;
        letter-spacing: -0.01em;
        font-weight: 600 !important;
    }}

    p, span, label, li {{
        color: {c['text']};
    }}

    /* ---- métriques : lecture rapide, valeur en mono ---- */
    [data-testid="stMetric"] {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-left: 2px solid {c['border']};
        border-radius: 10px;
        padding: 0.95rem 1.1rem 0.85rem 1.1rem;
        transition: border-left-color 0.15s ease;
    }}
    [data-testid="stMetric"]:hover {{
        border-left-color: {c['accent']};
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'Manrope', sans-serif !important;
        text-transform: uppercase;
        font-size: 0.68rem !important;
        letter-spacing: 0.07em;
        font-weight: 600 !important;
        color: {c['text_muted']} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.55rem !important;
        color: {c['text']} !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* ---- conteneurs / expanders ---- */
    [data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 10px;
    }}

    [data-testid="stExpander"] summary {{
        font-family: 'Manrope', sans-serif;
        font-weight: 600;
    }}

    /* ---- onglets ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        border-bottom: 1px solid {c['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Manrope', sans-serif;
        font-weight: 600;
        font-size: 0.86rem;
        color: {c['text_muted']};
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1rem;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {c['text']};
        background-color: {c['surface_alt']};
    }}
    .stTabs [aria-selected="true"] {{
        color: {c['accent']} !important;
        background-color: {c['accent_soft']} !important;
        box-shadow: inset 0 -2px 0 {c['accent']};
    }}

    /* ---- boutons ---- */
    .stButton > button, .stDownloadButton > button {{
        font-family: 'Manrope', sans-serif;
        font-weight: 600;
        font-size: 0.86rem;
        background-color: {c['surface_alt']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {c['accent_border']};
        color: {c['accent']};
    }}
    .stButton > button[kind="primary"] {{
        background-color: {c['accent']};
        color: {c['on_accent']};
        border: 1px solid {c['accent']};
        font-weight: 700;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {c['accent']};
        opacity: 0.92;
        color: {c['on_accent']};
    }}

    /* ---- inputs ---- */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    [data-baseweb="select"] {{
        background-color: {c['surface']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }}

    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
        border-color: {c['accent']} !important;
        box-shadow: 0 0 0 1px {c['accent']} !important;
    }}

    /* ---- radios / checkbox labels ---- */
    .stRadio label, .stCheckbox label {{
        font-family: 'Manrope', sans-serif;
    }}

    .stRadio [role="radiogroup"] label[data-baseweb="radio"] div:first-child {{
        border-color: {c['border']} !important;
    }}

    /* ---- alerts ---- */
    [data-testid="stAlert"] {{
        border-radius: 8px;
        border: 1px solid {c['border']};
        font-family: 'Manrope', sans-serif;
    }}

    /* ---- dataframes / json ---- */
    [data-testid="stDataFrame"], [data-testid="stJson"] {{
        border: 1px solid {c['border']};
        border-radius: 10px;
        overflow: hidden;
        font-family: 'JetBrains Mono', monospace !important;
    }}

    hr {{
        border-color: {c['border']} !important;
    }}

    /* ---- composants custom (voir fonctions ci-dessous) ---- */
    .sig-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {c['accent']};
        margin-bottom: 0.3rem;
        font-weight: 500;
    }}

    .sig-hero-title {{
        font-family: 'Sora', sans-serif;
        font-size: 2.15rem;
        font-weight: 700;
        color: {c['text']};
        margin: 0 0 0.4rem 0;
        line-height: 1.15;
    }}

    .sig-hero-subtitle {{
        color: {c['text_muted']};
        font-size: 1rem;
        max-width: 46rem;
        line-height: 1.6;
    }}

    /* ---- rail d'étape : numérotation réelle du pipeline ---- */
    .sig-rail {{
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
        margin: 2.2rem 0 0.7rem 0;
    }}
    .sig-rail-badge {{
        flex-shrink: 0;
        width: 2.15rem;
        height: 2.15rem;
        border-radius: 7px;
        border: 1px solid {c['border']};
        background-color: {c['surface']};
        color: {c['accent']};
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .sig-rail-text {{
        border-bottom: 1px solid {c['border']};
        flex-grow: 1;
        padding-bottom: 0.7rem;
        padding-top: 0.15rem;
    }}
    .sig-rail-title {{
        font-family: 'Sora', sans-serif;
        font-size: 1.28rem;
        font-weight: 600;
        color: {c['text']};
        margin: 0;
    }}
    .sig-rail-desc {{
        color: {c['text_muted']};
        font-size: 0.87rem;
        margin-top: 0.15rem;
        line-height: 1.5;
    }}

    /* ---- puces de statut ---- */
    .sig-pill {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.03em;
        font-weight: 500;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
    }}
    .sig-pill-accent {{
        color: {c['accent']};
        background-color: {c['accent_soft']};
        border: 1px solid {c['accent_border']};
    }}
    .sig-pill-success {{
        color: {c['success']};
        background-color: {c['success_soft']};
        border: 1px solid rgba(79, 165, 124, 0.4);
    }}
    .sig-pill-warning {{
        color: {c['warning']};
        background-color: {c['warning_soft']};
        border: 1px solid rgba(201, 138, 62, 0.4);
    }}
    .sig-pill-danger {{
        color: {c['danger']};
        background-color: {c['danger_soft']};
        border: 1px solid rgba(194, 91, 91, 0.4);
    }}
    .sig-pill-neutral {{
        color: {c['text_muted']};
        background-color: {c['surface_alt']};
        border: 1px solid {c['border']};
    }}

    /* ---- cartes ---- */
    .sig-card {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        height: 100%;
    }}
    .sig-card-title {{
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 1.02rem;
        color: {c['text']};
        margin-bottom: 0.45rem;
    }}
    .sig-card-body {{
        color: {c['text_muted']};
        font-size: 0.89rem;
        line-height: 1.6;
    }}
    .sig-card-body ul {{ padding-left: 1.1rem; margin: 0.3rem 0 0 0; }}
    """


def apply_theme() -> None:
    """Injecte le CSS du système de design. À appeler en tête de chaque page."""
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    """En-tête de page : eyebrow technique + titre display + sous-titre."""
    st.markdown(
        f"""
        <div>
            <div class="sig-eyebrow">{eyebrow}</div>
            <div class="sig-hero-title">{title}</div>
            <div class="sig-hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_header(num: int, total: int, title: str, desc: str = "") -> None:
    """
    En-tête de section stylé comme un jalon de pipeline : badge numéroté
    (§ num / total) relié à un séparateur. L'ordre est réel — ce sont les
    étapes effectives du pipeline, pas une décoration.
    """
    st.markdown(
        f"""
        <div class="sig-rail">
            <div class="sig-rail-badge">{num:02d}</div>
            <div class="sig-rail-text">
                <p class="sig-rail-title">{title}</p>
                {f'<p class="sig-rail-desc">{desc}</p>' if desc else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, tone: str = "accent") -> str:
    """
    Retourne le HTML d'une puce de statut (à insérer dans un st.markdown).
    tone : "accent" | "success" | "warning" | "danger" | "neutral"
    """
    return f'<span class="sig-pill sig-pill-{tone}">{text}</span>'


def card(title: str, body_html: str) -> None:
    """Petite carte avec titre et corps HTML libre (listes, texte...)."""
    st.markdown(
        f"""
        <div class="sig-card">
            <div class="sig-card-title">{title}</div>
            <div class="sig-card-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )