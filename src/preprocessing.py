# ============================================================
# preprocessing.py
# Preprocessing automatique des données
# ============================================================

import os
import re
import joblib
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    LabelEncoder,
    PowerTransformer
)
from sklearn.feature_selection import VarianceThreshold


# ============================================================
# 1. PRE-NETTOYAGE LEGER
# ============================================================

def basic_cleaning(df):
    """
    Réalise un pré-nettoyage léger :
    - suppression des lignes totalement vides
    - suppression des colonnes totalement vides
    - suppression des doublons
    """

    try:
        df = df.copy()

        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        df = df.drop_duplicates().reset_index(drop=True)

        return df

    except Exception as e:
        raise Exception(f"Error during basic cleaning: {e}")


# ============================================================
# 2. SEPARATION FEATURES / LABEL / COLONNES EXCLUES
# ============================================================

def split_features_label(df, structure, manual_features_to_keep=None):
    """
    Sépare les features X et le label y si un label potentiel est détecté.
    Permet aussi de réintégrer manuellement certaines colonnes exclues.
    """

    if manual_features_to_keep is None:
        manual_features_to_keep = []

    id_columns = structure.get("id_columns", [])
    possible_label_columns = structure.get("possible_label_columns", [])

    y = None
    label_col = None

    if len(possible_label_columns) > 0:
        label_col = possible_label_columns[0]

        if label_col in df.columns:
            y = df[label_col].copy()

    excluded_columns = list(set(id_columns + possible_label_columns))

    reintegrated_columns = [
        col for col in manual_features_to_keep
        if col in df.columns
    ]

    excluded_columns = [
        col for col in excluded_columns
        if col not in reintegrated_columns
    ]

    X = df.drop(columns=excluded_columns, errors="ignore")

    return X, y, label_col, excluded_columns, reintegrated_columns


def validate_manual_features(
    manual_features_to_keep,
    structure,
    allow_high_risk_features=False
):
    """
    Vérifie les colonnes sélectionnées manuellement avant réintégration.

    Si allow_high_risk_features=False :
        - Les colonnes de type ID ou label potentiel sont refusées.

    Si allow_high_risk_features=True :
        - Les colonnes risquées peuvent être réintégrées après avertissement.
    """

    if manual_features_to_keep is None:
        manual_features_to_keep = []

    dataset_columns = structure.get("columns", [])
    id_columns = structure.get("id_columns", [])
    label_columns = structure.get("possible_label_columns", [])

    risky_columns = list(set(id_columns + label_columns))

    warnings = []
    allowed_columns = []

    for col in manual_features_to_keep:

        if dataset_columns and col not in dataset_columns:
            warnings.append({
                "column": col,
                "warning": "Colonne inexistante dans le dataset.",
                "risk_level": "invalide",
                "allowed": False
            })
            continue

        if col in risky_columns:

            warnings.append({
                "column": col,
                "warning": "Colonne risquée : identifiant ou label potentiel.",
                "risk_level": "élevé",
                "allowed": allow_high_risk_features
            })

            if allow_high_risk_features:
                allowed_columns.append(col)

        else:
            allowed_columns.append(col)

    return allowed_columns, warnings


def get_usable_columns_by_type(structure, X, excluded_columns=None):
    """
    Récupère les colonnes utilisables par type.

    Important :
    On utilise excluded_columns après réintégration manuelle,
    et non uniquement structure["excluded_columns"].
    """

    if excluded_columns is None:
        excluded_columns = structure.get("excluded_columns", [])

    numeric_cols = [
        col for col in structure.get("numeric_columns", [])
        if col in X.columns and col not in excluded_columns
    ]

    categorical_cols = [
        col for col in structure.get("categorical_columns", [])
        if col in X.columns and col not in excluded_columns
    ]

    date_cols = [
        col for col in structure.get("date_columns", [])
        if col in X.columns and col not in excluded_columns
    ]

    return numeric_cols, categorical_cols, date_cols


# ============================================================
# 3. VALEURS MANQUANTES
# ============================================================

def detect_missing_values(df):
    """
    Détecte les valeurs manquantes et retourne un rapport.
    """

    if len(df) == 0:
        return pd.DataFrame(
            columns=["column", "missing_count", "missing_percentage"]
        )

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isnull().sum().values,
        "missing_percentage": (
            df.isnull().sum() / len(df) * 100
        ).round(2).values
    })

    report = report[report["missing_count"] > 0]

    report = report.sort_values(
        by="missing_percentage",
        ascending=False
    ).reset_index(drop=True)

    return report


def drop_high_missing_columns(df, threshold=60, columns_to_drop=None):
    """
    Supprime les colonnes dont le pourcentage de valeurs manquantes dépasse le seuil.

    En mode transform :
        columns_to_drop fourni -> supprime les mêmes colonnes qu'en fit.
    """

    df_clean = df.copy()

    if columns_to_drop is not None:
        df_clean = df_clean.drop(columns=columns_to_drop, errors="ignore")
        return df_clean, columns_to_drop

    missing_pct = df_clean.isnull().mean() * 100
    dropped_columns = missing_pct[missing_pct > threshold].index.tolist()

    df_clean = df_clean.drop(columns=dropped_columns, errors="ignore")

    return df_clean, dropped_columns


def impute_numeric_columns(df, numeric_cols, medians=None):
    """
    Impute les colonnes numériques avec la médiane.
    Ajoute une colonne indicatrice *_missing.
    """

    df_clean = df.copy()
    computed_medians = {}

    for col in numeric_cols:

        if col not in df_clean.columns:
            continue

        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        if df_clean[col].isnull().sum() > 0:
            df_clean[f"{col}_missing"] = df_clean[col].isnull().astype(int)

        if medians is not None:
            median_value = medians.get(col, 0)
        else:
            median_value = df_clean[col].median()
            if pd.isna(median_value):
                median_value = 0

        computed_medians[col] = median_value
        df_clean[col] = df_clean[col].fillna(median_value)

    return df_clean, computed_medians


def impute_categorical_columns(df, categorical_cols, fill_values=None):
    """
    Impute les colonnes catégorielles avec 'unknown'.
    Ajoute une colonne indicatrice *_missing.
    """

    df_clean = df.copy()
    computed_fill = {}

    invalid_values = [
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "NaN",
        "None",
        "NULL",
        "NA",
        "N/A"
    ]

    for col in categorical_cols:

        if col not in df_clean.columns:
            continue

        df_clean[col] = df_clean[col].astype("object")
        df_clean[col] = df_clean[col].replace(invalid_values, pd.NA)

        if df_clean[col].isnull().sum() > 0:
            df_clean[f"{col}_missing"] = df_clean[col].isnull().astype(int)

        fill_val = "unknown" if fill_values is None else fill_values.get(col, "unknown")
        computed_fill[col] = fill_val

        df_clean[col] = df_clean[col].fillna(fill_val)

    return df_clean, computed_fill


# ============================================================
# 4. TRAITEMENT DES DATES
# ============================================================

def transform_date_columns(df, date_cols, date_medians=None):
    """
    Transforme les colonnes dates en variables exploitables :
    année, mois, jour, jour de la semaine.
    """

    df_clean = df.copy()
    computed_date_medians = {}

    for col in date_cols:

        if col not in df_clean.columns:
            continue

        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

        created_cols = {
            f"{col}_year": df_clean[col].dt.year,
            f"{col}_month": df_clean[col].dt.month,
            f"{col}_day": df_clean[col].dt.day,
            f"{col}_dayofweek": df_clean[col].dt.dayofweek,
        }

        for created_col, values in created_cols.items():
            df_clean[created_col] = values

            if date_medians is not None:
                median_val = date_medians.get(created_col, 0)
            else:
                median_val = df_clean[created_col].median()
                if pd.isna(median_val):
                    median_val = 0

            computed_date_medians[created_col] = median_val
            df_clean[created_col] = df_clean[created_col].fillna(median_val)

        df_clean = df_clean.drop(columns=[col], errors="ignore")

    return df_clean, computed_date_medians


# ============================================================
# 5. SKEWNESS
# ============================================================

def analyze_skewness(df, numeric_cols):
    """
    Calcule la skewness des colonnes numériques.
    """

    report = []

    for col in numeric_cols:

        if col not in df.columns:
            continue

        try:
            skew_value = df[col].skew()

            report.append({
                "column": col,
                "skewness": round(skew_value, 3),
                "abs_skewness": round(abs(skew_value), 3)
            })

        except Exception:
            continue

    skewness_report = pd.DataFrame(report)

    if not skewness_report.empty:
        skewness_report = skewness_report.sort_values(
            by="abs_skewness",
            ascending=False
        ).reset_index(drop=True)

    return skewness_report


def transform_skewed_features(
    df,
    numeric_cols,
    skew_threshold=1,
    skew_transformers=None
):
    """
    Applique Log1p ou Yeo-Johnson aux variables asymétriques.
    """

    df_clean = df.copy()
    transformation_report = []
    computed_transformers = {}

    if skew_transformers is not None:

        for col, info in skew_transformers.items():

            if col not in df_clean.columns:
                continue

            method = info.get("method")
            transformer = info.get("transformer")

            if method == "Log1p":
                df_clean[col] = np.log1p(df_clean[col].clip(lower=0))

            elif method == "Yeo-Johnson" and transformer is not None:
                df_clean[col] = transformer.transform(df_clean[[col]]).flatten()

        return df_clean, skew_transformers, pd.DataFrame()

    for col in numeric_cols:

        if col not in df_clean.columns:
            continue

        try:
            skew_before = abs(df_clean[col].skew())

            if pd.isna(skew_before) or skew_before < skew_threshold:
                continue

            if (df_clean[col] >= 0).all():
                transformed = np.log1p(df_clean[col])
                skew_after = abs(pd.Series(transformed).skew())

                if skew_after < skew_before:
                    df_clean[col] = transformed
                    computed_transformers[col] = {
                        "method": "Log1p",
                        "transformer": None
                    }

                    transformation_report.append({
                        "column": col,
                        "method": "Log1p",
                        "skew_before": round(skew_before, 3),
                        "skew_after": round(skew_after, 3)
                    })

            else:
                pt = PowerTransformer(method="yeo-johnson")
                transformed = pt.fit_transform(df_clean[[col]]).flatten()

                skew_after = abs(pd.Series(transformed).skew())

                if skew_after < skew_before:
                    df_clean[col] = transformed
                    computed_transformers[col] = {
                        "method": "Yeo-Johnson",
                        "transformer": pt
                    }

                    transformation_report.append({
                        "column": col,
                        "method": "Yeo-Johnson",
                        "skew_before": round(skew_before, 3),
                        "skew_after": round(skew_after, 3)
                    })

        except Exception:
            continue

    return df_clean, computed_transformers, pd.DataFrame(transformation_report)


# ============================================================
# 6. OUTLIERS
# ============================================================

def detect_outliers_iqr(df, numeric_cols):
    """
    Détecte les outliers avec la méthode IQR.
    """

    report = []

    for col in numeric_cols:

        if col not in df.columns:
            continue

        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0 or pd.isna(IQR):
                continue

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = len(
                df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            )

            report.append({
                "column": col,
                "lower_bound": round(lower_bound, 3),
                "upper_bound": round(upper_bound, 3),
                "outlier_count": outlier_count,
                "outlier_ratio": round((outlier_count / len(df)) * 100, 2)
            })

        except Exception:
            continue

    return pd.DataFrame(report)


def treat_outliers(df, numeric_cols, iqr_bounds=None):
    """
    Traitement automatique des outliers :
    - < 5%  : aucune action
    - 5-15% : winsorisation
    - > 15% : signalement uniquement
    """

    df_clean = df.copy()
    report = []
    computed_bounds = {}

    for col in numeric_cols:

        if col not in df_clean.columns:
            continue

        try:
            if iqr_bounds is not None:
                bounds = iqr_bounds.get(col)

                if bounds is None:
                    continue

                lower = bounds["lower"]
                upper = bounds["upper"]
                action = bounds["action"]

                if action == "Winsorized":
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

                continue

            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0 or pd.isna(IQR):
                computed_bounds[col] = {
                    "lower": float(Q1),
                    "upper": float(Q3),
                    "action": "None"
                }

                report.append({
                    "column": col,
                    "outlier_ratio": 0,
                    "action": "None"
                })

                continue

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outlier_ratio = (
                ((df_clean[col] < lower) | (df_clean[col] > upper)).mean() * 100
            )

            action = "None"

            if 5 <= outlier_ratio <= 15:
                df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                action = "Winsorized"

            elif outlier_ratio > 15:
                action = "Flagged only - delegated to Isolation Forest"

            computed_bounds[col] = {
                "lower": float(lower),
                "upper": float(upper),
                "action": action
            }

            report.append({
                "column": col,
                "outlier_ratio": round(outlier_ratio, 2),
                "action": action
            })

        except Exception:
            continue

    return df_clean, computed_bounds, pd.DataFrame(report)


# ============================================================
# 7. NETTOYAGE GENERIQUE DES VARIABLES CATEGORIELLES
# ============================================================

def normalize_text_value(value):
    """
    Normalise une valeur catégorielle.
    """

    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))

    value = re.sub(r"[^a-zA-Z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    if value in ["", "nan", "none", "null", "na", "n/a"]:
        return "unknown"

    return value


def similarity_score(value1, value2):
    """
    Calcule une similarité simple entre deux chaînes.
    """

    return SequenceMatcher(None, value1, value2).ratio()


def standardize_similar_categories(
    series,
    min_frequency=3,
    similarity_threshold=0.90,
    reference_values=None
):
    """
    Standardise les valeurs catégorielles similaires.
    """

    series_clean = series.apply(normalize_text_value)

    if reference_values is None:
        value_counts = series_clean.value_counts()

        reference_values = value_counts[
            value_counts >= min_frequency
        ].index.tolist()

    corrected_series = series_clean.copy()
    corrections_report = []

    for value in series_clean.unique():

        if value in reference_values:
            continue

        best_match = None
        best_score = 0

        for ref_value in reference_values:
            score = similarity_score(value, ref_value)

            if score > best_score:
                best_score = score
                best_match = ref_value

        if best_match is not None and best_score >= similarity_threshold:
            corrected_series = corrected_series.replace(value, best_match)

            corrections_report.append({
                "original_value": value,
                "corrected_value": best_match,
                "similarity_score": round(best_score, 3)
            })

    return corrected_series, corrections_report, reference_values


def clean_categorical_variants_generic(
    df,
    categorical_cols,
    min_frequency=3,
    similarity_threshold=0.90,
    reference_values_map=None
):
    """
    Nettoie les variantes d'écriture dans les colonnes catégorielles.
    """

    df_clean = df.copy()
    global_report = {}
    computed_references = {}

    for col in categorical_cols:

        if col not in df_clean.columns:
            continue

        if reference_values_map is None:
            ref_values = None
        else:
            ref_values = reference_values_map.get(col)

        corrected_col, corrections, ref_values_used = standardize_similar_categories(
            df_clean[col],
            min_frequency=min_frequency,
            similarity_threshold=similarity_threshold,
            reference_values=ref_values
        )

        df_clean[col] = corrected_col
        global_report[col] = corrections
        computed_references[col] = ref_values_used

    return df_clean, global_report, computed_references


# ============================================================
# 8. ENCODAGE
# ============================================================

def encode_categorical_columns(df, categorical_cols, encoders=None):
    """
    Encode les variables catégorielles avec LabelEncoder.
    """

    df_encoded = df.copy()
    computed_encoders = {}

    for col in categorical_cols:

        if col not in df_encoded.columns:
            continue

        if encoders is not None:
            le = encoders.get(col)

            if le is None:
                continue

            known_classes = set(le.classes_)

            df_encoded[col] = df_encoded[col].astype(str).apply(
                lambda x: x if x in known_classes else "unknown"
            )

            if "unknown" not in known_classes:
                le.classes_ = np.append(le.classes_, "unknown")

            df_encoded[col] = le.transform(df_encoded[col].astype(str))
            computed_encoders[col] = le

        else:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            computed_encoders[col] = le

    return df_encoded, computed_encoders


# ============================================================
# 9. SCALING
# ============================================================

def choose_scaler(outlier_report):
    """
    Choisit automatiquement le scaler :
    - RobustScaler si ratio moyen d'outliers >= 5%
    - StandardScaler sinon
    """

    if outlier_report is None or outlier_report.empty:
        return StandardScaler(), "StandardScaler"

    avg_outlier_ratio = outlier_report["outlier_ratio"].mean()

    if pd.isna(avg_outlier_ratio):
        return StandardScaler(), "StandardScaler"

    if avg_outlier_ratio >= 5:
        return RobustScaler(), "RobustScaler"

    return StandardScaler(), "StandardScaler"


def scale_numeric_features(df, numeric_cols, scaler, mode="fit"):
    """
    Applique le scaler aux colonnes numériques.
    """

    df_scaled = df.copy()

    cols_present = [
        col for col in numeric_cols
        if col in df_scaled.columns
    ]

    if len(cols_present) == 0:
        return df_scaled, scaler

    if mode == "fit":
        df_scaled[cols_present] = scaler.fit_transform(df_scaled[cols_present])
    else:
        df_scaled[cols_present] = scaler.transform(df_scaled[cols_present])

    return df_scaled, scaler


# ============================================================
# 10. SELECTION DES FEATURES
# ============================================================

def apply_variance_threshold(
    df,
    numeric_cols,
    threshold=0.0,
    selected_cols=None
):
    """
    Supprime les colonnes à faible variance.
    """

    df_filtered = df.copy()

    cols_present = [
        col for col in numeric_cols
        if col in df_filtered.columns
    ]

    if len(cols_present) == 0:
        return df_filtered, [], []

    if selected_cols is not None:
        cols_to_keep = [
            col for col in selected_cols
            if col in df_filtered.columns
        ]

        removed = [
            col for col in cols_present
            if col not in cols_to_keep
        ]

        df_filtered = df_filtered.drop(columns=removed, errors="ignore")

        return df_filtered, cols_to_keep, removed

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df_filtered[cols_present])

    selected = [
        col for col, keep in zip(cols_present, selector.get_support())
        if keep
    ]

    removed = [
        col for col in cols_present
        if col not in selected
    ]

    df_filtered = df_filtered.drop(columns=removed, errors="ignore")

    return df_filtered, selected, removed


def detect_correlated_features(
    df,
    numeric_cols,
    threshold=0.90,
    cols_to_drop=None
):
    """
    Détecte ou rejoue la suppression des features corrélées.
    """

    cols_present = [
        col for col in numeric_cols
        if col in df.columns
    ]

    if cols_to_drop is not None:
        return cols_to_drop, pd.DataFrame()

    if len(cols_present) <= 1:
        return [], pd.DataFrame()

    corr_matrix = df[cols_present].corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    computed_cols_to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    return computed_cols_to_drop, corr_matrix


def remove_correlated_features(df, cols_to_drop):
    """
    Supprime les colonnes corrélées.
    """

    return df.drop(columns=cols_to_drop, errors="ignore")


# ============================================================
# 11. SAUVEGARDE / CHARGEMENT DES ARTEFACTS
# ============================================================

def save_object(obj, path):
    """
    Sauvegarde un objet avec joblib.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(obj, path)


def load_object(path):
    """
    Charge un objet sauvegardé avec joblib.
    """

    return joblib.load(path)


# ============================================================
# 12. PIPELINE COMPLET - MODE FIT
# ============================================================

def fit_pipeline(
    df,
    structure,
    missing_threshold=60,
    skew_threshold=1,
    corr_threshold=0.90,
    variance_threshold=0.0,
    categorical_min_frequency=3,
    categorical_similarity_threshold=0.90,
    manual_features_to_keep=None,
    allow_high_risk_features=False,
    artifacts_dir="models"
):
    """
    Apprend le preprocessing sur les données d'entraînement.
    """

    reports = {}

    # 1. Pré-nettoyage
    df_clean = basic_cleaning(df)

    # 2. Validation des features réintégrées manuellement
    manual_warnings = []

    if manual_features_to_keep is not None:
        manual_features_to_keep, manual_warnings = validate_manual_features(
            manual_features_to_keep=manual_features_to_keep,
            structure=structure,
            allow_high_risk_features=allow_high_risk_features
        )

    reports["manual_feature_warnings"] = manual_warnings

    # 3. Séparation X / y
    X, y, label_col, excluded_columns, reintegrated_columns = split_features_label(
        df_clean,
        structure,
        manual_features_to_keep=manual_features_to_keep
    )

    # 4. Colonnes utilisables après réintégration
    numeric_cols, categorical_cols, date_cols = get_usable_columns_by_type(
        structure=structure,
        X=X,
        excluded_columns=excluded_columns
    )

    # 5. Transformation des dates
    X, date_medians = transform_date_columns(X, date_cols)

    generated_date_cols = [
        col for col in X.columns
        if any(col.endswith(suffix) for suffix in [
            "_year",
            "_month",
            "_day",
            "_dayofweek"
        ])
    ]

    numeric_cols = list(set(numeric_cols + generated_date_cols))

    # 6. Valeurs manquantes
    reports["missing_values"] = detect_missing_values(X)

    X, dropped_high_missing = drop_high_missing_columns(
        X,
        threshold=missing_threshold
    )

    numeric_cols = [
        col for col in numeric_cols
        if col in X.columns
    ]

    categorical_cols = [
        col for col in categorical_cols
        if col in X.columns
    ]

    reports["dropped_high_missing_columns"] = dropped_high_missing

    X, numeric_medians = impute_numeric_columns(X, numeric_cols)
    X, categorical_fill = impute_categorical_columns(X, categorical_cols)

    # 7. Nettoyage des variantes catégorielles
    X, cat_report, reference_values_map = clean_categorical_variants_generic(
        X,
        categorical_cols,
        min_frequency=categorical_min_frequency,
        similarity_threshold=categorical_similarity_threshold
    )

    reports["categorical_variants_cleaning"] = cat_report

    missing_indicator_cols = [
        col for col in X.columns
        if col.endswith("_missing")
    ]

    numeric_cols = list(set(numeric_cols + missing_indicator_cols))

    # 8. Skewness
    reports["skewness"] = analyze_skewness(X, numeric_cols)

    X, skew_transformers, skew_report = transform_skewed_features(
        X,
        numeric_cols,
        skew_threshold=skew_threshold
    )

    reports["skew_transformations"] = skew_report

    # 9. Outliers
    reports["outliers_detected"] = detect_outliers_iqr(X, numeric_cols)

    X, iqr_bounds, outlier_report = treat_outliers(X, numeric_cols)

    reports["outlier_treatment"] = outlier_report

    # 10. Encodage catégoriel
    X, encoders = encode_categorical_columns(X, categorical_cols)

    numeric_cols = [
        col for col in X.columns
        if pd.api.types.is_numeric_dtype(X[col])
    ]

    # 11. Scaling
    outlier_df = outlier_report

    if outlier_df.empty:
        outlier_df = pd.DataFrame([{"outlier_ratio": 0}])

    scaler, scaler_name = choose_scaler(outlier_df)

    numeric_cols_at_scaling = [
        col for col in numeric_cols
        if col in X.columns
    ]

    X, scaler = scale_numeric_features(
        X,
        numeric_cols_at_scaling,
        scaler,
        mode="fit"
    )

    # 12. Variance threshold
    X, selected_cols, removed_low_var = apply_variance_threshold(
        X,
        numeric_cols,
        threshold=variance_threshold
    )

    numeric_cols = [
        col for col in numeric_cols
        if col in X.columns
    ]

    reports["removed_low_variance"] = removed_low_var

    # 13. Corrélation
    cols_to_drop, corr_matrix = detect_correlated_features(
        X,
        numeric_cols,
        threshold=corr_threshold
    )

    X = remove_correlated_features(X, cols_to_drop)

    numeric_cols = [
        col for col in numeric_cols
        if col in X.columns
    ]

    reports["removed_correlated_features"] = cols_to_drop
    reports["correlation_matrix"] = corr_matrix

    # 14. Sauvegarde des artefacts
    artifacts = {
        "structure": structure,
        "dropped_high_missing": dropped_high_missing,
        "date_medians": date_medians,
        "numeric_medians": numeric_medians,
        "categorical_fill": categorical_fill,
        "reference_values_map": reference_values_map,
        "skew_transformers": skew_transformers,
        "iqr_bounds": iqr_bounds,
        "encoders": encoders,
        "scaler": scaler,
        "scaler_name": scaler_name,
        "selected_cols_after_variance": selected_cols,
        "cols_to_drop_corr": cols_to_drop,
        "final_features": X.columns.tolist(),
        "numeric_cols_at_scaling": numeric_cols_at_scaling,
        "numeric_cols_final": numeric_cols,
        "categorical_cols": categorical_cols,
        "excluded_columns": excluded_columns,
        "reintegrated_columns": reintegrated_columns,
        "manual_feature_warnings": manual_warnings,
        "allow_high_risk_features": allow_high_risk_features,
        "label_col": label_col,
        "missing_threshold": missing_threshold,
        "skew_threshold": skew_threshold,
        "corr_threshold": corr_threshold,
        "variance_threshold": variance_threshold,
    }

    os.makedirs(artifacts_dir, exist_ok=True)

    artifacts_path = os.path.join(
        artifacts_dir,
        "pipeline_artifacts.pkl"
    )

    save_object(artifacts, artifacts_path)

    preprocessing_info = {
        "initial_shape": df.shape,
        "final_shape": X.shape,
        "label_column": label_col,
        "excluded_columns": excluded_columns,
        "reintegrated_columns": reintegrated_columns,
        "manual_feature_warnings": manual_warnings,
        "allow_high_risk_features": allow_high_risk_features,
        "final_features": X.columns.tolist(),
        "scaler_used": scaler_name,
        "artifacts_saved_at": artifacts_path
    }

    print(f"fit_pipeline termine : {df.shape} -> {X.shape}")
    print(f"Artefacts sauvegardes : {artifacts_path}")

    return X, y, preprocessing_info, reports


# ============================================================
# 13. PIPELINE COMPLET - MODE TRANSFORM
# ============================================================

def transform_pipeline(df, artifacts_path="models/pipeline_artifacts.pkl"):
    """
    Rejoue exactement le même preprocessing sur de nouvelles données.
    """

    artifacts = load_object(artifacts_path)

    structure = artifacts["structure"]
    dropped_high_missing = artifacts["dropped_high_missing"]
    date_medians = artifacts["date_medians"]
    numeric_medians = artifacts["numeric_medians"]
    categorical_fill = artifacts["categorical_fill"]
    reference_values_map = artifacts["reference_values_map"]
    skew_transformers = artifacts["skew_transformers"]
    iqr_bounds = artifacts["iqr_bounds"]
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]
    selected_cols = artifacts["selected_cols_after_variance"]
    cols_to_drop_corr = artifacts["cols_to_drop_corr"]
    categorical_cols = artifacts["categorical_cols"]
    excluded_columns = artifacts["excluded_columns"]

    df_clean = basic_cleaning(df)

    X = df_clean.drop(columns=excluded_columns, errors="ignore")

    date_cols = structure.get("date_columns", [])

    numeric_cols = [
        col for col in structure.get("numeric_columns", [])
        if col in X.columns
    ]

    X, _ = transform_date_columns(
        X,
        date_cols,
        date_medians=date_medians
    )

    generated_date_cols = [
        col for col in X.columns
        if any(col.endswith(suffix) for suffix in [
            "_year",
            "_month",
            "_day",
            "_dayofweek"
        ])
    ]

    numeric_cols = list(set(numeric_cols + generated_date_cols))

    X, _ = drop_high_missing_columns(
        X,
        columns_to_drop=dropped_high_missing
    )

    numeric_cols = [
        col for col in numeric_cols
        if col in X.columns
    ]

    categorical_cols_present = [
        col for col in categorical_cols
        if col in X.columns
    ]

    X, _ = impute_numeric_columns(
        X,
        numeric_cols,
        medians=numeric_medians
    )

    X, _ = impute_categorical_columns(
        X,
        categorical_cols_present,
        fill_values=categorical_fill
    )

    X, _, _ = clean_categorical_variants_generic(
        X,
        categorical_cols_present,
        reference_values_map=reference_values_map
    )

    missing_indicator_cols = [
        col for col in X.columns
        if col.endswith("_missing")
    ]

    numeric_cols = list(set(numeric_cols + missing_indicator_cols))

    X, _, _ = transform_skewed_features(
        X,
        numeric_cols,
        skew_transformers=skew_transformers
    )

    X, _, _ = treat_outliers(
        X,
        numeric_cols,
        iqr_bounds=iqr_bounds
    )

    X, _ = encode_categorical_columns(
        X,
        categorical_cols_present,
        encoders=encoders
    )

    numeric_cols = [
        col for col in X.columns
        if pd.api.types.is_numeric_dtype(X[col])
    ]

    scale_cols = artifacts["numeric_cols_at_scaling"]

    for col in scale_cols:
        if col not in X.columns:
            X[col] = 0

    X, _ = scale_numeric_features(
        X,
        scale_cols,
        scaler,
        mode="transform"
    )

    X, _, _ = apply_variance_threshold(
        X,
        numeric_cols,
        selected_cols=selected_cols
    )

    X = remove_correlated_features(X, cols_to_drop_corr)

    final_features = artifacts["final_features"]

    for col in final_features:
        if col not in X.columns:
            X[col] = 0

    X = X[final_features]

    print(f"transform_pipeline termine : {df.shape} -> {X.shape}")

    return X


# ============================================================
# 14. POINT D'ENTREE SIMPLIFIE
# ============================================================

def run_preprocessing_pipeline(
    df,
    structure=None,
    mode="fit",
    artifacts_path="models/pipeline_artifacts.pkl",
    **kwargs
):
    """
    Point d'entrée unifié pour le pipeline.

    mode='fit'       -> fit_pipeline
    mode='transform' -> transform_pipeline
    """

    if mode == "fit":

        if structure is None:
            raise ValueError("structure est obligatoire en mode fit.")

        return fit_pipeline(df, structure, **kwargs)

    elif mode == "transform":

        return transform_pipeline(
            df,
            artifacts_path=artifacts_path
        )

    else:
        raise ValueError(
            f"Mode non reconnu : '{mode}'. Utilisez 'fit' ou 'transform'."
        )