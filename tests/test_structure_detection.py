import pandas as pd

from src.structure_detection import detect_structure, generate_structure_summary


def test_detect_structure():
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "age": [25, 30, 35, 40],
        "country": ["Tunisie", "France", "Tunisie", "Maroc"],
        "signup_date": ["2024-01-01", "2024-01-05", "2024-02-01", "2024-02-10"],
        "churn": [0, 1, 0, 1]
    })

    structure = detect_structure(df)

    assert "user_id" in structure["id_columns"]
    assert "age" in structure["numeric_columns"]
    assert "country" in structure["categorical_columns"]
    assert "signup_date" in structure["date_columns"]
    assert "churn" in structure["possible_label_columns"]

    assert "user_id" in structure["excluded_columns"]
    assert "churn" in structure["excluded_columns"]
    assert "age" in structure["feature_columns"]


def test_generate_structure_summary():
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "age": [25, 30, 35, 40],
        "country": ["Tunisie", "France", "Tunisie", "Maroc"],
        "signup_date": ["2024-01-01", "2024-01-05", "2024-02-01", "2024-02-10"],
        "churn": [0, 1, 0, 1]
    })

    structure = detect_structure(df)
    summary = generate_structure_summary(structure)

    assert summary["nombre_lignes"] == 4
    assert summary["nombre_colonnes"] == 5
    assert summary["nb_identifiants"] >= 1
    assert summary["nb_labels_potentiels"] >= 1