# ============================================================
# prepare_huggingface_sqlite_test.py
# Téléchargement de 150 lignes depuis Hugging Face
# Création d'une base SQLite avec 100 lignes initiales
# Conservation de 50 lignes pour tester le planificateur
# ============================================================

import os
import sqlite3
from itertools import islice

import pandas as pd
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "MaxPrestige/credit-card-fraud-CLEAN"
DATABASE_PATH = "data/raw/credit_card_test.db"
PENDING_DATA_PATH = "data/raw/pending_50_transactions.csv"

TABLE_NAME = "transactions"

INITIAL_ROW_COUNT = 100
NEW_ROW_COUNT = 50
TOTAL_ROW_COUNT = INITIAL_ROW_COUNT + NEW_ROW_COUNT


# ============================================================
# UTILITAIRES
# ============================================================

def normalize_column_names(dataframe):
    """
    Standardise les noms des colonnes.
    """

    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in dataframe.columns
    ]

    return dataframe


def prepare_test_database():
    """
    Charge 150 lignes depuis Hugging Face :

    - 100 lignes sont insérées dans SQLite ;
    - 50 lignes sont sauvegardées dans un fichier CSV ;
    - ces 50 lignes seront ajoutées plus tard pour tester le scheduler.
    """

    os.makedirs("data/raw", exist_ok=True)

    print("Chargement du dataset Hugging Face...")
    print(f"Dataset : {DATASET_NAME}")

    # Streaming évite de télécharger immédiatement le dataset complet.
    streaming_dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    records = list(
        islice(
            streaming_dataset,
            TOTAL_ROW_COUNT
        )
    )

    if len(records) < TOTAL_ROW_COUNT:
        raise ValueError(
            f"Le dataset n'a retourné que {len(records)} lignes, "
            f"alors que {TOTAL_ROW_COUNT} sont nécessaires."
        )

    dataframe = pd.DataFrame(records)
    dataframe = normalize_column_names(dataframe)

    # Ajouter un identifiant stable pour éviter les doublons.
    dataframe.insert(
        0,
        "transaction_id",
        range(1, len(dataframe) + 1)
    )

    # Ajouter une colonne de date utile au planificateur.
    download_timestamp = pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    dataframe["ingested_at"] = download_timestamp

    initial_dataframe = dataframe.iloc[
        :INITIAL_ROW_COUNT
    ].copy()

    pending_dataframe = dataframe.iloc[
        INITIAL_ROW_COUNT:TOTAL_ROW_COUNT
    ].copy()

    # Supprimer l'ancienne base pour repartir d'un état propre.
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)

    connection = sqlite3.connect(DATABASE_PATH)

    initial_dataframe.to_sql(
        TABLE_NAME,
        connection,
        if_exists="replace",
        index=False
    )

    # Index utile pour éviter les doublons.
    connection.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{TABLE_NAME}_transaction_id
        ON {TABLE_NAME}(transaction_id)
    """)

    connection.commit()

    total_in_database = connection.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME}"
    ).fetchone()[0]

    connection.close()

    pending_dataframe.to_csv(
        PENDING_DATA_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n===== PRÉPARATION TERMINÉE =====")
    print(f"Base SQLite : {DATABASE_PATH}")
    print(f"Table : {TABLE_NAME}")
    print(f"Lignes initiales : {total_in_database}")
    print(f"Lignes conservées : {len(pending_dataframe)}")
    print(f"Fichier temporaire : {PENDING_DATA_PATH}")
    print(f"Colonnes : {list(dataframe.columns)}")

    print("\nScénario du test :")
    print("Référence : 100 lignes")
    print("Nouvelles données : 50 lignes")
    print("Taux de nouvelles données : 50 %")
    print("Résultat attendu : déclenchement du pipeline")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    prepare_test_database()