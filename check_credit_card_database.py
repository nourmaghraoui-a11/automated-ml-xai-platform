# ============================================================
# check_credit_card_database.py
# Vérification de la base SQLite de test
# ============================================================

import sqlite3

import pandas as pd


DATABASE_PATH = "data/raw/credit_card_test.db"
TABLE_NAME = "transactions"


def check_database():
    """
    Affiche le nombre de lignes et un aperçu de la table.
    """

    connection = sqlite3.connect(DATABASE_PATH)

    count_dataframe = pd.read_sql_query(
        f"""
        SELECT COUNT(*) AS total_rows
        FROM {TABLE_NAME}
        """,
        connection
    )

    preview_dataframe = pd.read_sql_query(
        f"""
        SELECT *
        FROM {TABLE_NAME}
        ORDER BY transaction_id DESC
        LIMIT 10
        """,
        connection
    )

    columns_dataframe = pd.read_sql_query(
        f"""
        PRAGMA table_info({TABLE_NAME})
        """,
        connection
    )

    connection.close()

    print("===== NOMBRE DE LIGNES =====")
    print(count_dataframe.to_string(index=False))

    print("\n===== COLONNES =====")
    print(
        columns_dataframe[
            ["name", "type"]
        ].to_string(index=False)
    )

    print("\n===== DERNIÈRES TRANSACTIONS =====")
    print(preview_dataframe.to_string(index=False))


if __name__ == "__main__":
    check_database()