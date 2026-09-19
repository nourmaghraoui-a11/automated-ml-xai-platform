# ============================================================
# add_pending_transactions.py
# Ajout des 50 transactions conservées dans SQLite
# ============================================================

import os
import sqlite3

import pandas as pd


DATABASE_PATH = "data/raw/credit_card_test.db"
PENDING_DATA_PATH = "data/raw/pending_50_transactions.csv"
TABLE_NAME = "transactions"


def add_pending_transactions():
    """
    Ajoute les 50 lignes conservées dans la base SQLite.

    Les transaction_id déjà présents sont ignorés pour éviter
    les doublons si le script est exécuté plusieurs fois.
    """

    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(
            f"Base SQLite introuvable : {DATABASE_PATH}"
        )

    if not os.path.exists(PENDING_DATA_PATH):
        raise FileNotFoundError(
            f"Fichier de nouvelles données introuvable : "
            f"{PENDING_DATA_PATH}"
        )

    pending_dataframe = pd.read_csv(
        PENDING_DATA_PATH
    )

    if pending_dataframe.empty:
        print("Aucune transaction à ajouter.")
        return

    connection = sqlite3.connect(DATABASE_PATH)

    before_count = connection.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME}"
    ).fetchone()[0]

    existing_ids = pd.read_sql_query(
        f"""
        SELECT transaction_id
        FROM {TABLE_NAME}
        """,
        connection
    )["transaction_id"].tolist()

    rows_to_insert = pending_dataframe[
        ~pending_dataframe["transaction_id"].isin(
            existing_ids
        )
    ].copy()

    if rows_to_insert.empty:
        connection.close()

        print(
            "Les 50 transactions ont déjà été ajoutées. "
            "Aucun doublon n'a été créé."
        )

        return

    rows_to_insert["ingested_at"] = (
        pd.Timestamp.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    rows_to_insert.to_sql(
        TABLE_NAME,
        connection,
        if_exists="append",
        index=False
    )

    connection.commit()

    after_count = connection.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME}"
    ).fetchone()[0]

    connection.close()

    inserted_count = after_count - before_count

    if before_count > 0:
        new_data_rate = (
            inserted_count
            / before_count
            * 100
        )
    else:
        new_data_rate = 100.0

    print("===== AJOUT TERMINÉ =====")
    print(f"Lignes avant ajout : {before_count}")
    print(f"Lignes insérées : {inserted_count}")
    print(f"Lignes après ajout : {after_count}")
    print(f"Taux des nouvelles données : {new_data_rate:.2f} %")

    if new_data_rate > 10:
        print(
            "Résultat attendu : le planificateur doit "
            "déclencher le pipeline."
        )
    else:
        print(
            "Résultat attendu : le pipeline ne doit pas "
            "être déclenché."
        )


if __name__ == "__main__":
    add_pending_transactions()