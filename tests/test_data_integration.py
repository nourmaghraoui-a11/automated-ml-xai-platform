import pandas as pd

from src.data_integration import (
    standardize_column_names,
    apply_column_mapping,
    cast_column_types,
    aggregate_event_table,
    build_unified_dataset
)


def test_standardize_column_names():
    df = pd.DataFrame({
        "User ID": [1, 2],
        "Signup Date": ["2024-01-01", "2024-01-02"]
    })

    result = standardize_column_names(df)

    assert "user_id" in result.columns
    assert "signup_date" in result.columns


def test_apply_column_mapping():
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "montant": [100, 200]
    })

    mapping = {
        "customer_id": "user_id",
        "montant": "amount"
    }

    result = apply_column_mapping(df, mapping)

    assert "user_id" in result.columns
    assert "amount" in result.columns


def test_cast_column_types():
    df = pd.DataFrame({
        "user_id": [1, 2],
        "amount": ["100.5", "200.0"],
        "transaction_date": ["2024-01-01", "2024-01-02"]
    })

    schema = {
        "user_id": "str",
        "amount": "float",
        "transaction_date": "datetime"
    }

    result = cast_column_types(df, schema)

    assert pd.api.types.is_string_dtype(result["user_id"])
    assert result["amount"].dtype == "float64"
    assert pd.api.types.is_datetime64_any_dtype(result["transaction_date"])


def test_aggregate_event_table():
    transactions = pd.DataFrame({
        "transaction_id": [101, 102, 103],
        "user_id": [1, 1, 2],
        "amount": [100, 200, 50]
    })

    result = aggregate_event_table(
        transactions,
        group_key="user_id",
        aggregations={
            "amount": ["sum", "mean"],
            "transaction_id": ["count"]
        }
    )

    user_1 = result[result["user_id"] == 1].iloc[0]

    assert user_1["amount_sum"] == 300
    assert user_1["amount_mean"] == 150
    assert user_1["transaction_id_count"] == 2


def test_build_unified_dataset_multi_source():
    users = pd.DataFrame({
        "user_id": [1, 2],
        "age": [25, 30]
    })

    transactions = pd.DataFrame({
        "transaction_id": [101, 102, 103],
        "user_id": [1, 1, 2],
        "amount": [100, 200, 50]
    })

    dataframes = {
        "users": users,
        "transactions": transactions
    }

    config = {
        "join_key": "user_id",
        "event_aggregations": {
            "transactions": {
                "group_key": "user_id",
                "aggregations": {
                    "amount": ["sum", "mean"],
                    "transaction_id": ["count"]
                }
            }
        }
    }

    result = build_unified_dataset(dataframes, config)

    assert result.shape[0] == 2
    assert "age" in result.columns
    assert "amount_sum" in result.columns
    assert "transaction_id_count" in result.columns