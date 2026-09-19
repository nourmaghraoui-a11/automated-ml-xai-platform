import pandas as pd

from src.pipeline_controller import start_pipeline


def test_start_pipeline_single_source(tmp_path):
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, 30, None, 45, 50],
        "country": ["Tunisie", "tunisie", "France", "FRANCE", None],
        "signup_date": ["2024-01-01", "2024-01-10", "2024-02-01", "2024-03-01", None],
        "churn": [0, 1, 0, 1, 0]
    })

    result = start_pipeline(
        data=df,
        integration_config=None,
        manual_features_to_keep=None,
        models_base_dir=str(tmp_path / "models"),
        reports_base_dir=str(tmp_path / "reports")
    )

    assert "X_ready" in result
    assert "structure" in result
    assert "preprocessing_info" in result
    assert result["X_ready"].shape[0] == 5
    assert result["preprocessing_info"]["label_column"] == "churn"


def test_start_pipeline_multi_source(tmp_path):
    users = pd.DataFrame({
        "user_id": [1, 2, 3],
        "age": [25, 30, 40],
        "country": ["Tunisie", "France", "Maroc"],
        "churn": [0, 1, 0]
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

    integration_config = {
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

    result = start_pipeline(
        data=dataframes,
        integration_config=integration_config,
        manual_features_to_keep=None,
        models_base_dir=str(tmp_path / "models"),
        reports_base_dir=str(tmp_path / "reports")
    )

    assert result["dataset_final"].shape[0] == 3
    assert "amount_sum" in result["dataset_final"].columns
    assert "X_ready" in result
    assert result["X_ready"].shape[0] == 3