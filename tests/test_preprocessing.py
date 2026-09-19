import pandas as pd

from src.structure_detection import detect_structure
from src.preprocessing import run_preprocessing_pipeline, transform_pipeline


def test_run_preprocessing_pipeline_fit(tmp_path):
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, 30, None, 45, 50],
        "country": ["Tunisie", "tunisie", "France", "FRANCE", None],
        "signup_date": ["2024-01-01", "2024-01-10", "2024-02-01", "2024-03-01", None],
        "churn": [0, 1, 0, 1, 0]
    })

    structure = detect_structure(df)

    artifacts_dir = tmp_path / "models"

    X_ready, y, preprocessing_info, reports = run_preprocessing_pipeline(
        df,
        structure,
        mode="fit",
        artifacts_dir=str(artifacts_dir)
    )

    assert X_ready.shape[0] == 5
    assert y is not None
    assert preprocessing_info["label_column"] == "churn"
    assert "user_id" in preprocessing_info["excluded_columns"]
    assert "churn" in preprocessing_info["excluded_columns"]
    assert len(preprocessing_info["final_features"]) > 0


def test_transform_pipeline(tmp_path):
    df_train = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, 30, None, 45, 50],
        "country": ["Tunisie", "tunisie", "France", "FRANCE", None],
        "signup_date": ["2024-01-01", "2024-01-10", "2024-02-01", "2024-03-01", None],
        "churn": [0, 1, 0, 1, 0]
    })

    structure = detect_structure(df_train)

    artifacts_dir = tmp_path / "models"

    X_train, y, preprocessing_info, reports = run_preprocessing_pipeline(
        df_train,
        structure,
        mode="fit",
        artifacts_dir=str(artifacts_dir)
    )

    artifacts_path = preprocessing_info["artifacts_saved_at"]

    df_new = pd.DataFrame({
        "user_id": [6, 7],
        "age": [28, None],
        "country": ["Tunisie", "Italie"],
        "signup_date": ["2024-04-01", None],
        "churn": [0, 1]
    })

    X_new = transform_pipeline(
        df_new,
        artifacts_path=artifacts_path
    )

    assert X_new.shape[0] == 2
    assert list(X_new.columns) == list(X_train.columns)