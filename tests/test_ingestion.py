import pandas as pd

from src.ingestion import load_csv, get_dataset_info


def test_load_csv(tmp_path):
    file_path = tmp_path / "users.csv"

    df = pd.DataFrame({
        "user_id": [1, 2],
        "age": [25, 30],
        "country": ["Tunisie", "France"]
    })

    df.to_csv(file_path, index=False)

    loaded_df = load_csv(str(file_path))

    assert loaded_df.shape == (2, 3)
    assert "user_id" in loaded_df.columns
    assert "age" in loaded_df.columns


def test_get_dataset_info():
    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "age": [25, None, 40]
    })

    info = get_dataset_info(df)

    assert info["n_rows"] == 3
    assert info["n_columns"] == 2
    assert info["missing_values_total"] == 1