import logging

import pandas as pd

from src.scheduler_service import _check_new_data_job, _save_state, _load_state


def test_check_new_data_job_initializes_reference(tmp_path, monkeypatch, caplog):
    config = {
        "source": {
            "type": "csv",
            "path": str(tmp_path / "data.csv")
        },
        "data_change_trigger": {
            "enabled": True,
            "threshold_pct": 10,
            "check_interval_minutes": 30
        },
        "state_path": str(tmp_path / "scheduler_state.json")
    }

    df = pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "amount": [10, 20, 30]
    })
    df.to_csv(config["source"]["path"], index=False)

    logger = logging.getLogger("test_scheduler_service_initial")
    logger.setLevel(logging.INFO)
    _check_new_data_job(config, logger=logger)

    state = _load_state(config["state_path"])

    assert state["last_row_count"] == 3
    assert "Aucune référence de volume précédente" in caplog.text


def test_check_new_data_job_triggers_when_threshold_exceeded(tmp_path, monkeypatch, caplog):
    config = {
        "source": {
            "type": "csv",
            "path": str(tmp_path / "data.csv")
        },
        "data_change_trigger": {
            "enabled": True,
            "threshold_pct": 10,
            "check_interval_minutes": 30
        },
        "state_path": str(tmp_path / "scheduler_state.json")
    }

    initial_df = pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "amount": [10, 20, 30]
    })
    initial_df.to_csv(config["source"]["path"], index=False)
    _save_state({"last_row_count": 3}, config["state_path"])

    updated_df = pd.DataFrame({
        "transaction_id": [1, 2, 3, 4],
        "amount": [10, 20, 30, 40]
    })
    updated_df.to_csv(config["source"]["path"], index=False)

    def fake_run_job(config_arg, reason, logger):
        state = _load_state(config_arg["state_path"])
        state["last_run_id"] = "triggered"
        state["last_row_count"] = 4
        _save_state(state, config_arg["state_path"])

    monkeypatch.setattr("src.scheduler_service._run_job", fake_run_job)

    logger = logging.getLogger("test_scheduler_service_trigger")
    logger.setLevel(logging.INFO)
    _check_new_data_job(config, logger=logger)

    state = _load_state(config["state_path"])

    assert state["last_run_id"] == "triggered"
    assert state["last_row_count"] == 4
    assert "Vérification volumétrique" in caplog.text
