from dataclasses import asdict
from pathlib import Path

import pytest
import tomli_w

from gpusnatcher.configs import ConfigData


@pytest.fixture
def config_path(tmp_path: Path, config_data: ConfigData) -> Path:
    """Fixture to provide a temporary config file path."""
    config_path = tmp_path / "gpusnatcher.toml"

    with config_path.open("wb") as f:
        tomli_w.dump(asdict(config_data), f)

    return config_path


@pytest.fixture
def config_data() -> ConfigData:
    """Fixture to provide a temporary config data."""
    return ConfigData(
        gpu_nums=1,
        gpu_times_min=10,
        email_host="smtp.example.com",
        email_user="user@example.com",
        email_pwd="password",  # noqa: S106
        email_sender="sender@example.com",
        email_receivers=["receiver@example.com"],
    )
