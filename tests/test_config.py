from pathlib import Path

import pytest
from pytest_mock import MockerFixture
from rich.table import Table

from gpusnatcher.configs import ConfigData, ConfigManager


@pytest.fixture
def config_manager_default() -> ConfigManager:
    """Fixture to provide a ConfigManager instance with default config path."""
    return ConfigManager(config_path=None)


@pytest.fixture
def config_manager_with_path(config_path: Path) -> ConfigManager:
    """Fixture to provide a ConfigManager instance with a specific config path."""
    return ConfigManager(config_path=config_path)


def test_load_config(config_manager_with_path: ConfigManager) -> None:
    """Test loading configuration from a file."""
    config_manager_with_path.load_or_create()
    assert config_manager_with_path.config_data is not None
    assert isinstance(config_manager_with_path.config_data, ConfigData)


def test_save_config(config_manager_with_path: ConfigManager, tmp_path: Path) -> None:
    """Test saving configuration to a file."""
    config_manager_with_path.load_or_create()
    config_manager_with_path.save_config(tmp_path / "gpusnatcher.toml")
    assert (tmp_path / "gpusnatcher.toml").exists()


def test_pad_config(config_manager_with_path: ConfigManager) -> None:
    """Test the padding of configuration data."""
    config_manager_with_path.load_or_create()
    table, fields_list = config_manager_with_path.pad_config()
    assert isinstance(table, Table)
    assert isinstance(fields_list, list)
    assert len(fields_list) > 0
    assert all(isinstance(field, str) for field in fields_list)


def test_update_config(config_manager_with_path: ConfigManager, mocker: MockerFixture) -> None:
    """Test updating configuration interactively."""
    config_manager_with_path.load_or_create()
    mocker.patch(
        "gpusnatcher.configs.prompt.ask",
        side_effect=[
            "0.9",  # gpu_free_memory_ratio_threshold
            "5",  # friendly_min
            "smtp.new.com",  # email_host
            "newuser",  # email_user
            "newpwd",  # email_pwd
            "newsender",  # email_sender
            "a@x.com, b@y.com",  # email_receivers
        ],
    )

    updated_config = config_manager_with_path.update_config()
    assert updated_config is not None
    assert isinstance(updated_config, ConfigData)

    assert updated_config.gpu_free_memory_ratio_threshold == 0.9
    assert updated_config.email_host == "smtp.new.com"
    assert updated_config.email_user == "newuser"
    assert updated_config.email_pwd == "newpwd"  # noqa: S105
    assert updated_config.email_sender == "newsender"
    assert updated_config.email_receivers == ["a@x.com", "b@y.com"]


def test_confirm_config(config_manager_with_path: ConfigManager, mocker: MockerFixture) -> None:
    """Test confirming configuration interactively."""
    config_manager_with_path.load_or_create()
    mocker.patch("gpusnatcher.configs.console.input", return_value="y")
    mocker.patch("gpusnatcher.configs.ConfigManager.save_config", return_value=None)
    mocker.patch("gpusnatcher.configs.console.log", return_value=None)

    config_manager_with_path.confirm_config()
    assert config_manager_with_path.config is not None


def test_confirm_config_update_then_keep(config_manager_with_path: ConfigManager, mocker: MockerFixture) -> None:
    """Test confirming configuration updates and keeping them."""
    config_manager_with_path.load_or_create()
    inputs = iter(["n", "0", "y"])
    mocker.patch("gpusnatcher.configs.console.input", side_effect=lambda _: next(inputs))
    mocker.patch("gpusnatcher.configs.ConfigManager.save_config", return_value=None)
    mocker.patch("gpusnatcher.configs.console.log", return_value=None)

    def fake_update(keys: list[str] | str) -> ConfigData:
        cfg = config_manager_with_path.config
        for k in keys:
            if k == "gpu_free_memory_ratio_threshold":
                cfg.gpu_free_memory_ratio_threshold = 0.9
            elif k == "friendly_min":
                cfg.friendly_min = 5
        return cfg

    mocker.patch("gpusnatcher.configs.ConfigManager.update_config", side_effect=fake_update)

    config_manager_with_path.confirm_config()

    assert config_manager_with_path.config is not None
    assert config_manager_with_path.config.gpu_free_memory_ratio_threshold == 0.9
    assert config_manager_with_path.config.friendly_min == 5
