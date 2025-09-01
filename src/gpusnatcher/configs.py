import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import tomli_w
from rich.table import Table

from gpusnatcher.logger import console, prompt


@dataclass
class ConfigData:
    """Configuration data for GPU Snatcher."""

    gpu_nums: int
    gpu_times_min: int
    email_host: str
    email_user: str
    email_pwd: str
    email_sender: str
    email_receivers: list[str]


class ConfigManager:
    """Manage the configuration for GPU Snatcher."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the configuration manager."""
        if config_path is None:
            config_path = self.search_config_file()

        self.config_path = config_path if config_path else Path.home() / ".config" / "gpusnatcher" / "gpusnatcher.toml"
        self.config: ConfigData | None

    @property
    def config_data(self) -> ConfigData | None:
        """Get the current configuration data."""
        return self.config

    def load_or_create(self) -> ConfigData:
        """Load configuration if exists, otherwise create new one interactively."""
        if self.config_path.exists():
            self.config = self.load_config(self.config_path)
        else:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config = self.update_config()
            self.save_config(self.config_path)

    def confirm_config(self) -> None:
        """Confirm the current configuration."""
        while True:
            config, fields_list = self.pad_config()
            console.print(config)
            choice = console.input("Do you want to keep this configuration? (y/n): ").strip().lower()
            if choice in ["y", ""]:
                break
            user_input = console.input("Enter the index of key to update (comma-separated): ").strip()
            selection_indices = [int(x) for x in user_input.split(",") if x.isdigit()]

            update_keys = [fields_list[i] for i in selection_indices if 0 <= i < len(fields_list)]
            self.config = self.update_config(update_keys)

        self.save_config(self.config_path)

    def update_config(self, key: str | list[str] | None = None) -> ConfigData:
        """Update one or more fields interactively."""
        if not hasattr(self, "config"):
            if key:
                raise ValueError("Cannot update a single key when config is not initialized.")
            placeholders = {f.name: None for f in fields(ConfigData)}
            self.config = ConfigData(**placeholders)

        keys_to_update = key if key else [f.name for f in fields(ConfigData)]
        keys_to_update = [keys_to_update] if isinstance(keys_to_update, str) else keys_to_update

        for k in keys_to_update:
            current_value = getattr(self.config, k)
            if k in ["gpu_nums", "gpu_times_min"]:
                new_value = prompt.ask(f"Please enter the {k}: ", default=str(current_value or ""))
                setattr(self.config, k, int(new_value))
            elif k == "email_receivers":
                new_value = prompt.ask(
                    f"Please enter the {k} (comma-separated): ",
                    default=",".join(current_value or []),
                )
                setattr(self.config, k, [email.strip() for email in new_value.split(",") if email.strip()])
            elif k == "email_pwd":
                current_value = "*" * 8
                new_value = prompt.ask(f"Please enter the {k}: ", default=str(current_value or ""), password=True)
                setattr(self.config, k, new_value)
            else:
                new_value = prompt.ask(f"Please enter the {k}: ", default=str(current_value or ""))
                setattr(self.config, k, new_value)

        return self.config

    def search_config_file(self) -> Path | None:
        """Search for the configuration file."""
        # Search for configuration file in current directory, home directory, and ~/.config/gpusnatcher
        for root in [Path.cwd(), Path.home(), Path.home() / ".config" / "gpusnatcher"]:
            config_path = root / "gpusnatcher.toml"
            if config_path.exists():
                return config_path
        return None

    def load_config(self, cf_path: Path) -> ConfigData:
        """Load configuration from a file."""
        with open(cf_path, "rb") as f:
            data = tomllib.load(f)

        valid_keys = {f.name for f in fields(ConfigData)}

        return ConfigData(**{k: v for k, v in data.items() if k in valid_keys})

    def save_config(self, cf_path: Path) -> None:
        """Save configuration to a file."""
        with open(cf_path, "wb") as f:
            tomli_w.dump(asdict(self.config), f)

    def pad_config(self) -> tuple[Table, list[str]]:
        """Pad the configuration data with default values."""
        fields_list = [f.name for f in fields(ConfigData)]

        table = Table(title="Current Configuration")
        table.add_column("Index", style="cyan")
        table.add_column("Field", style="magenta")
        table.add_column("Current Value", style="green")

        for i, field_name in enumerate(fields_list):
            current_value = getattr(self.config, field_name)
            if field_name == "email_pwd":
                current_value = "*" * 8
            if isinstance(current_value, list):
                current_value = ", ".join(current_value)
            table.add_row(str(i), field_name, str(current_value))

        return table, fields_list
