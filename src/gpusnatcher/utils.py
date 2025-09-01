import time

from rich.live import Live
from rich.spinner import Spinner

from gpusnatcher.logger import console


def compute_storage_size(memory: int, dtype: str = "float32", len_shape: int = 3) -> list[int]:
    """Compute the storage size required for a given GPU memory.

    Args:
        memory (int): The GPU memory size in MiB.
        dtype (str): float32 or float64
        len_shape (int): The length of the shape.

    Returns:
        int: The estimated storage size in bytes.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'.")

    bytes_per_element = 4 if dtype == "float32" else 8

    memory = memory * 0.9

    sz = pow(memory * 1024 * 1024 // bytes_per_element, 1 / len_shape)

    return [int(sz)] * len_shape


def countdown_timer(minutes: int, description: str = "Waiting") -> None:
    """Display a spinner with MM:SS countdown for the given minutes."""
    total_seconds = minutes * 60

    with Live(console=console, refresh_per_second=10) as live:
        for remaining in range(total_seconds, 0, -1):
            mins, secs = divmod(remaining, 60)
            spinner = Spinner("dots", text=f"{description} in {mins:02d}:{secs:02d}")
            live.update(spinner)
            time.sleep(1)

    console.print(f"\n[bold green]{description} completed![/bold green]")
