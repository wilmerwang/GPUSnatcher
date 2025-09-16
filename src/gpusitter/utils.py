import getpass
import queue
import socket
import time
from contextlib import nullcontext

import psutil
from rich.live import Live
from rich.spinner import Spinner

from gpusitter.gpu import GPUManager
from gpusitter.logger import console


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


def countdown_timer(minutes: int | float, description: str = "Waiting", debug: bool = False) -> None:
    """Display a spinner with MM:SS countdown for the given minutes."""
    total_seconds = int(minutes * 60)

    context = nullcontext() if debug else Live(console=console, refresh_per_second=10)
    with context as live:
        for remaining in range(total_seconds, 0, -1):
            mins, secs = divmod(remaining, 60)

            text = f"{description} in {mins:02d}:{secs:02d}"
            if debug:
                console.log(f"[blue]Debug:[/blue] {text}")
            else:
                spinner = Spinner("dots", text=f"[green]{text}[/green]")
                live.update(spinner)
            time.sleep(1)

    console.log(f"\n[green]{description} completed![/green]")


class DummyStatus:
    """A dummy status context manager for debug mode."""

    def update(self, message: str) -> None:
        """Update the status message."""
        console.log(message)


def check_jobs(jobs: queue.Queue, gpu_manager: GPUManager) -> list | None:
    """Check the status of jobs in the queue and allocate GPUs as needed."""
    all_gpus = gpu_manager.get_all_gpus()

    failure_results = [job for job in list(jobs.queue) if job.required_gpus > len(all_gpus)]

    return failure_results if failure_results else None


def get_server_info() -> tuple[str, str | None, str]:
    """Get server information including hostname and GPU details."""
    hostname = socket.gethostname()
    username = getpass.getuser()

    iface = "ppp0"
    ip = None
    if iface in psutil.net_if_addrs():
        for snic in psutil.net_if_addrs()[iface]:
            if snic.family == 2:  # AF_INET
                ip = snic.address

    return hostname, ip, username
