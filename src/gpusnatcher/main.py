import argparse
import multiprocessing
import secrets
import time
from typing import Any

import torch

from gpusnatcher.configs import ConfigData, ConfigManager
from gpusnatcher.emails import EmailManager
from gpusnatcher.gpu import GPUManager
from gpusnatcher.logger import console
from gpusnatcher.utils import compute_storage_size, countdown_timer


def set_args() -> argparse.Namespace:
    """Set command line arguments."""
    parser = argparse.ArgumentParser(description="GPU Snatcher")
    parser.add_argument("-c", "--config", default=None, type=str, help="Path to config file")
    return parser.parse_args()


def worker(idx: int, free_mem: int, ready_event: Any) -> None:
    """Worker function to process GPU tasks."""
    try:
        size = compute_storage_size(free_mem, dtype="float32", len_shape=3)
        tmp = torch.zeros(size, dtype=torch.float32, device=f"cuda:{idx}")
        ready_event.set()
    except Exception as e:
        console.print(f"[red]Failed to allocate memory on GPU {idx}: {e}[/red]")
        return

    while True:
        tmp.mul_(tmp)
        if secrets.randbelow(100) < 50:
            time.sleep(1)


def main() -> None:
    """The main entry point."""
    args = set_args()

    config_manager = ConfigManager(config_path=args.config)
    config_manager.load_or_create()
    config_manager.confirm_config()

    config: ConfigData = config_manager.config

    gpu_manager = GPUManager(
        num_gpus=config.gpu_nums,
        gpu_free_memory_ratio_threshold=config.gpu_free_memory_ratio_threshold,
    )

    email_manager = EmailManager(
        host_server=config.email_host,
        user=config.email_user,
        pwd=config.email_pwd,
        sender=config.email_sender,
        receivers=config.email_receivers,
    )

    processes = []

    try:
        while True:
            if gpu_manager.num_snatched_gpus >= gpu_manager.num_gpus:
                break

            num_gpus_needed = gpu_manager.get_num_gpus_needed()
            free_gpus_needed = gpu_manager.get_free_gpus()[:num_gpus_needed]

            if not free_gpus_needed:
                continue

            for gpu in free_gpus_needed:
                ready_event = multiprocessing.Event()
                p = multiprocessing.Process(
                    target=worker,
                    args=(
                        gpu["index"],
                        gpu["memory.free"],
                        ready_event,
                    ),
                )
                p.start()

                if ready_event.wait(timeout=60):
                    processes.append(p)
                    console.print(f"Started GPU worker for GPU {gpu['index']}")
                else:
                    console.print(f"[red]GPU worker for GPU {gpu['index']} failed to start.[/red]")

            processes = [p for p in processes if p.is_alive()]
            gpu_manager.set_num_snatched_gpus(len(processes))

            email_manager.send_email(
                subject=f"GPUSnatcher: Snatched GPU {[gpu['index'] for gpu in free_gpus_needed]}",
                body=f"Successfully snatched GPU {gpu['index']}. "
                f"Total: {gpu_manager.num_snatched_gpus}/{gpu_manager.num_gpus}",
            )
            console.print(f"Sent email notification for GPU {[gpu['index'] for gpu in free_gpus_needed]}")

        email_manager.send_email(
            subject=f"GPUSnatcher: Snatched {gpu_manager.num_snatched_gpus}/{gpu_manager.num_gpus} GPUs",
            body=(
                f"Currently snatched: {gpu_manager.num_snatched_gpus}/{gpu_manager.num_gpus}.\n"
                f"These GPUs will be released automatically after {config.gpu_times_min} minutes."
            ),
        )
        console.print(
            f"Sent final email notification. Snatched {gpu_manager.num_snatched_gpus}/{gpu_manager.num_gpus} GPUs, "
            f"releasing in {config.gpu_times_min} minutes."
        )

        countdown_timer(config.gpu_times_min, description="Releasing GPUs...")

    finally:
        console.print("[red]Cleaning up GPU workers...[/red]")
        for p in processes:
            p.terminate()
            p.join()
        console.print("[green]All GPU workers terminated. Exiting.[/green]")


if __name__ == "__main__":
    main()
