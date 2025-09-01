import argparse
import multiprocessing

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


def worker(idx: int, free_mem: int) -> None:
    """Worker function to process GPU tasks."""
    size = compute_storage_size(free_mem, dtype="float32", len_shape=3)
    tmp = torch.zeros(size, dtype=torch.float32, device=f"cuda:{idx}")
    while True:
        torch.mul(tmp[0], tmp[-1])


def main() -> None:
    """The main entry point."""
    args = set_args()

    config_manager = ConfigManager(config_path=args.config)
    config_manager.load_or_create()
    config_manager.confirm_config()

    config: ConfigData = config_manager.config

    gpu_manager = GPUManager(num_gpus=config.gpu_nums)
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
                p = multiprocessing.Process(
                    target=worker,
                    args=(
                        gpu["index"],
                        gpu["memory.free"],
                    ),
                )
                p.start()
                processes.append(p)
                console.print(f"Started GPU worker for GPU {gpu['index']}")

            alive_gpus = [p.is_alive() for p in processes]
            gpu_manager.set_num_snatched_gpus(len(alive_gpus))

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
