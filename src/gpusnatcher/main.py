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

        while True:
            tmp.mul_(tmp)
            if secrets.randbelow(100) < 50:
                time.sleep(1)

    except KeyboardInterrupt:
        return
    except Exception as e:
        console.log(f"[red]Failed to allocate memory on GPU {idx}: {e}[/red]")
        return


def notify_gpu_snatch(
    email_mgr: EmailManager,
    gpu_indices: list[int],
    snatched: int,
    total: int,
    final: bool = False,
    gpu_times_min: int = 1,
) -> None:
    """Notify about GPU snatching status."""
    if not gpu_indices:
        return

    gpu_list_str = ", ".join(str(i) for i in gpu_indices)

    if final:
        subject = f"GPUSnatcher: Snatched {snatched}/{total} GPUs"
        body = (
            f"Successfully snatched GPU [{gpu_list_str}]\n"
            f"These GPUs will be released automatically after {gpu_times_min} minutes."
        )
    else:
        subject = f"GPUSnatcher: Snatched GPU [{gpu_list_str}]"
        body = f"Successfully snatched GPU [{gpu_list_str}]. Now total: {snatched}/{total}"

    email_mgr.send_email(subject=subject, body=body)
    console.log(f"Sent {'final ' if final else ''}email notification for GPU [{gpu_list_str}]")


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
        with console.status("[green]Snatching GPUs...[/green]"):
            while gpu_manager.num_snatched_gpus < gpu_manager.num_gpus:
                num_gpus_needed = gpu_manager.get_num_gpus_needed()
                free_gpus_needed = gpu_manager.get_free_gpus()[:num_gpus_needed]

                if not free_gpus_needed:
                    continue

                countdown_timer(
                    config.friendly_min,
                    description=(
                        "Be friendly... "
                        "waiting before allocation to avoid OOM from previous job's final test/cleanup... "
                    ),
                )
                successful_gpus_index = []
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
                        successful_gpus_index.append(gpu["index"])
                        console.log(f"Started GPU worker for GPU {gpu['index']}")
                    else:
                        console.log(f"[red]GPU worker for GPU {gpu['index']} failed to start.[/red]")

                gpu_manager.snatched_gpus.extend(successful_gpus_index)

                notify_gpu_snatch(
                    email_manager, successful_gpus_index, gpu_manager.num_snatched_gpus, gpu_manager.num_gpus
                )

        notify_gpu_snatch(
            email_manager,
            gpu_manager.snatched_gpus,
            gpu_manager.num_snatched_gpus,
            gpu_manager.num_gpus,
            final=True,
            gpu_times_min=config.gpu_times_min,
        )

        countdown_timer(config.gpu_times_min, description="Releasing GPUs...")

    except KeyboardInterrupt:
        console.log("[red]Interrupted by user. Cleaning up...[/red]")
    finally:
        console.log("[red]Cleaning up GPU workers...[/red]")
        for p in processes:
            p.terminate()
            p.join()
        console.log("[green]All GPU workers terminated. Exiting.[/green]")


if __name__ == "__main__":
    main()
