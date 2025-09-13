import argparse
import multiprocessing
import os
import queue
import subprocess
from contextlib import nullcontext
from typing import Any

from gpusnatcher.configs import ConfigData, ConfigManager
from gpusnatcher.emails import EmailManager
from gpusnatcher.gpu import GPUManager
from gpusnatcher.logger import console
from gpusnatcher.utils import countdown_timer


def set_args() -> argparse.Namespace:
    """Set command line arguments."""
    parser = argparse.ArgumentParser(description="Manage and run GPU jobs automatically when GPU is free.")
    parser.add_argument("--job", dest="jobs", action="append", help="Job command to run when GPU is FREE.")
    parser.add_argument("-c", "--config", default=None, type=str, help="Path to config file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    return parser.parse_args()


class Job:
    """A job to be executed when a GPU is free."""

    def __init__(self, cmd: str, required_gpus: int = 1) -> None:
        """Initialize a Job instance."""
        self.cmd = cmd
        self.required_gpus = required_gpus

    def __repr__(self) -> str:
        """Return a string representation of the Job."""
        return f"<Job cmd={self.cmd!r} gpus={self.required_gpus}>"


def worker(gpu_indices: list[int], job: Job, ready_event: Any) -> None:
    """Run a job on assigned GPUs."""
    try:
        gpu_str = ",".join(map(str, gpu_indices))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str

        subprocess.Popen(job.cmd, env=env)  # noqa S603

        ready_event.set()
    except Exception as e:
        console.log(f"[red]Failed to start job {job}: {e}[/red]")


def parse_job(job_str: str) -> Job:
    """Parse a job string into a Job instance."""
    if ":" in job_str:
        cmd, gpus = job_str.rsplit(":", 1)
        job = Job(cmd.strip(), int(gpus))
    else:
        job = Job(job_str.strip(), 1)
    return job


def send_job_notification(email_mgr: EmailManager, job: Job, gpus: list[int], status: str) -> None:
    """Send a notification email about job status."""
    gpu_str = ", ".join(map(str, gpus))
    if status == "started":
        subject = f"GPUSitter: Job started on GPUs {gpu_str}"
        body = f"Job {job.cmd} started successfully on GPUs {gpu_str}."
    elif status == "finished":
        subject = f"GPUSitter: Job finished on GPUs {gpu_str}"
        body = f"Job {job.cmd} has finished execution on GPUs {gpu_str}."
    else:
        subject = "GPUSitter: Job status unknown"
        body = f"Job {job.cmd} on GPUs {gpu_str} has unknown status: {status}"

    email_mgr.send_email(subject=subject, body=body)
    console.log(f"[blue]Notification sent: {subject}[/blue]")


def check_finished(processes: list[tuple[multiprocessing.Process, Job, list[int]]], email_mgr: EmailManager) -> None:
    """Check for finished processes and send notifications."""
    finished = []
    for p, job, assigned in processes:
        if not p.is_alive():
            send_job_notification(email_mgr, job, assigned, "finished")
            p.join()
            finished.append((p, job, assigned))
    for item in finished:
        processes.remove(item)


def start_job(job: Job, assigned: list[int], email_mgr: EmailManager) -> multiprocessing.Process | None:
    """Start a job in a separate process."""
    ready_event = multiprocessing.Event()
    p = multiprocessing.Process(target=worker, args=(assigned, job, ready_event))
    p.start()

    if ready_event.wait(timeout=5) and p.is_alive():
        send_job_notification(email_mgr, job, assigned, "started")
        console.log(f"[green]Job {job} started successfully on GPUs {assigned}[/green]")
        return p

    console.log(f"[red]Job {job} failed to start on GPUs {assigned}[/red]")
    try:
        p.terminate()
        p.join()
    except Exception as e:
        console.log(f"[red]Failed to terminate process for job {job}: {e}[/red]")
    return None


def main() -> None:
    """The main entry point."""
    args = set_args()

    config_manager = ConfigManager(config_path=args.config)
    config_manager.load_or_create()
    config_manager.confirm_config()

    config: ConfigData = config_manager.config

    gpu_manager = GPUManager(
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

    jobs = queue.Queue()
    for job_str in args.jobs or []:
        jobs.put(parse_job(job_str))

    try:
        context = nullcontext() if args.debug else console.status("[green]Waiting for jobs...[/green]")
        with context:
            while not jobs.empty():
                free_gpus = gpu_manager.get_free_gpus()

                # Clean up finished processes and send notifications、
                check_finished(processes, email_manager)

                if not free_gpus:
                    continue

                # Wait a friendly amount of time before allocating GPUs
                countdown_timer(
                    config.friendly_min,
                    description=(
                        "Be friendly... "
                        "waiting before allocation to avoid OOM from previous job's final test/cleanup... "
                    ),
                    debug=args.debug,
                )

                # Re-check free GPUs after waiting
                free_gpu_indices = [
                    gpu_manager.gpu_maps(gpu["index"]) if gpu_manager.gpu_maps is not None else gpu["index"]
                    for gpu in free_gpus
                ]
                job = jobs.get()
                if len(free_gpu_indices) < job.required_gpus:
                    jobs.put(job)
                    continue
                assigned = free_gpu_indices[: job.required_gpus]

                # Start the job in a separate process
                p = start_job(job, assigned, email_manager)
                if p:
                    processes.append((p, job, assigned))
                else:
                    jobs.put(job)
                    console.log(f"[yellow]Job {job} re-queued due to failed start[/yellow]")

        for p, job, assigned in processes:
            p.join()
            send_job_notification(email_manager, job, assigned, "finished")

    except KeyboardInterrupt:
        console.log("[red]Interrupted by user. Cleaning up...[/red]")
    finally:
        console.log("[red]Cleaning up GPU workers...[/red]")
        for p, _, _ in processes:
            p.terminate()
            p.join()
        console.log("[green]All GPU workers terminated. Exiting.[/green]")


if __name__ == "__main__":
    main()
