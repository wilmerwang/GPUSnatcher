import contextlib
import os

import pynvml


def query_gpu() -> list[dict[str, int]] | None:
    """Query GPU information using pynvml.

    Returns:
        list[dict[str, int]] | None: A list of dictionaries containing GPU information, or None if querying fails.
        Each dictionary contains:
            - index (int): GPU index
            - memory.free (int): Free memory in MiB
            - memory.total (int): Total memory in MiB
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info.append(
                {
                    "index": i,
                    "memory.free": mem.free // (1024**2),  # bytes -> MiB
                    "memory.total": mem.total // (1024**2),  # bytes -> MiB
                }
            )

        return gpu_info if gpu_info else None

    except Exception as e:
        raise RuntimeError("Failed to query GPU using pynvml:") from e

    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


class GPUManager:
    """A class to manage GPU selection based on memory availability."""

    def __init__(self, gpu_free_memory_ratio_threshold: float = 0.85) -> None:
        """Initialize the GPU manager.

        Args:
            gpu_free_memory_ratio_threshold (float): The threshold for the free memory ratio to consider a GPU as free.
        """
        self.gpu_free_memory_ratio_threshold = gpu_free_memory_ratio_threshold

        self._gpu_maps: dict[int, int] | None = None

    def get_all_gpus(self) -> list[dict[str, int]]:
        """Get a list of all GPUs.

        Returns:
            list[dict[str, int]]: A list of dictionaries containing information about all GPUs.
        """
        gpus = query_gpu()
        if gpus is None:
            return []

        return self.get_visible_gpus(gpus)

    def get_free_gpus(self) -> list[dict[str, int] | None]:
        """Get a list of free GPUs.

        Returns:
            list[dict[str, int] | None]: A list of dictionaries containing information about free GPUs.
        """
        all_gpus = self.get_all_gpus()
        if not all_gpus:
            return []

        return [
            gpu for gpu in all_gpus if gpu["memory.free"] / gpu["memory.total"] > self.gpu_free_memory_ratio_threshold
        ]

    @property
    def gpu_maps(self) -> dict[int, int] | None:
        """Get the GPU mapping."""
        return self._gpu_maps

    @gpu_maps.setter
    def gpu_maps(self, value: dict[int, int] | None) -> None:
        """Set the GPU mapping."""
        self._gpu_maps = value

    def get_visible_gpus(self, gpus: list[dict[str, int]] | None) -> list[dict[str, int]] | None:
        """Filter GPUs based on the CUDA_VISIBLE_DEVICES environment variable."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is None:
            return gpus

        visible_gpus = [gpu for gpu in gpus if gpu["index"] in map(int, cuda_visible.split(","))]
        self.gpu_maps = {gpu["index"]: i for i, gpu in enumerate(visible_gpus)}

        return visible_gpus
