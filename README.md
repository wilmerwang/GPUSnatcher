# GPUSitter

Watch for idle GPUs and run your jobs: launches jobs in tmux, keeps logs/status and sends start/finish emails..

## Features

- Real-time GPU usage monitoring
- Command-line interface, easy to integrate into workflows
- Email notifications
- Scheduled automatic job running

## Installation

```
pip install gpusitter
```

## Usage

```
gpust --job="python train.py"  # with 1 gpu

gpust --job="python train.py:4"  # with 4 gpus
```

Parameter description:

```
class ConfigData:
    """Configuration data for GPU Snatcher."""

    gpu_free_memory_ratio_threshold: float
    friendly_min: float
    email_host: str
    email_user: str
    email_pwd: str
    email_sender: str
    email_receivers: list[str]
```

- gpu_free_memory_ratio_threshold: The minimum free GPU memory ratio required to consider a GPU available. Only GPUs with free memory above this threshold will be used.
- friendly_min: Waiting time (in seconds) before allocating GPUs. Helps prevent OOM from previous jobs.
- email_host: Email server, e.g., smtp.qq.com
- email_user: Email address
- email_pwd: SMTP authorization code
- email_sender: Sender
- email_receivers: Recipients

# Contribution

Issues and pull requests are welcome. Please follow the project's code style guidelines.

# License

This project is licensed under the MIT License.
