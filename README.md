[简体中文](README-CN.md) | English

# GPUSnatcher

GPUSnatcher is a tool for GPU resource monitoring and snatching, designed to help users temporarily monitor and grab idle GPU resources.

## Features

- Real-time GPU usage monitoring
- Command-line interface, easy to integrate into workflows
- Email notifications
- Scheduled automatic resource release

## Installation

```
# Without pytorch
pip install gpusnatcher

# with pytorch
pip install gpusnatcher[cuda129]
```

Local installation：

```
git clone https://github.com/wilmerwang/GPUSnatcher.git
cd GPUSnatcher

# Without pytorch
pip install .

# with pytorch
pip install .[cuda129]
```

## Usage

```
# Run for the first time to configure parameters
gpusk
```

Parameter description:

```
class ConfigData:
    """Configuration data for GPU Snatcher."""

    gpu_nums: int
    gpu_times_min: int
    email_host: str
    email_user: str
    email_pwd: str
    email_sender: str
    email_receivers: list[str]
```

- gpu_nums: Number of GPUs to snatch
- gpu_times_min: Time to release resources after completing the specified GPU snatching plan
- email_host: Email server, e.g., smtp.qq.com
- email_user: Email address
- email_pwd: SMTP authorization code
- email_sender: Sender
- email_receivers: Recipients

# Contribution

Issues and pull requests are welcome. Please follow the project's code style guidelines.

# License

This project is licensed under the MIT License.
