[English](README.md) | 简体中文

# GPUSnatcher

GPUSnatcher 是一个用于 GPU 资源监控和掠夺的工具，旨在帮助用户临时监控并抢占空闲资源。

## 功能特点

- 实时监控 GPU 使用情况
- 提供命令行界面，易于集成到工作流
- 邮件提醒
- 定时自动释放资源

## 安装方法

```
# 不包含 pytorch 依赖
pip install GPUSnatcher

# 包含 pytorch 依赖
pip install GPUSnatcher[cuda129]
```

本地安装：

```
git clone https://github.com/yourusername/GPUSnatcher.git
cd GPUSnatcher

# 不包含 pytorch 依赖
pip install .

# 包含 pytorch 依赖
pip install .[cuda129]
```

## 使用方法

```bash
# 首次运行需要配置参数
gpusk
```

参数说明：

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

- gpu_nums: 抢占 GPU 数量
- gpu_times_min: 完成指定 GPU 数量抢占计划后，释放资源的时间
- email_host: 邮箱服务器，比如 `smtp.qq.com`
- email_user: 邮箱地址
- email_pwd: SMTP授权码
- email_sender: 发送者
- email_receivers: 接收者

## 贡献指南

欢迎提交 issue 和 pull request。请遵循项目的代码规范。

## 许可证

本项目采用 MIT 许可证。
