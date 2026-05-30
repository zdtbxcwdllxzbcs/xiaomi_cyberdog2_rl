# gym/ —  Striker  RL 训练

本目录包含 Cyberdog  Striker 角色的强化学习环境和 PPO 训练器。

```text
gym/
├── soccer_env.py          # Gymnasium 环境（SoccerEnv）
├── train_striker.py       # PPO 训练器，支持课程阶段
└── export_policy_onnx.py  # 将训练好的检查点导出为 ONNX
```

---

## 环境安装

RL 依赖通过 `uv` 管理，与 ROS2 环境相互独立。

```bash
# 在仓库根目录执行
uv sync
```

此命令安装 `stable-baselines3`、`gymnasium`、`torch`、`onnx`、`onnxruntime` 以及可选的 `swanlab`。

---

## 训练

### 快速开始

```bash
uv run python gym/train_striker.py --phase base
```

`base` 依次运行 Phase 1（空场地）和 Phase 2（脚本化 Goalkeeper ）。

### 课程阶段

| 阶段      | 描述                    | 前置阶段 |
| --------- | ----------------------- | -------- |
| `phase1`  | 空场地                  | —        |
| `phase2`  | 脚本化 Goalkeeper             | phase1   |
| `phase3`  | 2v2 对抗（脚本化对手）  | phase2   |
| `phase4`  | 2v2 + 运行时延迟微调    | phase3   |

简写课程：`base` = phase1+phase2，`all` = 全部四个阶段。

### 从检查点开始单阶段训练

```bash
uv run python gym/train_striker.py --phase phase3 \
  --start-checkpoint checkpoints/striker_rl/phase2/best/best_model.zip
```

### 自定义输出名称的微调

```bash
uv run python gym/train_striker.py --phase phase4 \
  --start-checkpoint checkpoints/striker_rl/phase3/best/best_model.zip \
  --output-name phase4_latency_ft
```

### 设备选择

默认设备为 `cpu`。对于小型 MLP 策略（256×256 两层网络）配合多个并行环境，CPU 通常比 GPU 更快，原因见下文。

```bash
# 默认：CPU
uv run python gym/train_striker.py --phase base

# 自动选择 MPS（Apple Silicon）或 CUDA
uv run python gym/train_striker.py --phase base --device auto

# 显式指定
uv run python gym/train_striker.py --phase base --device cuda
uv run python gym/train_striker.py --phase base --device mps
```

#### 为什么 CPU 对本任务通常比 GPU 更快

PPO 使用 `SubprocVecEnv` 在多个 CPU worker 上并行收集 rollout。每次前向传播是一个小型两层 MLP（256→256→3），批量大小最多为 `n_envs × n_steps = 8 × 2048 = 16 384` 条观测。在这个规模下：

- **GPU kernel 启动开销**超过实际计算时间。每次前向传播需要 host→device 拷贝、kernel 调度和 device→host 拷贝。对于能放入 L2 缓存的批量，这个往返开销比实际运算更耗时。
- **CPU BLAS**（OpenBLAS / MKL）对小型稠密矩阵乘法效率很高，且无任何传输开销。
- **数据并行已在 CPU 上**：`SubprocVecEnv` 的 worker 运行在 CPU 核心上。将观测发送到 GPU 再取回会在关键路径上增加延迟。

当策略网络较大（CNN、Transformer）或批量大小达到百万级时，GPU 才有优势。对于本任务的 12 维输入 MLP，CPU 更快。

### 并行环境数量

```bash
uv run python gym/train_striker.py --phase base --n-envs 8
```

`n-envs > 1` 使用 `SubprocVecEnv`（多进程）。设为 1 时使用 `DummyVecEnv`（进程内），便于调试。

### 延迟随机化

Phase 4 默认开启延迟。可对任意阶段覆盖：

```bash
# 强制 phase3 开启延迟
uv run python gym/train_striker.py --phase phase3 --latency on

# 调整延迟参数
uv run python gym/train_striker.py --phase phase4 \
  --policy-hz-min 8 --policy-hz-max 12 \
  --action-latency-min 0.0 --action-latency-max 0.08 \
  --observation-latency-min 0.0 --observation-latency-max 0.05
```

### SwanLab 日志

```bash
uv run python gym/train_striker.py --phase base --swanlab \
  --swanlab-project cyberdog-soccer-rl

# 本地/离线模式
uv run python gym/train_striker.py --phase base --swanlab \
  --swanlab-mode local
```

---

## 导出为 ONNX

机器人运行 Python 3.6 + glibc 2.27，无法安装 PyTorch 和 SB3。在开发机上将训练好的策略导出为 ONNX，再将 `.onnx` 文件复制到机器人上。

```bash
# 自动检测最佳可用检查点
uv run python gym/export_policy_onnx.py

# 显式指定检查点和输出路径
uv run python gym/export_policy_onnx.py \
  --checkpoint checkpoints/striker_rl/phase4/best/best_model.zip \
  --output checkpoints/striker_rl/phase4/best/best_model.onnx
```

导出器执行以下步骤：

1. 在 CPU 上加载 SB3 PPO 检查点。
1. 将 actor 网络封装为可追踪的 `nn.Module`。
1. 以 `opset=11` 导出，并将 ONNX IR 版本固定为 7（机器人上 `onnxruntime <=1.10.0` 的要求）。
1. 在 200 条随机观测上对比 ONNX 输出与原始 SB3 策略（最大允许误差：1e-4）。

---

## 在机器人上部署（Python 3.6，onnxruntime 1.10.0）

机器人运行 Ubuntu 18.04，Python 3.6.9，glibc 2.27。

### 安装 onnxruntime

```bash
pip3 install onnxruntime==1.10.0
```

`onnxruntime 1.10.0` 是最后一个官方支持 Python 3.6 和 glibc ≥ 2.17 的版本，提供自包含 wheel，无需 PyTorch 依赖。

若机器人无法访问 PyPI，可在联网机器上下载 wheel 后传输：

```bash
# 在开发机上下载
pip3 download onnxruntime==1.10.0 \
  --platform manylinux_2_17_aarch64 \
  --python-version 36 \
  --only-binary=:all: \
  -d /tmp/ort_wheels

# 传输到机器人
scp /tmp/ort_wheels/onnxruntime-1.10.0-*.whl mi@<机器人IP>:~/

# 在机器人上安装
pip3 install ~/onnxruntime-1.10.0-*.whl
```

> **架构说明**：Cyberdog 2 使用 NVIDIA Orin（aarch64），使用上述 `manylinux_2_17_aarch64` 平台标签。x86_64 机器人请替换为 `manylinux_2_17_x86_64`。

### 复制 ONNX 模型

```bash
scp checkpoints/striker_rl/phase4/best/best_model.onnx \
  mi@<机器人IP>:~/cyberdog_soccer/checkpoints/striker_rl/phase4/best/
```

### 运行时推理

`src/lib/policy_onnx.py` 提供 `PPO.predict()` 的直接替代：

```python
from src.lib.policy_onnx import OnnxPolicy

policy = OnnxPolicy("checkpoints/striker_rl/phase4/best/best_model.onnx")
action, _ = policy.predict(obs)
```

运行时 Striker 角色（`src/striker_policy.py`）在 `striker_policy.model_path` 以 `.onnx` 结尾时自动加载 ONNX 模型。在 `config/config.yaml` 中设置路径：

```yaml
striker_policy:
  model_path: checkpoints/striker_rl/phase4/best/best_model.onnx
```

### 在机器人上验证

```python
import numpy as np
from src.lib.policy_onnx import OnnxPolicy

policy = OnnxPolicy("checkpoints/striker_rl/phase4/best/best_model.onnx")
obs = np.zeros(12, dtype=np.float32)
action, _ = policy.predict(obs)
print("action:", action)  # 应输出 3 个在 [-1, 1] 范围内的浮点数
```

---

## 检查点目录结构

```text
checkpoints/striker_rl/
├── phase1/
│   ├── best/best_model.zip
│   └── striker_final.zip
├── phase2/
│   ├── best/best_model.zip
│   └── striker_final.zip
├── phase3/
│   ├── best/best_model.zip
│   └── striker_final.zip
└── phase4/
    ├── best/best_model.zip
    ├── best/best_model.onnx   ← 导出供机器人部署
    └── striker_final.zip
```
