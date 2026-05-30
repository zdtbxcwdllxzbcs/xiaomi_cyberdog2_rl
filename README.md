# Cyberdog2 RL Lab

> 小米 Cyberdog 2 机器狗 Isaac Sim 强化学习训练全流程

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-green)](https://developer.nvidia.com/isaac/sim)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

本项目提供了在 **Isaac Sim 4.5.0** 下使用 **RSL-RL (PPO)** 训练 Cyberdog 2 机器狗平地步态控制和足球策略的完整代码。包含仿真环境配置、多阶段课程训练脚本、策略导出工具，以及实战部署代码。

---

## 目录结构

```
cyberdog2_rl_lab_opensource/
├── README.md                           # 本文件
├── setup.py / pyproject.toml           # Python 包安装配置
├── config/extension.toml               # Isaac Lab 扩展元数据
├── attack.py                           # 足球进攻独立测试脚本
├── cyberdog2_rl_lab/                   # Isaac Lab 环境 + 部署库源码
│   ├── assets/                         # Cyberdog2 URDF + 网格文件
│   │   └── data/cyberdog2/
│   │       ├── meshes/                 # .dae 网格
│   │       └── urdf/cyberdog2.urdf     # 运动学模型
│   ├── official/                       # 真机部署：命令映射 + LCM桥接
│   │   ├── command_mapper.py           # 速度→MotionServoCmd 映射
│   │   ├── command_types.py            # 命令类型定义
│   │   ├── deploy_schema.py            # 部署配置 schema
│   │   ├── lcm_bridge.py               # LCM 通讯桥
│   │   └── motion_catalog.py           # 步态目录 (motion_id 303/305/308)
│   ├── runtime/                        # 部署推理
│   │   ├── soccer_policy_runner.py     # 足球策略 ONNX 推理
│   │   ├── vrpn_state_receiver.py      # VRPN 动捕状态接收
│   │   ├── official_locomotion_policy_runner.py  # 官方步态策略推理
│   │   └── official_cmd_publisher.py   # 命令发布器
│   ├── tasks/
│   │   ├── locomotion/                 # 平地步态任务 (Manager-based)
│   │   │   ├── robots/cyberdog2/velocity_env_cfg.py  # 速度跟踪配置
│   │   │   ├── mdp/                    # 观测/奖励/指令
│   │   │   └── agents/                 # PPO 超参
│   │   ├── official_locomotion/        # 官方步态环境 (Direct RL)
│   │   └── soccer/                     # 2v2 足球环境 (Direct MARL)
│   │       ├── soccer_env.py           # 四阶段课程 RL 主环境
│   │       ├── soccer_env_cfg.py       # 环境配置
│   │       ├── observation_schema.py   # 73维观测定义
│   │       ├── mdp/                    # 观测/奖励/终止条件
│   │       ├── agents/rsl_rl_ppo_cfg.py # PPO 超参
│   │       ├── locomotion_policy_controller.py  # 步态策略控制器
│   │       ├── motion_id_controller.py # motion_id 调度器
│   │       └── field_config.py         # 场地配置加载
│   └── utils/
├── scripts/
│   ├── rsl_rl/train.py                 # 步态训练入口
│   ├── rsl_rl/train_soccer.py          # 足球训练入口 (4阶段课程)
│   ├── rsl_rl/play.py                  # 步态回放 + 导出
│   ├── rsl_rl/play_soccer.py           # 足球回放 + 导出
│   ├── rsl_rl/little.py                # 轻量训练测试
│   ├── export_soccer_policy.py         # 导出为 Flamez 兼容 ONNX
│   ├── export_policy_onnx.py           # 通用 ONNX 导出
│   ├── export_locomotion_policy.py     # 步态策略导出
│   ├── smoke_direct_tasks.py           # 冒烟测试 (Direct RL)
│   ├── smoke_official_bridge.py        # 冒烟测试 (LCM桥接)
│   ├── smoke_soccer_locomotion_bridge.py # 冒烟测试 (足球+步态)
│   ├── list_envs.py                    # 列出注册的 Gym 任务
│   └── check_ckpt.py                   # 检查 checkpoint 信息
├── checkpoints/                        # 预训练模型
│   └── velocity_flat/                  # 步态行走策略
│       ├── exported/policy.onnx         # ONNX 模型 (opset 15, 760KB)
│       ├── exported/policy.pt           # PyTorch 模型 (760KB)
│       └── params/                     # 训练超参 + 观测/动作配置
│           ├── agent.yaml              # PPO + MLP 超参
│           ├── deploy.yaml             # 观测/动作配置
│           ├── env.yaml                # 完整环境配置
│           └── velocity_env_cfg.py     # 环境类配置
└── Messi/                              # 实战部署代码 (submodule)
    ├── main.py                         # 主入口
    ├── start.sh                        # 启动脚本
    ├── vrpn.sh                         # VRPN 客户端脚本
    ├── config/                         # 比赛配置
    │   ├── config.yaml                 # 主配置
    │   └── sim_2v2_*.yaml              # 2v2 仿真配置
    ├── sim/                            # 仿真/战术板
    ├── src/
    │   ├── striker.py / striker_policy.py  # Striker 角色 (规则/RL)
    │   ├── goalkeeper.py / goalkeeper_policy.py  # Goalkeeper 角色
    │   └── lib/                        # 可复用库 (move, locator, geometry 等)
    └── tests/                          # 单元测试
```

---

## 环境准备

### 1. Isaac Sim 环境

在 Windows 上的 NVIDIA Isaac Sim 4.5.0 环境中运行仿真训练：

```powershell
# 安装 Isaac Lab
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
# 按 Isaac Lab 文档安装 isaacsim 等依赖

# 安装本扩展 (可编辑模式)
cd cyberdog2_rl_lab_opensource
python -m pip install -e .
```

### 2. Messi 部署环境 (真机)

Messi 子目录可在真机 (Ubuntu 18.04 / Python 3.6 / ROS 2) 或任何有 ROS 2 的开发机上运行：

```bash
# Linux — 系统 ROS2
source /opt/ros/${ROS_DISTRO}/setup.bash
python3 main.py --config config/config.yaml

# macOS — conda 环境 (RoboStack)
conda activate cyberdog
python3 main.py --config config/config.yaml
```

Messi 部署依赖详见 [Messi/README.md](Messi/README.md)。

---

## 仿真训练

### 步态训练 (平地步态控制)

```powershell
conda activate env_isaaclab

# 小规模调试 (128 环境)
python scripts/rsl_rl/train.py --task Cyberdog2-Velocity-Flat-v0 --num_envs 128 --max_iterations 200

# 正式训练 (2048 环境，无头)
python scripts/rsl_rl/train.py --task Cyberdog2-Velocity-Flat-v0 --headless --num_envs 2048

# 回放最新 checkpoint (GUI)
python scripts/rsl_rl/play.py --task Cyberdog2-Velocity-Flat-Play-v0 --num_envs 1
```

### 足球训练 (2v2 足球)

```powershell
conda activate env_isaaclab

# 完整阶段训练 (Phase 1 → 2 → 3 → 4)
python scripts/rsl_rl/train_soccer.py --task Cyberdog2-Soccer-2v2-Play-v0 --curriculum all

# 仅基础阶段 (Phase 1 + 2)
python scripts/rsl_rl/train_soccer.py --curriculum base

# 指定迭代次数
python scripts/rsl_rl/train_soccer.py --curriculum all --phase1_iterations 500 --phase2_iterations 500 --phase3_iterations 1000 --phase4_iterations 300
```

**课程阶段说明：**

| 阶段 | 描述 | 场景 |
|------|------|------|
| Phase 1 (Skill 1) | 找球 + 站到球后方 | 空场地，仅受控 striker |
| Phase 2 (Skill 2) | + 带球推进 | 空场地 |
| Phase 3 (Skill 3) | + 射门得分 | 空场地 |


| Phase 2 | + 脚本化守门员 | striker + red goalkeeper |
| Phase 3 | + 队友守门员 + 对手 striker | 完整 2v2 |
| Phase 4 | + 延迟随机化 | 完整 2v2 + latency sim-to-real |

---

## 预训练模型

本仓库包含一个已训练完成的**平地步态行走策略**，可直接在 Isaac Sim 中加载使用。

### 步态模型 `velocity_flat`

| 属性 | 值 |
|------|-----|
| 任务 | `Cyberdog2-Velocity-Flat-v0` |
| 观测维度 | **48** (base线速度3 + base角速度3 + 重力投影3 + 速度指令3 + 关节位置12 + 关节速度12 + 上一动作12) |
| 动作维度 | **12** (12个关节的目标位置偏移) |
| 网络结构 | MLP \[512, 256, 128\] + ELU |
| 控制频率 | 50 Hz (decimation=4, sim dt=0.005s) |
| 命令范围 | vx ∈ [-1.6, 1.6] m/s, vy ∈ [-0.55, 0.55] m/s, ωz ∈ [-2.5, 2.5] rad/s |
| 训练 | PPO, 2048 并行环境, 25400 迭代 |
| 格式 | PyTorch (.pt) + ONNX (.onnx, opset 15) |

### 在仿真中加载（Isaac Sim）

```powershell
conda activate env_isaaclab

# 使用预训练步态模型回放
python scripts/rsl_rl/play.py \
    --task Cyberdog2-Velocity-Flat-Play-v0 \
    --num_envs 1 \
    --checkpoint checkpoints/velocity_flat/exported/policy.pt
```

> **注意**：`play.py` 默认从 `logs/` 目录查找 checkpoint。若要从 `checkpoints/` 加载，需要指定 `--checkpoint` 参数。你也可以将 `exported/policy.pt` 复制到 logs 对应目录下。

### 用 ONNX 直接推理（无需 Isaac Sim）

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("checkpoints/velocity_flat/exported/policy.onnx")
obs = np.zeros((1, 48), dtype=np.float32)  # 48维观测
action = session.run(None, {"obs": obs})[0]  # 输出 12维关节位置
print(action.shape)  # (1, 12)
```

### 查看训练配置

```bash
# 查看 PPO 超参
cat checkpoints/velocity_flat/params/agent.yaml

# 查看观测/动作/命令配置
cat checkpoints/velocity_flat/params/deploy.yaml
```

---

## 模型导出

### 导出为通用 ONNX (Isaac Sim 策略)

```powershell
python scripts/rsl_rl/play_soccer.py --task Cyberdog2-Soccer-2v2-Play-v0 --num_envs 1
# ONNX 自动导出到 logs/rsl_rl/<experiment>/<run>/exported/policy.onnx
```

### 导出为 Messi 兼容 ONNX (12维 → 3维)

```powershell
python scripts/export_soccer_policy.py \
    --checkpoint logs/rsl_rl/cyberdog2_soccer_2v2/<run>/model_XXXXX.pt \
    --output checkpoints/striker_rl/phase3/best/best_model.onnx
```

导出脚本会自动：
1. 从 checkpoint 提取 MLP 权重（跳过 obs_normalizer）
2. 构建 `Linear(12)→ELU→Linear(256)→ELU→Linear(256)→ELU→Linear(3)→tanh` 网络
3. 导出为 opset 15 ONNX (兼容 onnxruntime 1.10.0 / Python 3.6 / aarch64)

---

## Sim-to-Real 部署

### 架构

```
VRPN 动捕 → /vrpn/{rigid}/pose → Locator (场地坐标系)
                                    ↓
                          /soccer/world/* (归一化坐标)
                                    ↓
                          StrikerPolicyNode (策略推理)
                                    ↓
                          MoveCommander (速度限幅 + body-frame 转换)
                                    ↓
                          /{dog_name}/motion_servo_cmd (ROS 2)
```

### 真机部署流程

1. **导出版权 ONNX 模型** (开发机)
   ```powershell
   python scripts/export_soccer_policy.py --checkpoint model_XXXXX.pt --output best_model.onnx
   ```

2. **拷贝到真机**
   ```bash
   scp best_model.onnx mi@<狗IP>:~/cyberdog_soccer/checkpoints/
   ```

3. **编辑配置文件**
   编辑 `Messi/config/config.yaml`，设置 VRPN 名字、狗命名空间、模型路径等

4. **启动**
   ```bash
   cd Messi && python3 main.py --config config/config.yaml
   ```


---

## 注册的任务列表

| 任务 ID | 类型 | 描述 |
|---------|------|------|
| `Cyberdog2-Velocity-Flat-v0` | Manager-based | 平地步态速度跟踪 (训练) |
| `Cyberdog2-Velocity-Flat-Play-v0` | Manager-based | 平地步态回放 |
| `Cyberdog2-Official-Locomotion-Play-v0` | Direct RL | 官方步态回放 |
| `Cyberdog2-Soccer-2v2-Play-v0` | Direct MARL | 2v2 足球 (训练+回放) |
| `Cyberdog2-Soccer-Single-Play-v0` | Direct MARL | 单人足球回放 |

---

## 技术栈

- **仿真引擎**: NVIDIA Isaac Sim 4.5.0 + PhysX
- **RL 框架**: RSL-RL (PPO, 256→256→128 MLP + ELU)
- **并行使能**: Isaac Lab Multi-Agent RL + SubprocVecEnv (2048 envs)
- **机器人模型**: Cyberdog 2 URDF (12 关节, 四足)
- **部署**: ROS 2 + VRPN + ONNX Runtime 1.10.0 (aarch64)

---

## License

Apache 2.0 — 详见 [LICENSE](LICENSE) 文件。

---

## 致谢

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — 仿真框架
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — PPO 实现
- [Flamez Cyberdog Soccer](https://github.com/zgdllt/Flamez_cyberdog_soccer) — Messi 子目录基于火仔队开源的赛事实战方案
