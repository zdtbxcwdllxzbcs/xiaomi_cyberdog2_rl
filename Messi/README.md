# Messi Cyberdog Soccer

2026 年 Messi 队开源仓库。

基于 **Isaac Sim + Isaac Lab** 三维物理仿真，采用 **双层强化学习架构** 训练 Cyberdog 2 足球机器人。底层模型负责稳定运动（对角步伐行走），高层策略负责足球战术决策（找球、带球、射门、协同）。

---

## 训练架构

### 双层控制

```
高层策略（足球战术）        →  速度指令 [vx, vy, ωz]
        ↓
底层运动模型（冻结推理）    →  12 关节角度
        ↓
Cyberdog 2 机器人
```

- **底层模型**：RSL-RL PPO 训练，48 维观测（含 base_lin_vel），输出 12 个关节位置目标。在 Isaac Sim 中完成速度跟踪训练，支持对角步态（trot），速度范围 ±1.6 m/s。
- **高层策略**：12 维 Flamez 观测（球、门、队友、对手的相对位置），采用 Canonical 坐标对齐队伍方向。冻结底层的 ONNX 模型后，在 Isaac Lab 足球环境中直接训练高层 PPO 策略。

### 四阶段课程训练

| 阶段 | 名称 | 技能 | 对手 |
|------|------|------|------|
| **Skill 1** | 找球绕后 | 移动至球后方（朝向对方球门） | 无 |
| **Skill 2** | 带球前进 | 持球向对方球门推进 | 无 |
| **Skill 3** | 射门 | 在合适位置射门得分 | 静态守门员 |
| **Skill 4** | 完整 2v2 | 队友配合，对抗完整对方队伍 | 守门员 + 前锋 |

每阶段训练完成后，模型作为下一阶段的预训练权重热启动，逐步掌握完整足球行为。

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 底层模型 | RSL-RL PPO | 速度跟踪，对角步态 |
| 预热步数 | 40 步（0.8s） | 速度指令从 0 渐增，降低初始摔倒 |
| 出生高度 | 0.35m | 降低初始悬空高度 |
| 速度限制 | vx=0.8, vy=0.5, ω=1.5 | 限制高层输出，匹配底层能力 |
| 仿真环境数 | 512 | 并行训练加速 |
| 每回合时长 | 30s | 1500 步仿真步 |

---

## 运行

从仓库根目录启动：

```bash
python3 main.py
```

`./start.sh` 是对上述命令的简单封装。

可复用的机器狗组件位于 `src/lib/`，`src/striker.py` 和 `src/goalkeeper.py` 专注于角色策略逻辑。

---

## 坐标系

Locator 维护三层坐标系：

- **动捕坐标系**：原始 `/vrpn/<rigid>/*` 数据。
- **场地坐标系**：归一化后的 `/soccer/world/*` 话题。
- **机器人本体坐标系**：归一化后的 `/soccer/relative/*` 话题及速度指令。

场地坐标系由两个球门中心刚体动态建立：

- 原点：`goal_left` 与 `goal_right` 的中点。
- `+y`：从 `goal_left` 指向 `goal_right`。
- `+x`：在平面场地坐标系中垂直于 `+y`。
- 场地长度：实测球门间距。
- 场地宽度：长度乘以配置的宽长比。

两个球门均须由动捕系统以刚体形式发布。运行时配置可显式指定 `goal_left` / `goal_right`；
旧版仿真配置则从 `goal_a` / `goal_b` 推断同一场地轴。

---

## 运行时结构

```text
main.py
  加载 config/config.yaml，在单进程中启动所有 ROS2 节点

src/lib/locator.py
  订阅 VRPN 刚体，重新发布 world/relative 足球话题

src/lib/move.py
  发布 /<dog_namespace>/motion_servo_cmd，带速度限幅

src/lib/ball_predictor.py
  根据球速度和防守球门刚体预测 Goalkeeper 防守目标

src/lib/path_planner.py
  在红队场地坐标系中用 A* 规划 Striker 路径

src/lib/path_follower.py
  跟随 Striker 路径点并向 MoveCommander 发送指令

src/striker.py
  Striker 状态机：wait、recover、approach、dribble

src/goalkeeper.py
  Goalkeeper 状态机：lateral_defend、active_intercept
```

---

## 配置

运行前编辑 `config/config.yaml`。

重要字段：

- `role`：`striker` 或 `goalkeeper`。
- `dog_namespace`：机器人命名空间，完整话题名为 `/<dog_namespace>/motion_servo_cmd`。
- `rigids.self`：本机器人的 VRPN 刚体名。
- `rigids.teammate`、`rigids.opponent_1`、`rigids.opponent_2`：动态障碍物。
- `rigids.ball`：足球刚体名。
- `rigids.attack_goal`、`rigids.defense_goal`：来自 VRPN 的球门刚体名。
- `field`：场地尺寸、球门宽度和边界安全距离。
- `move`：速度限幅、指令符号和 Cyberdog 步态 ID。
- `path`：A* 栅格分辨率、障碍物半径和路径点容差。
- `striker` / `goalkeeper`：角色专属控制增益和距离参数。

球门刚体未配置或未发布时，对应角色将等待或仅保持原地，不使用固定坐标回退。

---

## ROS2 话题

动捕输入：

- `/vrpn/<rigid>/pose`
- `/vrpn/<rigid>/twist`
- `/vrpn/<rigid>/accel`

Locator 输出：

- `/soccer/world/<object>`（`geometry_msgs/PoseStamped`）
- `/soccer/relative/<object>`（`geometry_msgs/PoseStamped`）
- `/soccer/world/ball_twist`（`geometry_msgs/TwistStamped`）
- `/soccer/relative/ball_twist`（`geometry_msgs/TwistStamped`）

Striker 规划：

- `/soccer/striker/target`（`geometry_msgs/PoseStamped`）
- `/soccer/striker/path`（`nav_msgs/Path`）

Goalkeeper 预测：

- `/soccer/goalkeeper/guard_target`（`geometry_msgs/PoseStamped`）

机器狗运动指令：

- `/<dog_namespace>/motion_servo_cmd`

测试环境中若 `protocol.msg.MotionServoCmd` 不可用，`src/lib/move.py` 会回退到在同一话题上发布 `geometry_msgs/Twist`，保证导入和干运行测试正常进行。

---

## 环境准备

### Linux

直接使用系统 ROS2（推荐）：

```bash
sudo apt install ros-humble-desktop   # 或 ros-jazzy-desktop 等
```

### macOS — 通过 RoboStack 在 conda 中安装 ROS2

macOS 没有官方 ROS2 二进制包，推荐用 [RoboStack](https://robostack.github.io/) 通过 conda 安装：

```bash
# 安装 mamba（比 conda 解析依赖更快）
conda install -n base -c conda-forge mamba

# 创建 cyberdog 环境
mamba create -n cyberdog python=3.11
conda activate cyberdog

# 添加 robostack 频道
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --set channel_priority strict

# 安装 ROS2 Humble（也可选 iron / jazzy）
mamba install ros-humble-desktop

# 安装其他依赖
pip install pygame pyyaml

# 验证
ros2 --version
```

激活 conda 环境后无需手动 `source setup.bash`，ROS2 环境自动生效。

---

## 运行

**Linux** — 直接使用系统 ROS2：

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
cd cyberdog_soccer
python3 main.py --config config/config.yaml
```

**macOS** — 使用 conda 环境：

```bash
conda activate cyberdog
python3 main.py --config config/config.yaml
```

切换角色，编辑配置文件：

```yaml
role: striker
```

或：

```yaml
role: goalkeeper
```

按 `Ctrl-C` 停止。`main.py` 在关闭时会发送零速停止指令。
