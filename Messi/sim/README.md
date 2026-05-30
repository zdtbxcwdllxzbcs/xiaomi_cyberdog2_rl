# sim/ — CyberDog 2v2 Pygame 战术板仿真

不依赖 Gazebo，纯 pygame + ROS2 实现的轻量战术板。`pygame_sim.py` 一个文件包含：

- 2D 物理引擎（机器人运动学、球摩擦/弹射、碰撞检测）
- ROS2 接口（发布 VRPN 坐标给策略节点，订阅速度指令）
- 比赛流程管理（开球、进球、出界、计分、计时）
- Pygame 实时渲染

---

## 环境准备

```bash
conda activate cyberdog
```

确认当前 shell 里有可用的 ROS2 环境（`ros2 --version` 能正常输出即可）。

安装 pygame（如未安装）：

```bash
pip install pygame
```

---

## 启动方式

### 方式一：仅启动仿真窗口（不带策略节点）

适合单独测试物理和渲染：

```bash
python3 pygame_sim.py
```

窗口打开后处于 **READY** 状态，按 `S` 开始比赛。

### 方式二：完整 2v2 仿真（推荐）

```bash
ros2 launch launch/sim_2v2_pygame.launch.py
```

可选参数：

```bash
ros2 launch launch/sim_2v2_pygame.launch.py kickoff_team:=team_b
```

启动顺序：

1. `t=0s` — `pygame_sim.py`（物理 + VRPN 发布 + 比赛管理 + 渲染）
2. `t=3s` — 四个策略进程（`main.py --config config/sim_2v2_team_*.yaml`）
3. `t=8s` — 自动调用 `/soccer/start`

### 方式三：接远端 ROS2 topic 的调试战术板

适合在本机打开 pygame 战术板，但 ROS2/VRPN graph 跑在 CyberDog 机器上。

#### 第一步：填写连接配置

编辑 [`sim/config/remote.yaml`](config/remote.yaml)，填入机器人 IP、SSH 用户名、密码和刚体名称：

```yaml
ssh:
  host: "192.168.1.100"   # 机器人 IP
  user: "mi"              # SSH 用户名
  password: "123"         # SSH 密码；已配置 SSH 密钥则留空
  remote_workspace: "~/cyberdog_soccer"

rigids:
  team_a_1: "SoTaGo1"    # 己方 Striker 的动捕刚体名
  team_a_2: "SoTaGo2"    # 己方 Goalkeeper 的动捕刚体名
  team_b_1: "opponent_1" # 对方 Striker 的动捕刚体名
  team_b_2: "opponent_2" # 对方 Goalkeeper 的动捕刚体名
  ball: "ball"            # 足球的动捕刚体名
```

#### 第二步：启动战术板

**Linux**（系统 ROS2）：

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
python3 sim/remote_tactical_board.py
```

**macOS**（conda 环境）：

```bash
conda activate cyberdog
python3 sim/remote_tactical_board.py
```

程序会自动读取 `sim/config/remote.yaml` 中的 SSH 参数和刚体映射，通过 SSH 在远端
`~/cyberdog_soccer` 启动一个小 router，订阅 `/vrpn/<rigid>/pose` 和
`/vrpn/<rigid>/twist`，再把 JSON snapshot 流回本地渲染。

#### 命令行覆盖

所有参数均可通过命令行覆盖 `sim/config/remote.yaml` 中的值：

```bash
# 覆盖 SSH 连接参数
python3 sim/remote_tactical_board.py \
  --host 10.0.0.54 \
  --user mi \
  --password 123

# 覆盖单个刚体映射
python3 sim/remote_tactical_board.py \
  --robot team_a_1=SoTaGo1 \
  --robot ball=soccer_ball

# 额外 source 远端 ROS2 环境
python3 sim/remote_tactical_board.py \
  --remote-setup ~/vrpn_client_ros2/src/install/setup.bash

# 打印生成的 SSH 命令后退出（调试用）
python3 sim/remote_tactical_board.py --print-command
```

#### 调试战术板键盘

| 键    | 功能                 |
| ----- | -------------------- |
| `R` | 重启 SSH topic router |
| `V` | 切换速度箭头叠加层   |
| `D` | 切换碰撞 footprint   |
| `Q` | 退出                 |

#### 侧边栏状态说明

| 标签 | 含义 |
| ---- | ---- |
| `SSH LIVE` | SSH 连接正常 |
| `SSH EXIT <code>` | SSH 进程已退出，3 秒后自动重连 |
| `topics: ok` | 所有刚体均在 0.6 s 内收到数据 |
| `missing: ...` | 列出从未收到数据的刚体 |
| `stale: ...` | 列出超过 0.6 s 未更新的刚体 |
| `state: approach` |  Striker 当前状态（approach / dribble / recover / waiting） |
| `vx / vy / wz` |  Striker 最新速度指令 |

---

## 键盘控制

| 键    | 功能               |
| ----- | ------------------ |
| `S` | 开始 / 恢复        |
| `P` | 暂停               |
| `R` | 重置               |
| `V` | 切换速度箭头叠加层 |
| `D` | 切换碰撞 footprint |
| `Q` | 退出               |

---

## ROS2 话题接口

### 仿真发布（策略节点订阅）

| 话题                         | 类型             | 说明                         |
| ---------------------------- | ---------------- | ---------------------------- |
| `/vrpn/{rigid_name}/pose`  | `PoseStamped`  | 机器人和球的位姿，z=0，20 Hz |
| `/vrpn/{rigid_name}/twist` | `TwistStamped` | 速度，20 Hz                  |
| `/vrpn/soccer_ball/pose`   | `PoseStamped`  | 球位姿                       |
| `/vrpn/goal_a/pose`        | `PoseStamped`  | 球门 A 静态位姿              |
| `/vrpn/goal_b/pose`        | `PoseStamped`  | 球门 B 静态位姿              |
| `/soccer/game_state`       | `String`       | 比赛状态                     |
| `/soccer/score`            | `String`       | 比分，格式 `team_a:team_b` |

rigid_name 对应：`team_a_1`、`team_a_2`、`team_b_1`、`team_b_2`

### 仿真订阅（策略节点发布）

| 话题                                 | 类型      | 说明                       |
| ------------------------------------ | --------- | -------------------------- |
| `/team_a/robot_1/motion_servo_cmd` | `Twist` | 机器人速度指令（体坐标系） |
| `/team_a/robot_2/motion_servo_cmd` | `Twist` |                            |
| `/team_b/robot_1/motion_servo_cmd` | `Twist` |                            |
| `/team_b/robot_2/motion_servo_cmd` | `Twist` |                            |

### 服务

| 服务              | 类型        | 说明           |
| ----------------- | ----------- | -------------- |
| `/soccer/start` | `Trigger` | 开始 / 恢复    |
| `/soccer/pause` | `Trigger` | 暂停           |
| `/soccer/reset` | `Trigger` | 重置到开球位置 |

---

## 比赛状态机

```text
INIT → READY → KICKOFF_TEAM_A/B → PLAYING → GOAL_TEAM_A/B → RESETTING → KICKOFF_*
                                           → OUT_OF_BOUNDS → RESETTING
                                           → PAUSED
```

- 开球等待 3s 后进入 PLAYING
- 进球后等待 3s 重置
- 出界后等待 2s 重置
- RESETTING 持续 2s 后进入下一次开球

---

## 物理参数

| 参数                  | 值        |
| --------------------- | --------- |
| 机器人接触 footprint  | 0.43 x 0.32 m |
| 机器人本体宽度        | 0.20 m    |
| 机器人调试包围圆半径  | 0.27 m    |
| 球半径                | 0.125 m   |
| 球摩擦减速度          | 1.5 m/s² |
| 墙壁弹性系数          | 0.6       |
| 机器人-球弹性系数     | 0.55      |
| 机器人-球切向耦合     | 0.12      |
| 物理步长              | 50 Hz     |
| 渲染帧率              | 30 FPS    |

---

## 目录结构

```text
sim/
├── pygame_sim.py      # 主仿真（物理 + ROS2 + 渲染）
├── cyberdog_2v2.py    # 机器人规格常量
└── README.md

launch/
└── sim_2v2_pygame.launch.py   # 完整 2v2 启动文件
```

---

## 调试

```bash
# 查看 VRPN 坐标是否正常发布
ros2 topic echo /vrpn/team_a_1/pose

# 查看比赛状态
ros2 topic echo /soccer/game_state

# 查看策略节点发出的速度指令
ros2 topic echo /team_a/robot_1/motion_servo_cmd

# 手动控制比赛
ros2 service call /soccer/start std_srvs/srv/Trigger "{}"
ros2 service call /soccer/pause std_srvs/srv/Trigger "{}"
ros2 service call /soccer/reset std_srvs/srv/Trigger "{}"
```
