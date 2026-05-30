"""Motion ID controller for Cyberdog2 soccer using TensorCommandMapper.

This controller uses the official CommandMapper to automatically select gait IDs
based on velocity commands, which is how the real robot system works.
"""

from __future__ import annotations

import torch

from cyberdog2_rl_lab.official.command_mapper import CommandMapperConfig, TensorCommandMapper


class MotionIdController:
    """Controller that uses TensorCommandMapper to automatically select motion IDs based on velocity commands.

    This mimics the real robot behavior where gait_id is automatically selected
    based on the commanded velocity:
    - Slow (vx <= 0.65, vy <= 0.30, wz <= 1.25): gait_id = 303
    - Medium (vx <= 1.00, vy <= 0.30, wz <= 1.25): gait_id = 308
    - Fast: gait_id = 305
    """

    def __init__(self, robots, num_envs, device):
        self.robots = robots
        self.num_envs = num_envs
        self.device = device
        
        # 使用 TensorCommandMapper，它会根据速度自动选择 gait_id
        self.command_mapper = TensorCommandMapper(
            num_envs=num_envs,
            device=device,
            cfg=CommandMapperConfig(use_speed_based_gait=True)
        )
        
        self.default_joint_pos = {
            agent: robot.data.default_joint_pos.clone()
            for agent, robot in robots.items()
        }
        
        print("[INFO] MotionIdController: Using TensorCommandMapper with speed-based gait selection")

    def reset(self, env_ids):
        """Reset controller state."""
        self.command_mapper.reset(env_ids)

    def compute_joint_targets(self, velocity_commands):
        """Compute joint targets based on velocity commands.

        Args:
            velocity_commands: Dict of velocity commands per agent, shape (N, 3)
                              [vx, vy, wz] in m/s and rad/s

        Returns:
            Dict of joint targets per agent, shape (N, num_joints)
        """
        joint_targets = {}
        for agent in self.robots.keys():
            # 使用 command_mapper 处理速度命令
            # 这会根据速度自动选择 gait_id
            self.command_mapper.map_normalized_action(velocity_commands[agent], dt=0.02)
            
            # 在仿真中返回默认关节位置
            # 在真实机器人上，这里的命令会被转换为 motion_id 发送给机器人
            joint_targets[agent] = self.default_joint_pos[agent].clone()
        
        return joint_targets

    def apply_joint_targets(self, joint_targets):
        """Apply joint targets to robots.

        For simulation: uses joint position control.
        For real robot: the gait_id would be sent via LCM/ROS.
        """
        for agent, robot in self.robots.items():
            robot.set_joint_position_target(joint_targets[agent])

    def get_current_gait_id(self, agent):
        """Get current gait ID for an agent.

        Returns:
            Tensor of gait_ids, shape (N,)
        """
        return self.command_mapper.last_gait_id

    def get_last_debug(self):
        """Get debug information about the last command mapping.

        Returns:
            Dict with gait selection info
        """
        return self.command_mapper.last_debug
