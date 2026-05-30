"""Reward helpers — Flamez-aligned with role-specific shaping terms."""

from __future__ import annotations

import torch


def distance_reward(dist: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.exp(-dist / scale)


def progress_towards_goal(ball_y: torch.Tensor, prev_ball_y: torch.Tensor, sign: float) -> torch.Tensor:
    return sign * (ball_y - prev_ball_y)


def projected_ball_speed(ball_vel_y: torch.Tensor, sign: float) -> torch.Tensor:
    return (sign * ball_vel_y).clamp_min(0.0)


def facing_direction_reward(robot_yaw: torch.Tensor, sign: float) -> torch.Tensor:
    target_yaw = sign * 0.5 * 3.141592653589793
    return torch.cos(robot_yaw - target_yaw)


def facing_ball_reward(
    robot_xy: torch.Tensor, ball_xy: torch.Tensor, robot_yaw: torch.Tensor, sign: float
) -> torch.Tensor:
    """Reward robot for facing toward the ball."""
    dx = ball_xy[:, 0] - robot_xy[:, 0]
    dy = sign * (ball_xy[:, 1] - robot_xy[:, 1])
    target_yaw = torch.atan2(dy, dx)
    return torch.cos(robot_yaw - target_yaw)


def dribbling_reward(
    robot_xy: torch.Tensor,
    ball_xy: torch.Tensor,
    ball_vel: torch.Tensor,
    robot_vel: torch.Tensor,
    sign: float,
    dist_to_ball: torch.Tensor,
) -> torch.Tensor:
    ball_near = (dist_to_ball < 0.5).float()
    ball_speed_toward_goal = (sign * ball_vel[:, 1]).clamp_min(0.0)
    return ball_near * ball_speed_toward_goal


def approaching_goal_reward(
    robot_xy: torch.Tensor,
    field_y_length: float,
    sign: float,
    scale: float = 0.8,
) -> torch.Tensor:
    goal_y = sign * 0.5 * field_y_length
    dist_to_goal = torch.abs(robot_xy[:, 1] - goal_y)
    return torch.exp(-dist_to_goal / scale)
