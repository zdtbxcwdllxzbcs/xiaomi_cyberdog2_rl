"""Termination and event helpers aligned with Flamez SoccerEnv."""

from __future__ import annotations

import torch


def goal_scored(ball_xy: torch.Tensor, field_y_length: float, goal_width: float) -> tuple[torch.Tensor, torch.Tensor]:
    in_goal_mouth = torch.abs(ball_xy[:, 0]) <= 0.5 * goal_width
    blue_scored = (ball_xy[:, 1] >= 0.5 * field_y_length) & in_goal_mouth
    red_scored = (ball_xy[:, 1] <= -0.5 * field_y_length) & in_goal_mouth
    return blue_scored, red_scored


def ball_out_of_bounds(ball_xy: torch.Tensor, field_x_width: float, field_y_length: float, margin: float) -> torch.Tensor:
    return (torch.abs(ball_xy[:, 0]) > 0.5 * field_x_width + margin) | (
        torch.abs(ball_xy[:, 1]) > 0.5 * field_y_length + margin
    )


def ball_hit_wall(ball_xy: torch.Tensor, field_x_width: float, field_y_length: float,
                   ball_radius: float, wall_margin: float = 0.03) -> torch.Tensor:
    half_w = 0.5 * field_x_width - ball_radius - wall_margin
    half_l = 0.5 * field_y_length - ball_radius - wall_margin
    return (torch.abs(ball_xy[:, 0]) > half_w) | (torch.abs(ball_xy[:, 1]) > half_l)


def robot_hit_wall(robot_xy: torch.Tensor, field_x_width: float, field_y_length: float,
                    robot_radius: float, wall_margin: float = 0.05) -> torch.Tensor:
    half_w = 0.5 * field_x_width - robot_radius - wall_margin
    half_l = 0.5 * field_y_length - robot_radius - wall_margin
    return (torch.abs(robot_xy[:, 0]) > half_w) | (torch.abs(robot_xy[:, 1]) > half_l)


def robot_collision(robot_a_xy: torch.Tensor, robot_b_xy: torch.Tensor,
                     robot_radius: float) -> torch.Tensor:
    min_dist = 2.0 * robot_radius + 0.05
    return torch.linalg.norm(robot_a_xy - robot_b_xy, dim=1) < min_dist


def robot_goal_collision(robot_xy: torch.Tensor, field_y_length: float,
                          goal_width: float, robot_radius: float) -> torch.Tensor:
    half_l = 0.5 * field_y_length
    in_goal_mouth = torch.abs(robot_xy[:, 0]) < 0.5 * goal_width + robot_radius
    in_goal_depth = (robot_xy[:, 1] > half_l) | (robot_xy[:, 1] < -half_l)
    return in_goal_mouth & in_goal_depth
