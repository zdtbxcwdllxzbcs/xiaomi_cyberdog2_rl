"""Unit tests for math-heavy code that does not need live motion capture.

Run these in the ROS2-enabled environment because planner/predictor modules
import standard ROS message packages for their node adapters.
"""

import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from geometry_msgs.msg import PoseStamped, TwistStamped

from src.lib.ball_predictor import predict_guard_target
from src.lib.geometry import (
    Point2D,
    build_field_transform,
    field_to_body,
    normalize_angle,
    quaternion_from_yaw,
    yaw_from_quaternion,
)
from src.lib.locator import LocatorNode
from src.lib.move import MoveCommander
from src.lib.path_follower import PathFollowerNode
from src.lib.path_planner import AStarPathPlanner, CircularObstacle
from src.striker import StrikerNode
from src.striker_legacy import StrikerLegacyNode


class GeometryTests(unittest.TestCase):
    """Verify the red-frame coordinate helpers used by all role controllers."""

    def test_field_to_body_forward_when_robot_faces_positive_y(self):
        local = field_to_body(0.0, 1.0, math.pi * 0.5)
        self.assertAlmostEqual(local.x, 1.0, places=6)
        self.assertAlmostEqual(local.y, 0.0, places=6)

    def test_yaw_round_trip_and_angle_normalization(self):
        _, _, qz, qw = quaternion_from_yaw(0.75)
        quaternion = SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw)
        self.assertAlmostEqual(yaw_from_quaternion(quaternion), 0.75, places=6)
        self.assertAlmostEqual(normalize_angle(3.0 * math.pi), math.pi, places=6)

    def test_field_transform_is_built_from_goal_centers(self):
        transform = build_field_transform(
            SimpleNamespace(x=10.0, y=20.0, z=1.0),
            SimpleNamespace(x=10.0, y=30.0, z=1.4),
            width_to_length_ratio=0.555,
        )

        left = transform.position_to_field(SimpleNamespace(x=10.0, y=20.0, z=1.0))
        right = transform.position_to_field(SimpleNamespace(x=10.0, y=30.0, z=1.4))
        center = transform.position_to_field(SimpleNamespace(x=12.0, y=25.0, z=1.2))

        self.assertAlmostEqual(transform.length, 10.0, places=6)
        self.assertAlmostEqual(transform.width, 5.55, places=6)
        self.assertAlmostEqual(left[0], 0.0, places=6)
        self.assertAlmostEqual(left[1], -5.0, places=6)
        self.assertAlmostEqual(right[0], 0.0, places=6)
        self.assertAlmostEqual(right[1], 5.0, places=6)
        self.assertAlmostEqual(center[0], 2.0, places=6)
        self.assertAlmostEqual(center[1], 0.0, places=6)
        self.assertAlmostEqual(center[2], 0.0, places=6)


class LocatorCoordinateTests(unittest.TestCase):
    """Verify locator conversion from mocap frame into field and body frames."""

    def _pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        msg = PoseStamped()
        msg.pose.position.x = x
        msg.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def test_world_pose_rotates_mocap_into_goal_defined_field_frame(self):
        locator = LocatorNode.__new__(LocatorNode)
        locator.field_frame = "field_red"
        locator.field_transform = build_field_transform(
            SimpleNamespace(x=0.0, y=0.0),
            SimpleNamespace(x=4.0, y=0.0),
            width_to_length_ratio=0.5,
        )

        world_pose = LocatorNode._world_pose(locator, self._pose(2.0, -1.0, 0.0))

        self.assertEqual(world_pose.header.frame_id, "field_red")
        self.assertAlmostEqual(world_pose.pose.position.x, 1.0, places=6)
        self.assertAlmostEqual(world_pose.pose.position.y, 0.0, places=6)
        self.assertAlmostEqual(
            yaw_from_quaternion(world_pose.pose.orientation),
            math.pi * 0.5,
            places=6,
        )

    def test_world_twist_rotates_mocap_velocity_into_field_frame(self):
        locator = LocatorNode.__new__(LocatorNode)
        locator.field_frame = "field_red"
        locator.field_transform = build_field_transform(
            SimpleNamespace(x=0.0, y=0.0),
            SimpleNamespace(x=4.0, y=0.0),
            width_to_length_ratio=0.5,
        )
        twist = TwistStamped()
        twist.twist.linear.x = 1.0
        twist.twist.linear.y = 0.0

        world_twist = LocatorNode._world_twist(locator, twist)

        self.assertEqual(world_twist.header.frame_id, "field_red")
        self.assertAlmostEqual(world_twist.twist.linear.x, 0.0, places=6)
        self.assertAlmostEqual(world_twist.twist.linear.y, 1.0, places=6)

    def test_goal_object_inference_uses_stable_left_right_rigid_names(self):
        locator = LocatorNode.__new__(LocatorNode)
        locator.config = {"locator": {}}
        locator.rigids = {
            "self": "team_b_1",
            "attack_goal": "goal_a",
            "defense_goal": "goal_b",
        }

        self.assertEqual(
            LocatorNode._resolve_field_goal_objects(locator),
            ("attack_goal", "defense_goal"),
        )


class PathPlannerTests(unittest.TestCase):
    """Verify A* path generation and obstacle inflation."""

    def test_obstacle_center_cell_is_blocked(self):
        planner = AStarPathPlanner(resolution=0.1)
        obstacle = CircularObstacle(Point2D(0.0, 0.0), 0.35)
        blocked = planner.inflate_obstacles([obstacle])
        self.assertIn(planner.point_to_cell(obstacle.center), blocked)

    def test_path_routes_around_obstacle(self):
        planner = AStarPathPlanner(resolution=0.1)
        path = planner.plan(
            Point2D(0.0, -2.0),
            Point2D(0.0, 2.0),
            [CircularObstacle(Point2D(0.0, 0.0), 0.45)],
        )
        self.assertGreater(len(path), 2)
        self.assertTrue(any(abs(point.x) > 0.3 for point in path[1:-1]))


class MoveCommanderTests(unittest.TestCase):
    """Verify velocity limiting at the shared motion-command boundary."""

    def _move(self):
        move = MoveCommander.__new__(MoveCommander)
        move.speed_x = 0.0
        move.speed_y = 0.0
        move.speed_yaw = 0.0
        move.max_speed_x = 1.0
        move.max_speed_y = 0.5
        move.max_speed_yaw = 1.5
        move.publish_on_change = False
        move.command_epsilon = 1e-4
        move._motion_changed = False
        return move

    def test_planar_limit_preserves_direction_when_x_saturates(self):
        move = self._move()

        MoveCommander.change_speed(move, 2.0, 0.5, 0.2)

        self.assertAlmostEqual(move.speed_x, 1.0)
        self.assertAlmostEqual(move.speed_y, 0.25)
        self.assertAlmostEqual(move.speed_yaw, 0.2)

    def test_planar_limit_preserves_direction_when_y_saturates(self):
        move = self._move()

        MoveCommander.change_speed(move, -0.5, 2.0, -3.0)

        self.assertAlmostEqual(move.speed_x, -0.125)
        self.assertAlmostEqual(move.speed_y, 0.5)
        self.assertAlmostEqual(move.speed_yaw, -1.5)


class StrikerStateTests(unittest.TestCase):
    """Verify striker approach/dribble transition conditions."""

    def _field_striker(self):
        return SimpleNamespace(
            field_width=5.55,
            field_length=10.0,
            goal_width=1.0,
            boundary_margin=0.15,
            wall_escape_margin=0.55,
            wall_escape_inward=1.0,
            endline_escape_backoff=1.0,
        )

    def test_close_ball_in_front_is_ready_to_dribble(self):
        _, _, qz, qw = quaternion_from_yaw(math.pi * 0.5)
        striker = SimpleNamespace(
            possession_distance=0.45,
            approach_target_tolerance=0.18,
            orbit_tolerance=0.18,
            self_pose=SimpleNamespace(
                pose=SimpleNamespace(
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
                )
            ),
        )
        self.assertTrue(
            StrikerNode._ready_to_dribble(
                striker,
                self_point=Point2D(0.0, -0.05),
                ball=Point2D(0.0, 0.0),
                approach_target=Point2D(0.0, -0.55),
            )
        )

    def test_close_to_ball_is_ready_to_dribble(self):
        _, _, qz, qw = quaternion_from_yaw(math.pi * 0.5)
        striker = SimpleNamespace(
            possession_distance=0.45,
            approach_target_tolerance=0.18,
            orbit_tolerance=0.18,
            self_pose=SimpleNamespace(
                pose=SimpleNamespace(
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
                )
            ),
        )
        self.assertTrue(
            StrikerNode._ready_to_dribble(
                striker,
                self_point=Point2D(0.0, -0.04),
                ball=Point2D(0.0, 0.0),
                approach_target=Point2D(0.0, -0.55),
            )
        )

    def test_far_from_ball_and_approach_target_is_not_ready_to_dribble(self):
        _, _, qz, qw = quaternion_from_yaw(math.pi * 0.5)
        striker = SimpleNamespace(
            possession_distance=0.45,
            approach_target_tolerance=0.18,
            orbit_tolerance=0.18,
            self_pose=SimpleNamespace(
                pose=SimpleNamespace(
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
                )
            ),
        )
        self.assertFalse(
            StrikerNode._ready_to_dribble(
                striker,
                self_point=Point2D(1.0, -1.0),
                ball=Point2D(0.0, 0.0),
                approach_target=Point2D(0.0, -0.55),
            )
        )

    def test_dribble_state_uses_release_hysteresis(self):
        _, _, qz, qw = quaternion_from_yaw(math.pi * 0.5)
        striker = SimpleNamespace(
            DRIBBLE="dribble",
            state="dribble",
            possession_distance=0.45,
            approach_target_tolerance=0.18,
            orbit_tolerance=0.18,
            dribble_release_distance=0.65,
            dribble_lateral_release=0.32,
            dribble_rear_release=0.12,
            self_pose=SimpleNamespace(
                pose=SimpleNamespace(
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
                )
            ),
        )

        self.assertTrue(
            StrikerNode._ready_to_dribble(
                striker,
                self_point=Point2D(0.0, -0.20),
                ball=Point2D(0.24, 0.0),
                approach_target=Point2D(0.0, -0.38),
            )
        )

    def test_approach_yaw_faces_the_ball(self):
        yaw = StrikerNode._approach_yaw(Point2D(0.0, -0.38), Point2D(0.0, 0.0))
        self.assertAlmostEqual(yaw, math.pi * 0.5, places=6)

    def test_midfield_kick_target_is_attack_goal(self):
        striker = self._field_striker()
        target = StrikerNode._kick_target(
            striker,
            ball=Point2D(0.0, 0.0),
            attack_goal=Point2D(0.0, 5.0),
        )

        self.assertAlmostEqual(target.x, 0.0)
        self.assertAlmostEqual(target.y, 5.0)

    def test_side_wall_kick_target_pulls_infield(self):
        striker = self._field_striker()
        target = StrikerNode._kick_target(
            striker,
            ball=Point2D(2.35, 0.0),
            attack_goal=Point2D(0.0, 5.0),
        )

        self.assertLess(target.x, 1.60)
        self.assertGreater(target.y, 0.0)

    def test_corner_kick_target_pulls_away_from_walls(self):
        striker = self._field_striker()
        target = StrikerNode._kick_target(
            striker,
            ball=Point2D(2.35, 4.55),
            attack_goal=Point2D(0.0, 5.0),
        )

        self.assertLess(target.x, 2.35)
        self.assertLess(target.y, 4.55)

    def test_central_goal_mouth_still_targets_attack_goal(self):
        striker = self._field_striker()
        target = StrikerNode._kick_target(
            striker,
            ball=Point2D(0.1, 4.55),
            attack_goal=Point2D(0.0, 5.0),
        )

        self.assertAlmostEqual(target.x, 0.0)
        self.assertAlmostEqual(target.y, 5.0)

    def test_aligned_dribble_triggers_fast_kick(self):
        class FakeMove:
            def __init__(self):
                self.motion_id = None
                self.speed = None

            def change_motion_id(self, motion_id):
                self.motion_id = motion_id

            def change_speed(self, speed_x, speed_y, speed_yaw):
                self.speed = (speed_x, speed_y, speed_yaw)

        _, _, qz, qw = quaternion_from_yaw(math.pi * 0.5)
        striker = StrikerNode.__new__(StrikerNode)
        striker.self_pose = SimpleNamespace(
            pose=SimpleNamespace(
                orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
            )
        )
        striker.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))
        striker.move = FakeMove()
        striker.ball_twist = None
        striker.attack_motion_id = 305
        striker.orbit_radius = 0.38
        striker.orbit_tolerance = 0.18
        striker.possession_distance = 0.45
        striker.dribble_speed = 0.65
        striker.kick_speed = 0.9
        striker.kick_duration = 0.45
        striker.kick_min_speed = 0.65
        striker.kick_lateral_tolerance = 0.24
        striker.kick_lateral_gain = 0.8
        striker.dribble_release_distance = 0.65
        striker.dribble_lateral_release = 0.32
        striker.dribble_rear_release = 0.12
        striker.approach_gain = 0.8
        striker.lateral_gain = 0.8
        striker.yaw_gain = 1.2
        striker.align_yaw_tolerance = 0.35
        striker._dribble_phase = "align"
        striker._kick_start_time = None
        striker.debug_log_interval = 0.0
        striker.field_width = 5.55
        striker.field_length = 10.0
        striker.boundary_margin = 0.15

        StrikerNode._dribble_toward_goal(
            striker,
            self_point=Point2D(0.0, -0.38),
            ball=Point2D(0.0, 0.0),
            attack_goal=Point2D(0.0, 5.0),
        )

        self.assertEqual(striker.move.motion_id, 305)
        self.assertEqual(striker._dribble_phase, "kick")
        self.assertGreater(striker.move.speed[0], striker.dribble_speed)
        self.assertLessEqual(striker.move.speed[0], striker.kick_speed)

    def test_lost_ball_aborts_dribble(self):
        class FakeMove:
            def __init__(self):
                self.motion_id = None
                self.speed = None

            def change_motion_id(self, motion_id):
                self.motion_id = motion_id

            def change_speed(self, speed_x, speed_y, speed_yaw):
                self.speed = (speed_x, speed_y, speed_yaw)

        _, _, qz, qw = quaternion_from_yaw(math.pi * 0.5)
        striker = StrikerNode.__new__(StrikerNode)
        striker.self_pose = SimpleNamespace(
            pose=SimpleNamespace(
                orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
            )
        )
        striker.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))
        striker.move = FakeMove()
        striker.ball_twist = None
        striker.attack_motion_id = 305
        striker.orbit_radius = 0.38
        striker.orbit_tolerance = 0.18
        striker.possession_distance = 0.45
        striker.dribble_speed = 0.65
        striker.kick_speed = 0.9
        striker.kick_duration = 0.45
        striker.kick_min_speed = 0.65
        striker.kick_lateral_tolerance = 0.24
        striker.kick_lateral_gain = 0.8
        striker.dribble_release_distance = 0.65
        striker.dribble_lateral_release = 0.32
        striker.dribble_rear_release = 0.12
        striker.approach_gain = 0.8
        striker.lateral_gain = 0.8
        striker.yaw_gain = 1.2
        striker.align_yaw_tolerance = 0.35
        striker._dribble_phase = "kick"
        striker._kick_start_time = 0.0
        striker.debug_log_interval = 0.0
        striker.field_width = 5.55
        striker.field_length = 10.0
        striker.boundary_margin = 0.15

        StrikerNode._dribble_toward_goal(
            striker,
            self_point=Point2D(0.0, -0.38),
            ball=Point2D(0.75, 0.0),
            attack_goal=Point2D(0.0, 5.0),
        )

        self.assertEqual(striker._dribble_phase, "align")
        self.assertEqual(striker.move.speed, (0.0, 0.0, 0.0))

    def test_unstable_kick_returns_to_align_control(self):
        class FakeMove:
            def __init__(self):
                self.speed = None

            def change_motion_id(self, motion_id):
                pass

            def change_speed(self, speed_x, speed_y, speed_yaw):
                self.speed = (speed_x, speed_y, speed_yaw)

        _, _, qz, qw = quaternion_from_yaw(math.pi * 0.5)
        striker = StrikerNode.__new__(StrikerNode)
        striker.self_pose = SimpleNamespace(
            pose=SimpleNamespace(
                orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
            )
        )
        striker.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0.2e9))
        striker.move = FakeMove()
        striker.ball_twist = SimpleNamespace(
            twist=SimpleNamespace(linear=SimpleNamespace(x=0.0, y=-0.2))
        )
        striker.attack_motion_id = 305
        striker.orbit_radius = 0.38
        striker.orbit_tolerance = 0.18
        striker.possession_distance = 0.45
        striker.dribble_speed = 0.65
        striker.kick_speed = 0.9
        striker.kick_duration = 0.45
        striker.kick_min_speed = 0.65
        striker.kick_lateral_tolerance = 0.24
        striker.kick_lateral_gain = 0.8
        striker.dribble_release_distance = 0.65
        striker.dribble_lateral_release = 0.32
        striker.dribble_rear_release = 0.12
        striker.approach_gain = 0.8
        striker.lateral_gain = 0.8
        striker.yaw_gain = 1.2
        striker.align_yaw_tolerance = 0.35
        striker._dribble_phase = "kick"
        striker._kick_start_time = 0.0
        striker.debug_log_interval = 0.0
        striker.field_width = 5.55
        striker.field_length = 10.0
        striker.boundary_margin = 0.15

        StrikerNode._dribble_toward_goal(
            striker,
            self_point=Point2D(0.0, -0.38),
            ball=Point2D(0.0, 0.0),
            attack_goal=Point2D(0.0, 5.0),
        )

        self.assertEqual(striker._dribble_phase, "align")
        self.assertIsNotNone(striker.move.speed)
        self.assertLessEqual(striker.move.speed[0], striker.dribble_speed)


class StrikerLegacyTests(unittest.TestCase):
    """Verify low-cost legacy striker helpers used without A*."""

    def _pose(self, x: float, y: float, yaw: float = 0.0):
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)),
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y),
                orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
            ),
        )

    def _legacy(self):
        _, _, qz, qw = quaternion_from_yaw(math.pi * 0.5)
        striker = StrikerLegacyNode.__new__(StrikerLegacyNode)
        striker.field_width = 5.55
        striker.field_length = 10.0
        striker.goal_width = 1.0
        striker.boundary_margin = 0.15
        striker.wall_escape_margin = 0.55
        striker.wall_escape_inward = 1.0
        striker.endline_escape_backoff = 1.0
        striker.gk_block_corridor = 0.4
        striker.gk_dodge_offset = 0.55
        striker._opponent_poses = {}
        striker.obstacle_corridor = 0.48
        striker.obstacle_influence = 0.85
        striker.obstacle_repulse_max = 0.6
        striker.obstacle_slowdown_gain = 0.45
        striker.obstacle_slowdown_min = 0.45
        striker.dodge_hold_time = 0.45
        striker._dodge_side = 0.0
        striker._dodge_until = 0.0
        striker.ball_path_keepout = 0.40
        striker.ball_path_clearance = 0.56
        striker.ball_avoid_hold_time = 0.40
        striker._ball_avoid_side = 0.0
        striker._ball_avoid_until = 0.0
        striker.recover_infield_margin = 0.35
        striker.dribble_orbit_side_offset = 0.58
        striker.dribble_handoff_lateral = 0.30
        striker.dribble_gate_behind_offset = 0.30
        striker.dribble_behind_min = 0.18
        striker.latency_compensation = 0.12
        striker.max_prediction_time = 0.30
        striker.ball_twist = None
        striker.ball_pose = None
        striker.state = StrikerLegacyNode.APPROACH
        striker.possession_distance = 0.45
        striker._align_yaw_tol = 0.26
        striker._kick_duration = 0.40
        striker._kick_speed = 0.90
        striker._kick_min_speed = 0.65
        striker.kick_contact_distance = 0.38
        striker.kick_lateral_tolerance = 0.24
        striker.kick_lateral_gain = 0.8
        striker.orbit_radius = 0.38
        striker.orbit_tol = 0.14
        striker.dribble_release_distance = 0.65
        striker.dribble_lateral_release = 0.32
        striker.dribble_rear_release = 0.12
        striker.debug_log_interval = 0.0
        striker._last_debug_log_time = 0.0
        striker.self_pose = SimpleNamespace(
            pose=SimpleNamespace(
                orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
            )
        )
        striker.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
        return striker

    def test_latency_prediction_advances_ball_and_clamps_horizon(self):
        striker = self._legacy()
        striker.ball_pose = self._pose(0.0, 0.0)
        striker.ball_pose.header.stamp.sec = 9
        striker.ball_pose.header.stamp.nanosec = 700_000_000
        striker.ball_twist = SimpleNamespace(
            twist=SimpleNamespace(linear=SimpleNamespace(x=0.0, y=1.0))
        )

        predicted = StrikerLegacyNode._predicted_ball(striker, Point2D(0.0, 0.0))

        self.assertAlmostEqual(predicted.x, 0.0)
        self.assertAlmostEqual(predicted.y, 0.30)

    def test_wall_escape_target_pulls_corner_ball_infield(self):
        striker = self._legacy()
        target = StrikerLegacyNode._wall_escape_target(
            striker,
            ball=Point2D(2.35, 4.55),
            goal=Point2D(0.0, 5.0),
        )

        self.assertLess(target.x, 2.35)
        self.assertLess(target.y, 4.55)

    def test_goal_mouth_ball_still_targets_goal(self):
        striker = self._legacy()
        target = StrikerLegacyNode._wall_escape_target(
            striker,
            ball=Point2D(0.1, 4.55),
            goal=Point2D(0.0, 5.0),
        )

        self.assertAlmostEqual(target.x, 0.0)
        self.assertAlmostEqual(target.y, 5.0)

    def test_blocked_kick_target_sidesteps_lane(self):
        striker = self._legacy()
        striker._opponent_poses = {"opponent_1": self._pose(0.0, 2.0)}

        target = StrikerLegacyNode._pick_kick_target(
            striker,
            ball=Point2D(0.0, 0.0),
            goal=Point2D(0.0, 5.0),
        )

        self.assertGreater(abs(target.x), 0.20)
        self.assertLessEqual(abs(target.x), striker.field_width * 0.5)

    def test_obstacle_avoidance_adds_lateral_bias_and_slows(self):
        striker = self._legacy()
        striker._opponent_poses = {"opponent_1": self._pose(0.2, 0.4)}

        speed_scale, lateral_bias = StrikerLegacyNode._obstacle_avoidance(
            striker,
            me=Point2D(0.0, 0.0),
            target=Point2D(0.0, 2.0),
            yaw=math.pi * 0.5,
        )

        self.assertLess(speed_scale, 1.0)
        self.assertGreater(abs(lateral_bias), 0.05)

    def test_approach_target_diverts_around_ball_on_crossing_path(self):
        striker = self._legacy()

        target = StrikerLegacyNode._ball_safe_target(
            striker,
            me=Point2D(0.0, -1.0),
            target=Point2D(0.0, 1.0),
            ball=Point2D(0.0, 0.0),
        )

        self.assertGreater(abs(target.x), 0.40)
        self.assertAlmostEqual(target.y, 0.0, places=6)

    def test_dribble_state_uses_recoverable_hysteresis(self):
        striker = self._legacy()
        striker.state = StrikerLegacyNode.DRIBBLE

        self.assertTrue(
            StrikerLegacyNode._ready_to_dribble(
                striker,
                me=Point2D(0.0, -0.20),
                ball=Point2D(0.24, 0.0),
                goal=Point2D(0.0, 5.0),
            )
        )

    def test_legacy_kickoff_spacing_can_enter_dribble(self):
        striker = self._legacy()
        striker.possession_distance = 0.55
        striker.self_pose = self._pose(0.0, 0.5, -math.pi * 0.5)

        self.assertTrue(
            StrikerLegacyNode._ready_to_dribble(
                striker,
                me=Point2D(0.0, 0.5),
                ball=Point2D(0.0, 0.0),
                goal=Point2D(0.0, -5.0),
            )
        )

    def test_legacy_side_ball_exits_dribble_before_pushing(self):
        striker = self._legacy()
        striker.state = StrikerLegacyNode.DRIBBLE
        striker.possession_distance = 0.55
        striker.dribble_release_distance = 0.84
        striker.dribble_lateral_release = 0.46
        striker.dribble_rear_release = 0.22
        striker.orbit_tol = 0.18
        striker.kick_lateral_tolerance = 0.24
        striker.dribble_handoff_lateral = 0.30
        striker.self_pose = self._pose(-0.45, 0.05, -math.pi * 0.5)

        self.assertFalse(
            StrikerLegacyNode._ready_to_dribble(
                striker,
                me=Point2D(-0.45, 0.05),
                ball=Point2D(0.0, 0.0),
                goal=Point2D(0.0, -5.0),
            )
        )

    def test_legacy_kick_waits_until_contact_distance(self):
        class FakeMove:
            def __init__(self):
                self.motion_id = None
                self.speed = None

            def change_motion_id(self, motion_id):
                self.motion_id = motion_id

            def change_speed(self, speed_x, speed_y, speed_yaw):
                self.speed = (speed_x, speed_y, speed_yaw)

        striker = self._legacy()
        striker.move = FakeMove()
        striker.state = StrikerLegacyNode.DRIBBLE
        striker._dribble_sub = StrikerLegacyNode._ALIGN
        striker._kick_start_time = None
        striker.attack_motion_id = 305
        striker.dribble_speed = 0.65
        striker.approach_gain = 0.8
        striker.lateral_gain = 0.8
        striker.yaw_gain = 1.2
        striker.kick_contact_distance = 0.38
        striker.self_pose = self._pose(0.0, 0.45, -math.pi * 0.5)

        StrikerLegacyNode._dribble_step(
            striker,
            me=Point2D(0.0, 0.45),
            ball=Point2D(0.0, 0.0),
            goal=Point2D(0.0, -5.0),
        )

        self.assertEqual(striker._dribble_sub, StrikerLegacyNode._ALIGN)
        self.assertIsNotNone(striker.move.speed)

    def test_legacy_kick_triggers_at_contact_distance(self):
        class FakeMove:
            def __init__(self):
                self.motion_id = None
                self.speed = None

            def change_motion_id(self, motion_id):
                self.motion_id = motion_id

            def change_speed(self, speed_x, speed_y, speed_yaw):
                self.speed = (speed_x, speed_y, speed_yaw)

        striker = self._legacy()
        striker.move = FakeMove()
        striker.state = StrikerLegacyNode.DRIBBLE
        striker._dribble_sub = StrikerLegacyNode._ALIGN
        striker._kick_start_time = None
        striker.attack_motion_id = 305
        striker.dribble_speed = 0.65
        striker.approach_gain = 0.8
        striker.lateral_gain = 0.8
        striker.yaw_gain = 1.2
        striker.kick_contact_distance = 0.38
        striker.get_logger = lambda: SimpleNamespace(info=lambda *_args, **_kwargs: None)
        striker.self_pose = self._pose(0.0, 0.35, -math.pi * 0.5)

        StrikerLegacyNode._dribble_step(
            striker,
            me=Point2D(0.0, 0.35),
            ball=Point2D(0.0, 0.0),
            goal=Point2D(0.0, -5.0),
        )

        self.assertEqual(striker._dribble_sub, StrikerLegacyNode._KICK)

    def test_legacy_dribble_prefers_side_step_when_not_behind_ball(self):
        class FakeMove:
            def __init__(self):
                self.motion_id = None
                self.speed = None

            def change_motion_id(self, motion_id):
                self.motion_id = motion_id

            def change_speed(self, speed_x, speed_y, speed_yaw):
                self.speed = (speed_x, speed_y, speed_yaw)

        striker = self._legacy()
        striker.move = FakeMove()
        striker.state = StrikerLegacyNode.DRIBBLE
        striker._dribble_sub = StrikerLegacyNode._ALIGN
        striker._kick_start_time = None
        striker.attack_motion_id = 305
        striker.dribble_speed = 0.65
        striker.approach_gain = 0.8
        striker.lateral_gain = 0.8
        striker.yaw_gain = 1.2
        striker.kick_lateral_gain = 0.8
        striker.kick_lateral_tolerance = 0.24
        striker.orbit_radius = 0.38
        striker.orbit_tol = 0.18
        striker.possession_distance = 0.55
        striker.dribble_release_distance = 0.84
        striker.dribble_lateral_release = 0.46
        striker.dribble_rear_release = 0.22
        striker.dribble_orbit_side_offset = 0.58
        striker.dribble_behind_min = 0.18
        striker.debug_log_interval = 0.0
        striker._last_debug_log_time = 0.0
        striker.self_pose = self._pose(-0.45, 0.05, -math.pi * 0.5)
        striker.ball_pose = self._pose(0.00, 0.00, 0.0)
        striker.attack_goal_pose = self._pose(0.00, -5.0, 0.0)

        StrikerLegacyNode._dribble_step(
            striker,
            me=Point2D(-0.45, 0.05),
            ball=Point2D(0.00, 0.00),
            goal=Point2D(0.00, -5.0),
        )

        self.assertEqual(striker._dribble_sub, StrikerLegacyNode._ALIGN)
        self.assertIsNotNone(striker.move.speed)
        self.assertLessEqual(striker.move.speed[0], 0.0)
        self.assertGreater(abs(striker.move.speed[1]), 0.0)

    def test_legacy_publish_state_includes_render_intent(self):
        class FakePublisher:
            def __init__(self):
                self.last = None

            def publish(self, msg):
                self.last = msg

        striker = self._legacy()
        striker._team = "team_b"
        striker._robot_id = "robot_1"
        striker.state = StrikerLegacyNode.APPROACH
        striker._role_state_pub = FakePublisher()
        striker._intent_pub = FakePublisher()
        striker.self_pose = self._pose(0.0, 0.0, math.pi * 0.5)
        striker.ball_pose = self._pose(0.0, 0.5)

        StrikerLegacyNode._publish_state(striker)

        self.assertEqual(striker._role_state_pub.last.data, StrikerLegacyNode.APPROACH)
        payload = json.loads(striker._intent_pub.last.data)
        self.assertEqual(payload["role"], "striker")
        self.assertEqual(payload["state"], StrikerLegacyNode.APPROACH)
        self.assertEqual(payload["controller"], "legacy")


class PathFollowerTests(unittest.TestCase):
    """Verify that the follower uses the striker target orientation."""

    def test_target_orientation_drives_final_yaw(self):
        class FakeMove:
            def __init__(self):
                self.motion_id = None
                self.speed = None
                self.stopped = False

            def change_motion_id(self, motion_id):
                self.motion_id = motion_id

            def change_speed(self, speed_x, speed_y, speed_yaw):
                self.speed = (speed_x, speed_y, speed_yaw)

            def stop(self, use_stop_motion=False):
                self.stopped = True

        _, _, qz, qw = quaternion_from_yaw(math.pi * 0.5)
        follower = PathFollowerNode.__new__(PathFollowerNode)
        follower.enabled = True
        follower.move = FakeMove()
        follower.walk_motion_id = 303
        follower.position_gain = 0.8
        follower.lateral_gain = 0.8
        follower.yaw_gain = 1.2
        follower.waypoint_tolerance = 0.18
        follower.goal_tolerance = 0.18
        follower.lookahead_points = 2
        follower.self_pose = SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
        follower.target_pose = SimpleNamespace(
            pose=SimpleNamespace(
                orientation=SimpleNamespace(x=0.0, y=0.0, z=qz, w=qw),
            )
        )
        follower.path = [Point2D(1.0, 0.0), Point2D(2.0, 0.0)]
        follower.waypoint_index = 0

        PathFollowerNode._timer_callback(follower)

        self.assertEqual(follower.move.motion_id, 303)
        self.assertIsNotNone(follower.move.speed)
        self.assertAlmostEqual(follower.move.speed[2], 1.2 * math.pi * 0.5, places=6)


class MainBuildNodesTests(unittest.TestCase):
    """Verify the runtime node graph needed by each role."""

    def test_striker_build_includes_path_planner_and_follower(self):
        import main as app_main

        config = {
            "role": "striker",
            "sim": {"team": "team_a", "robot_id": "robot_1"},
        }
        locator = object()
        move = object()
        path_planner = object()
        path_follower = object()
        striker = object()

        with (
            patch.object(app_main, "LocatorNode", return_value=locator) as locator_cls,
            patch.object(app_main, "MoveCommander", return_value=move) as move_cls,
            patch.object(app_main, "PathPlannerNode", return_value=path_planner) as planner_cls,
            patch.object(app_main, "PathFollowerNode", return_value=path_follower) as follower_cls,
            patch.object(app_main, "StrikerNode", return_value=striker) as striker_cls,
        ):
            nodes, returned_move = app_main.build_nodes(config)

        self.assertEqual(nodes, [locator, move, path_planner, path_follower, striker])
        self.assertIs(returned_move, move)
        locator_cls.assert_called_once_with(config, name="locator_team_a_robot_1")
        move_cls.assert_called_once_with(config=config, name="move_commander_team_a_robot_1")
        planner_cls.assert_called_once_with(config, name="path_planner_team_a_robot_1")
        follower_cls.assert_called_once_with(config, move, name="path_follower_team_a_robot_1")
        striker_cls.assert_called_once_with(
            config,
            move,
            path_follower=path_follower,
            name="striker_team_a_robot_1",
        )

    def test_legacy_striker_build_skips_path_planner(self):
        import main as app_main

        config = {
            "role": "legacy_striker",
            "sim": {"team": "team_b", "robot_id": "robot_1"},
        }
        locator = object()
        move = object()
        striker = object()

        with (
            patch.object(app_main, "LocatorNode", return_value=locator) as locator_cls,
            patch.object(app_main, "MoveCommander", return_value=move) as move_cls,
            patch.object(app_main, "PathPlannerNode") as planner_cls,
            patch.object(app_main, "PathFollowerNode") as follower_cls,
            patch.object(app_main, "StrikerLegacyNode", return_value=striker) as striker_cls,
        ):
            nodes, returned_move = app_main.build_nodes(config)

        self.assertEqual(nodes, [locator, move, striker])
        self.assertIs(returned_move, move)
        locator_cls.assert_called_once_with(config, name="locator_team_b_robot_1")
        move_cls.assert_called_once_with(config=config, name="move_commander_team_b_robot_1")
        planner_cls.assert_not_called()
        follower_cls.assert_not_called()
        striker_cls.assert_called_once_with(
            config,
            move,
            name="legacy_striker_team_b_robot_1",
        )

    def test_policy_striker_build_skips_path_planner(self):
        import main as app_main

        config = {
            "role": "policy_striker",
            "sim": {"team": "team_a", "robot_id": "robot_1"},
        }
        locator = object()
        move = object()
        striker = object()

        with (
            patch.object(app_main, "LocatorNode", return_value=locator),
            patch.object(app_main, "MoveCommander", return_value=move),
            patch.object(app_main, "PathPlannerNode") as planner_cls,
            patch.object(app_main, "PathFollowerNode") as follower_cls,
            patch.object(app_main, "StrikerPolicyNode", return_value=striker) as striker_cls,
        ):
            nodes, returned_move = app_main.build_nodes(config)

        self.assertEqual(nodes, [locator, move, striker])
        self.assertIs(returned_move, move)
        planner_cls.assert_not_called()
        follower_cls.assert_not_called()
        striker_cls.assert_called_once_with(
            config,
            move,
            name="policy_striker_team_a_robot_1",
        )


class BallPredictorTests(unittest.TestCase):
    """Verify goalkeeper target prediction against a moving goal rigid."""

    def test_intersection_with_defense_goal_line(self):
        target = predict_guard_target(
            ball=Point2D(0.2, 0.0),
            velocity=Point2D(0.0, -1.0),
            defense_goal=Point2D(0.0, -5.0),
            goal_width=1.5,
            guard_depth=0.45,
            guard_margin=0.12,
            min_ball_speed=0.08,
        )
        self.assertAlmostEqual(target.x, 0.2, places=6)
        self.assertAlmostEqual(target.y, -4.55, places=6)

    def test_target_is_clamped_inside_goal_mouth(self):
        target = predict_guard_target(
            ball=Point2D(2.0, 0.0),
            velocity=Point2D(0.0, -1.0),
            defense_goal=Point2D(0.0, -5.0),
            goal_width=1.5,
            guard_depth=0.45,
            guard_margin=0.12,
            min_ball_speed=0.08,
        )
        self.assertLessEqual(target.x, 0.63)
        self.assertAlmostEqual(target.y, -4.55, places=6)


if __name__ == "__main__":
    unittest.main()
