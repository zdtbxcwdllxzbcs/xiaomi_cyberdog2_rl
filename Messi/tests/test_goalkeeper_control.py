import math
import unittest
from types import SimpleNamespace

from src.lib.geometry import Point2D
from src.lib.goalkeeper_control import (
    ACTIVE_INTERCEPT,
    LATERAL_DEFEND,
    BallTracker,
    GoalkeeperControlConfig,
    VelocityCommand,
    VelocitySmoother,
    damped_speed,
    goalkeeper_plan,
    lateral_stance_yaw,
    lateral_push_command,
    predict_ball,
    recover_avoidance_target,
    segment_distance_to_point,
)


def _stamp(seconds: float):
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    return SimpleNamespace(sec=sec, nanosec=nanosec)


class GoalkeeperTwoStateControlTests(unittest.TestCase):
    def _config(self):
        return GoalkeeperControlConfig(
            goal_width=1.0,
            guard_depth=0.45,
            guard_margin=0.12,
            guard_width=2.8,
            field_length=10.0,
            field_width=5.55,
            field_margin=0.40,
            activity_depth=3.8,
            activity_width=5.2,
            ball_control_distance=0.45,
            lost_ball_y_margin=0.12,
            clear_push_distance=1.6,
            clear_inward_pull=0.9,
            latency_bias=0.04,
            prediction_horizon=0.22,
            prediction_max_distance=0.30,
            lateral_prediction_horizon=3.0,
        )

    def test_penalty_area_enters_active_clear(self):
        plan = goalkeeper_plan(
            ball=Point2D(1.20, -4.05),
            ball_velocity=Point2D(0.0, -0.2),
            defense_goal=Point2D(0.0, -5.0),
            config=self._config(),
            self_point=Point2D(0.0, -4.55),
        )

        self.assertEqual(plan.state, ACTIVE_INTERCEPT)
        self.assertEqual(plan.phase, "approach_clear")
        self.assertEqual(plan.reason, "inside_activity_zone")
        self.assertAlmostEqual(plan.yaw, lateral_stance_yaw())
        self.assertGreater(plan.target.y, -4.55)

    def test_deep_half_inside_activity_enters_active_intercept(self):
        plan = goalkeeper_plan(
            ball=Point2D(0.55, -3.25),
            ball_velocity=Point2D(0.0, -1.5),
            defense_goal=Point2D(0.0, -5.0),
            config=self._config(),
            self_point=Point2D(0.0, -4.55),
        )

        self.assertEqual(plan.state, ACTIVE_INTERCEPT)
        self.assertAlmostEqual(plan.yaw, lateral_stance_yaw())
        self.assertEqual(plan.reason, "inside_activity_zone")

    def test_midfield_ball_does_not_trigger_active_intercept(self):
        plan = goalkeeper_plan(
            ball=Point2D(0.1, -1.00),
            ball_velocity=Point2D(0.0, -2.0),
            defense_goal=Point2D(0.0, -5.0),
            config=self._config(),
            self_point=Point2D(0.0, -4.55),
        )

        self.assertEqual(plan.state, LATERAL_DEFEND)
        self.assertEqual(plan.phase, "guard")
        self.assertEqual(plan.reason, "outside_activity_zone")

    def test_outside_activity_guard_returns_to_goal_center_unless_incoming(self):
        plan = goalkeeper_plan(
            ball=Point2D(2.2, 2.5),
            ball_velocity=Point2D(0.0, 0.0),
            defense_goal=Point2D(0.0, -5.0),
            config=self._config(),
            self_point=Point2D(-1.0, -4.0),
        )

        self.assertEqual(plan.state, LATERAL_DEFEND)
        self.assertAlmostEqual(plan.target.x, 0.0)

    def test_incoming_outside_activity_uses_guard_line_intersection(self):
        plan = goalkeeper_plan(
            ball=Point2D(1.0, -1.0),
            ball_velocity=Point2D(-0.4, -3.0),
            defense_goal=Point2D(0.0, -5.0),
            config=self._config(),
            self_point=Point2D(0.0, -4.55),
        )

        self.assertEqual(plan.state, LATERAL_DEFEND)
        self.assertLess(plan.target.x, 1.0)

    def test_side_clear_pulls_inward_and_forward(self):
        plan = goalkeeper_plan(
            ball=Point2D(2.10, -3.70),
            ball_velocity=Point2D(0.0, 0.0),
            defense_goal=Point2D(0.0, -5.0),
            config=self._config(),
            self_point=Point2D(2.05, -3.85),
        )

        self.assertEqual(plan.state, ACTIVE_INTERCEPT)
        self.assertIn(plan.phase, {"push_clear", "dribble_clear"})
        self.assertLess(plan.target.x, 2.10)
        self.assertGreater(plan.target.y, -3.70)

    def test_active_approach_targets_current_ball_before_contact(self):
        plan = goalkeeper_plan(
            ball=Point2D(0.45, -2.0),
            ball_velocity=Point2D(0.0, 0.0),
            defense_goal=Point2D(0.0, -5.0),
            config=self._config(),
            self_point=Point2D(-0.6, -4.0),
        )

        self.assertEqual(plan.phase, "approach_clear")
        self.assertAlmostEqual(plan.target.x, 0.45)
        self.assertAlmostEqual(plan.target.y, -2.0)

    def test_clear_target_stays_inside_field_margin(self):
        cfg = self._config()
        plan = goalkeeper_plan(
            ball=Point2D(2.55, -3.8),
            ball_velocity=Point2D(0.0, 0.0),
            defense_goal=Point2D(0.0, -5.0),
            config=cfg,
            self_point=Point2D(2.55, -3.85),
        )

        self.assertLessEqual(plan.target.x, cfg.field_width * 0.5 - cfg.field_margin + 1e-6)

    def test_ball_behind_self_recovers_to_guard(self):
        plan = goalkeeper_plan(
            ball=Point2D(0.15, -4.25),
            ball_velocity=Point2D(0.0, 0.0),
            defense_goal=Point2D(0.0, -5.0),
            config=self._config(),
            self_point=Point2D(0.10, -4.00),
        )

        self.assertEqual(plan.state, LATERAL_DEFEND)
        self.assertEqual(plan.phase, "recover_guard")
        self.assertTrue(plan.ball_behind)
        self.assertAlmostEqual(plan.target.x, 0.0)
        self.assertAlmostEqual(plan.target.y, -4.55)

    def test_ball_prediction_uses_age_bias_and_distance_cap(self):
        predicted = predict_ball(
            Point2D(0.0, -3.0),
            Point2D(3.0, -4.0),
            self._config(),
            sample_age=0.20,
        )

        self.assertLessEqual(math.hypot(predicted.x, predicted.y + 3.0), 0.30 + 1e-6)
        self.assertGreater(predicted.x, 0.0)
        self.assertLess(predicted.y, -3.0)

    def test_legacy_active_zone_fields_still_map_to_activity(self):
        cfg = GoalkeeperControlConfig(
            goal_width=1.0,
            guard_depth=0.45,
            guard_width=0.0,
            activity_depth=0.0,
            activity_width=0.0,
            active_zone_depth=2.5,
            active_zone_width=5.55,
        )
        plan = goalkeeper_plan(
            ball=Point2D(0.0, -3.0),
            ball_velocity=Point2D(0.0, 0.0),
            defense_goal=Point2D(0.0, -5.0),
            config=cfg,
            self_point=Point2D(0.0, -4.55),
        )

        self.assertEqual(plan.state, ACTIVE_INTERCEPT)

    def test_legacy_keeper_zone_fields_still_map_to_activity(self):
        cfg = GoalkeeperControlConfig(
            goal_width=1.0,
            guard_depth=0.45,
            guard_width=0.0,
            activity_depth=0.0,
            activity_width=0.0,
            keeper_zone_depth=3.0,
            keeper_zone_width=4.0,
        )
        plan = goalkeeper_plan(
            ball=Point2D(1.8, -2.4),
            ball_velocity=Point2D(0.0, 0.0),
            defense_goal=Point2D(0.0, -5.0),
            config=cfg,
            self_point=Point2D(0.0, -4.55),
        )

        self.assertEqual(plan.state, ACTIVE_INTERCEPT)

    def test_ball_tracker_uses_valid_source_stamp_age(self):
        tracker = BallTracker()
        tracker.record_pose(Point2D(0.0, 0.0), _stamp(9.80), received_time=9.90)
        tracker.record_pose(Point2D(0.10, 0.0), _stamp(9.90), received_time=9.95)

        prediction = tracker.predict(10.0, self._config())

        self.assertEqual(prediction.velocity_source, "history")
        self.assertAlmostEqual(prediction.sample_age, 0.10, places=6)
        self.assertGreater(prediction.predicted.x, 0.10)

    def test_ball_tracker_falls_back_to_received_age_when_stamp_incomparable(self):
        tracker = BallTracker()
        tracker.record_pose(Point2D(0.0, 0.0), _stamp(1000.0), received_time=9.90)
        tracker.record_pose(Point2D(0.05, 0.0), _stamp(1000.1), received_time=9.95)

        prediction = tracker.predict(10.0, self._config())

        self.assertAlmostEqual(prediction.sample_age, 0.05, places=6)

    def test_ball_tracker_rejects_unsynced_twist(self):
        tracker = BallTracker()
        tracker.record_pose(Point2D(0.0, 0.0), _stamp(9.80), received_time=9.80)
        tracker.record_pose(Point2D(0.10, 0.0), _stamp(9.90), received_time=9.90)

        prediction = tracker.predict(
            10.0,
            self._config(),
            twist=Point2D(0.0, 5.0),
            twist_stamp=_stamp(8.0),
        )

        self.assertEqual(prediction.velocity_source, "history")
        self.assertGreater(prediction.predicted.x, 0.10)
        self.assertAlmostEqual(prediction.predicted.y, 0.0)

    def test_ball_tracker_uses_synced_twist(self):
        tracker = BallTracker()
        tracker.record_pose(Point2D(0.0, 0.0), _stamp(9.90), received_time=9.90)

        prediction = tracker.predict(
            10.0,
            self._config(),
            twist=Point2D(0.0, 1.0),
            twist_stamp=_stamp(9.92),
        )

        self.assertEqual(prediction.velocity_source, "twist")
        self.assertGreater(prediction.predicted.y, 0.0)

    def test_ball_tracker_ignores_low_speed_jitter(self):
        tracker = BallTracker()
        tracker.record_pose(Point2D(0.0, 0.0), _stamp(9.80), received_time=9.80)
        tracker.record_pose(Point2D(0.002, -0.001), _stamp(9.90), received_time=9.90)

        prediction = tracker.predict(10.0, self._config())

        self.assertEqual(prediction.velocity_source, "history_slow")
        self.assertEqual(prediction.displacement, 0.0)
        self.assertAlmostEqual(prediction.predicted.x, 0.002)
        self.assertAlmostEqual(prediction.predicted.y, -0.001)

    def test_ball_tracker_rejects_direction_reversal_jitter(self):
        tracker = BallTracker()
        tracker.record_pose(Point2D(0.0, 0.0), _stamp(9.80), received_time=9.80)
        tracker.record_pose(Point2D(0.20, 0.0), _stamp(9.90), received_time=9.90)
        tracker.record_pose(Point2D(0.02, 0.0), _stamp(10.00), received_time=10.00)

        prediction = tracker.predict(10.05, self._config())

        self.assertEqual(prediction.velocity_source, "none")
        self.assertEqual(prediction.displacement, 0.0)
        self.assertAlmostEqual(prediction.predicted.x, 0.02)

    def test_velocity_smoother_limits_acceleration_and_jerk(self):
        dt = 0.1
        max_accel = 0.85
        max_jerk = 2.4
        smoother = VelocitySmoother(
            max_linear_accel=max_accel,
            max_linear_jerk=max_jerk,
            max_angular_accel=1.8,
            max_angular_jerk=5.0,
        )
        prev_v = 0.0
        prev_accel = 0.0

        for _ in range(80):
            command = smoother.step(VelocityCommand(1.0, 0.0, 0.0), dt)
            accel = (command.vx - prev_v) / dt
            jerk = (accel - prev_accel) / dt
            self.assertLessEqual(abs(accel), max_accel + 1e-6)
            self.assertLessEqual(abs(jerk), max_jerk + 1e-6)
            prev_v = command.vx
            prev_accel = accel

    def test_velocity_smoother_limits_planar_vector_magnitude(self):
        dt = 0.1
        max_accel = 0.85
        max_jerk = 2.4
        smoother = VelocitySmoother(
            max_linear_accel=max_accel,
            max_linear_jerk=max_jerk,
            max_angular_accel=1.8,
            max_angular_jerk=5.0,
        )
        prev = VelocityCommand(0.0, 0.0, 0.0)
        prev_accel = Point2D(0.0, 0.0)

        for _ in range(80):
            command = smoother.step(VelocityCommand(1.0, 1.0, 0.0), dt)
            accel = Point2D((command.vx - prev.vx) / dt, (command.vy - prev.vy) / dt)
            jerk = Point2D((accel.x - prev_accel.x) / dt, (accel.y - prev_accel.y) / dt)
            self.assertLessEqual(math.hypot(accel.x, accel.y), max_accel + 1e-6)
            self.assertLessEqual(math.hypot(jerk.x, jerk.y), max_jerk + 1e-6)
            prev = command
            prev_accel = accel

    def test_lateral_push_command_closes_side_contact_loop(self):
        cfg = self._config()
        defense_goal = Point2D(0.0, -5.0)

        centered = lateral_push_command(Point2D(0.10, cfg.sweep_side_offset), defense_goal, cfg)
        too_far_ahead = lateral_push_command(Point2D(0.10, 0.75), defense_goal, cfg)
        falling_behind = lateral_push_command(Point2D(0.10, -0.10), defense_goal, cfg)

        self.assertGreater(centered.vy, 0.0)
        self.assertGreater(centered.vx, 0.0)
        self.assertLessEqual(abs(centered.vy), cfg.sweep_max_speed)
        self.assertGreater(too_far_ahead.vy, centered.vy)
        self.assertLess(falling_behind.vy, centered.vy)

    def test_lateral_push_command_flips_for_opposite_goal(self):
        cfg = self._config()
        defense_goal = Point2D(0.0, 5.0)

        command = lateral_push_command(Point2D(0.0, -cfg.sweep_side_offset), defense_goal, cfg)

        self.assertLess(command.vy, 0.0)

    def test_recover_avoidance_routes_around_ball_on_direct_path(self):
        cfg = self._config()
        self_point = Point2D(0.0, -1.0)
        guard_target = Point2D(0.0, -4.55)
        ball = Point2D(0.0, -2.5)

        target, avoided = recover_avoidance_target(
            self_point,
            guard_target,
            ball,
            Point2D(0.0, -5.0),
            cfg,
        )

        self.assertTrue(avoided)
        self.assertNotAlmostEqual(target.x, guard_target.x)
        self.assertGreater(
            segment_distance_to_point(self_point, target, ball),
            cfg.recover_ball_avoid_radius,
        )

    def test_recover_avoidance_keeps_clear_path_direct(self):
        cfg = self._config()
        self_point = Point2D(0.0, -1.0)
        guard_target = Point2D(0.0, -4.55)
        ball = Point2D(2.0, -2.5)

        target, avoided = recover_avoidance_target(
            self_point,
            guard_target,
            ball,
            Point2D(0.0, -5.0),
            cfg,
        )

        self.assertFalse(avoided)
        self.assertEqual(target, guard_target)

    def test_damped_speed_has_deadband_and_speed_cap(self):
        self.assertEqual(damped_speed(0.10, 2.0, 1.0, 0.20), 0.0)
        self.assertAlmostEqual(damped_speed(0.45, 2.0, 1.0, 0.20), 0.50)
        self.assertAlmostEqual(damped_speed(2.0, 2.0, 0.75, 0.20), 0.75)

    def test_goalkeeper_node_exposes_only_two_states(self):
        try:
            from src.goalkeeper import GoalkeeperNode
        except Exception as exc:  # pragma: no cover - resolved in ROS Docker.
            self.skipTest(f"goalkeeper node import unavailable: {exc}")

        self.assertEqual(
            GoalkeeperNode.STATES,
            (GoalkeeperNode.LATERAL_DEFEND, GoalkeeperNode.ACTIVE_INTERCEPT),
        )
        self.assertFalse(hasattr(GoalkeeperNode, "WAITING"))
        self.assertFalse(hasattr(GoalkeeperNode, "GUARD"))
        self.assertFalse(hasattr(GoalkeeperNode, "INTERCEPT"))
        self.assertFalse(hasattr(GoalkeeperNode, "CLEAR"))


if __name__ == "__main__":
    unittest.main()
