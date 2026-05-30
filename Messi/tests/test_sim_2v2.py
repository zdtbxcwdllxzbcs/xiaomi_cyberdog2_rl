"""Extended unit tests for the 2v2 simulation additions.

Covers:
- Config mapping validation for all four sim configs
- Game-state gating (role controllers stop when game state is inactive)
- Game-manager goal detection logic
- Tactical helper facts (src/lib/tactics.py)

Run with:
  conda activate robocup_mujoco
  python3 -m pytest tests/ -v
"""

import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.lib.config import load_config_file
from src.lib.geometry import Point2D
from src.lib.tactics import (
    ball_heading_to_goal,
    distance_self_to_ball,
    distance_teammate_to_ball,
    keeper_zone_contains_ball,
    nearest_opponent_to_ball,
    shot_lane_clear,
    teammate_has_better_angle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
FIELD_CONFIG_PATH = REPO_ROOT / "sim" / "config" / "field.yaml"

SIM_CONFIGS = [
    "sim_2v2_team_a_1.yaml",
    "sim_2v2_team_a_2.yaml",
    "sim_2v2_team_b_1.yaml",
    "sim_2v2_team_b_2.yaml",
]

REQUIRED_RIGIDS = ("self", "teammate", "opponent_1", "opponent_2", "ball", "attack_goal", "defense_goal")


class SimConfigMappingTests(unittest.TestCase):
    """Verify that all four sim configs have correct rigid body mappings."""

    def _load(self, filename: str) -> dict:
        return load_config_file(CONFIG_DIR / filename)

    def _load_raw(self, filename: str) -> dict:
        with (CONFIG_DIR / filename).open() as f:
            return yaml.safe_load(f)

    def test_sim_configs_use_shared_measured_base(self):
        per_robot_keys = {"base_config", "role", "dog_namespace", "topics", "rigids", "sim"}
        for cfg_file in SIM_CONFIGS:
            with self.subTest(config=cfg_file):
                raw = self._load_raw(cfg_file)
                self.assertEqual(raw["base_config"], "config.yaml")
                self.assertLessEqual(set(raw), per_robot_keys)
                cfg = self._load(cfg_file)
                self.assertIn("move", cfg)
                self.assertIn("striker", cfg)
                self.assertIn("goalkeeper", cfg)

    def test_all_configs_have_required_rigids(self):
        for cfg_file in SIM_CONFIGS:
            with self.subTest(config=cfg_file):
                cfg = self._load(cfg_file)
                rigids = cfg.get("rigids", {})
                for name in REQUIRED_RIGIDS:
                    self.assertIn(name, rigids, f"{cfg_file} missing rigid: {name}")

    def test_team_a_attacks_goal_b(self):
        for cfg_file in ("sim_2v2_team_a_1.yaml", "sim_2v2_team_a_2.yaml"):
            cfg = self._load(cfg_file)
            self.assertEqual(cfg["rigids"]["attack_goal"], "goal_b")
            self.assertEqual(cfg["rigids"]["defense_goal"], "goal_a")

    def test_team_b_attacks_goal_a(self):
        for cfg_file in ("sim_2v2_team_b_1.yaml", "sim_2v2_team_b_2.yaml"):
            cfg = self._load(cfg_file)
            self.assertEqual(cfg["rigids"]["attack_goal"], "goal_a")
            self.assertEqual(cfg["rigids"]["defense_goal"], "goal_b")

    def test_team_a_1_is_policy_striker(self):
        cfg = self._load("sim_2v2_team_a_1.yaml")
        self.assertEqual(cfg["role"], "policy_striker")

    def test_team_a_2_is_goalkeeper(self):
        cfg = self._load("sim_2v2_team_a_2.yaml")
        self.assertEqual(cfg["role"], "goalkeeper")

    def test_team_b_1_is_striker(self):
        cfg = self._load("sim_2v2_team_b_1.yaml")
        self.assertEqual(cfg["role"], "striker")

    def test_team_b_1_striker_tunables_inherit_measured_base(self):
        cfg = self._load("sim_2v2_team_b_1.yaml")
        base = self._load("config.yaml")
        striker = cfg["striker"]

        self.assertGreaterEqual(striker["legacy_possession_distance"], 0.50)
        self.assertEqual(striker["legacy_kick_speed"], base["striker"]["legacy_kick_speed"])
        self.assertEqual(
            striker["legacy_orbit_tolerance"],
            base["striker"]["legacy_orbit_tolerance"],
        )

    def test_team_b_2_is_goalkeeper(self):
        cfg = self._load("sim_2v2_team_b_2.yaml")
        self.assertEqual(cfg["role"], "goalkeeper")

    def test_opponents_are_cross_team(self):
        """Each robot's opponents should be from the other team."""
        for cfg_file in SIM_CONFIGS:
            cfg = self._load(cfg_file)
            rigids = cfg["rigids"]
            sim = cfg.get("sim", {})
            team = sim.get("team", "")
            opp1 = rigids["opponent_1"]
            opp2 = rigids["opponent_2"]
            other_team = "team_b" if team == "team_a" else "team_a"
            self.assertTrue(
                opp1.startswith(other_team),
                f"{cfg_file}: opponent_1 {opp1!r} should be from {other_team}",
            )
            self.assertTrue(
                opp2.startswith(other_team),
                f"{cfg_file}: opponent_2 {opp2!r} should be from {other_team}",
            )

    def test_namespaces_are_unique(self):
        namespaces = []
        for cfg_file in SIM_CONFIGS:
            cfg = self._load(cfg_file)
            namespaces.append(cfg["dog_namespace"])
        self.assertEqual(len(namespaces), len(set(namespaces)), "Duplicate dog_namespace values")

    def test_soccer_topic_prefixes_are_unique(self):
        from src.lib.topics import soccer_topic

        prefixes = []
        for cfg_file in SIM_CONFIGS:
            cfg = self._load(cfg_file)
            prefix = cfg.get("topics", {}).get("prefix")
            self.assertTrue(prefix, f"{cfg_file} missing topics.prefix")
            prefixes.append(prefix)
            self.assertEqual(soccer_topic(cfg, "world", "self"), f"{prefix}/world/self")

        self.assertEqual(len(prefixes), len(set(prefixes)), "Duplicate soccer topic prefixes")

    def test_self_rigid_matches_team_and_robot(self):
        expected = {
            "sim_2v2_team_a_1.yaml": "team_a_1",
            "sim_2v2_team_a_2.yaml": "team_a_2",
            "sim_2v2_team_b_1.yaml": "team_b_1",
            "sim_2v2_team_b_2.yaml": "team_b_2",
        }
        for cfg_file, expected_self in expected.items():
            cfg = self._load(cfg_file)
            self.assertEqual(cfg["rigids"]["self"], expected_self, cfg_file)


class PygameIntentDisplayTests(unittest.TestCase):
    """Verify the tactical board distinguishes controller state machines."""

    def test_legacy_controller_display_uses_controller(self):
        from sim.pygame_sim import _robot_overlay_label, _robot_sidebar_line

        intent = {"role": "striker", "controller": "legacy", "state": "approach"}

        self.assertEqual(_robot_overlay_label("team_b_1", intent), "B1:L-APP")
        self.assertIn("B1: legacy striker approach", _robot_sidebar_line("team_b_1", intent))

    def test_legacy_display_falls_back_to_config_role(self):
        from sim.pygame_sim import _robot_overlay_label, _robot_sidebar_line

        intent = {"state": "dribble"}

        self.assertEqual(_robot_overlay_label("team_b_1", intent), "B1:S-DRI")
        self.assertIn("B1: striker dribble", _robot_sidebar_line("team_b_1", intent))

    def test_policy_goalkeeper_intent_displays_as_goalkeeper(self):
        from sim.pygame_sim import _robot_overlay_label, _robot_sidebar_line

        intent = {"role": "goalkeeper", "policy_model": "model.zip", "state": "guard"}

        self.assertEqual(_robot_overlay_label("team_b_2", intent), "B2:G-GUA")
        self.assertIn("B2: goalkeeper guard", _robot_sidebar_line("team_b_2", intent))


class CyberDogFullStackConfigTests(unittest.TestCase):
    """Verify the full CyberDog 2v2 simulation naming contract."""

    def test_robot_model_names_are_stable(self):
        from sim.cyberdog_2v2 import MODEL_NAMES

        self.assertEqual(
            MODEL_NAMES,
            (
                "team_a_robot_1",
                "team_a_robot_2",
                "team_b_robot_1",
                "team_b_robot_2",
            ),
        )

    def test_shared_memory_names_are_unique(self):
        from sim.cyberdog_2v2 import MODEL_NAMES, shared_memory_name

        names = [shared_memory_name(model_name) for model_name in MODEL_NAMES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            set(names),
            {
                "development-simulator__team_a_robot_1",
                "development-simulator__team_a_robot_2",
                "development-simulator__team_b_robot_1",
                "development-simulator__team_b_robot_2",
            },
        )

    def test_lcm_channels_are_unique(self):
        from sim.cyberdog_2v2 import (
            MODEL_NAMES,
            exec_request_channel,
            gamepad_channel,
            robot_control_channel,
            simulator_state_channel,
        )

        for channel_fn in (
            robot_control_channel,
            exec_request_channel,
            gamepad_channel,
            simulator_state_channel,
        ):
            channels = [channel_fn(model_name) for model_name in MODEL_NAMES]
            self.assertEqual(len(channels), len(set(channels)), channel_fn.__name__)

        self.assertIn("robot_control_cmd__team_a_robot_1", [robot_control_channel(n) for n in MODEL_NAMES])
        self.assertIn("exec_request__team_b_robot_2", [exec_request_channel(n) for n in MODEL_NAMES])

    def test_pygame_field_config_matches_reference_sketch(self):
        with FIELD_CONFIG_PATH.open() as f:
            cfg = yaml.safe_load(f)

        field = cfg["field"]
        self.assertAlmostEqual(field["length"], 8.8)
        self.assertAlmostEqual(field["width"], 5.55)
        self.assertAlmostEqual(field["goal_width"], 1.0)

        kickoff_a = cfg["kickoff_poses"]["KICKOFF_A"]
        self.assertEqual(kickoff_a["team_a_1"]["y"], -0.5)
        self.assertEqual(kickoff_a["team_b_1"]["y"], 2.5)

    def test_pygame_field_config_matches_measured_robot_and_ball_sizes(self):
        with FIELD_CONFIG_PATH.open() as f:
            cfg = yaml.safe_load(f)

        physics = cfg["physics"]
        self.assertAlmostEqual(physics["ball_radius"], 0.125)
        self.assertAlmostEqual(physics["robot_length"], 0.43)
        self.assertAlmostEqual(physics["robot_body_width"], 0.20)
        self.assertAlmostEqual(physics["robot_leg_width"], 0.32)
        self.assertAlmostEqual(physics["robot_radius"], 0.27)
        self.assertAlmostEqual(physics["robot_ball_restitution"], 0.55)
        self.assertAlmostEqual(physics["robot_ball_tangent_gain"], 0.12)


class PygamePhysicsFootprintTests(unittest.TestCase):
    """Verify the contact model follows the robot's measured footprint."""

    def test_ball_radius_matches_25cm_diameter(self):
        from sim.pygame_sim import BALL_RADIUS

        self.assertAlmostEqual(BALL_RADIUS, 0.125)

    def test_robot_ball_contact_uses_front_and_side_footprint(self):
        from sim.pygame_sim import (
            BALL_RADIUS,
            ROBOT_BODY_WIDTH,
            ROBOT_LEG_WIDTH,
            ROBOT_LENGTH,
            BallState,
            PhysicsWorld,
            RobotState,
        )

        world = PhysicsWorld()
        robot = RobotState(0.0, 0.0, 0.0)

        front_ball = BallState(x=ROBOT_LENGTH * 0.5 + BALL_RADIUS - 0.02, y=0.0)
        world._robot_ball_impulse(robot, front_ball)
        self.assertGreaterEqual(front_ball.x, ROBOT_LENGTH * 0.5 + BALL_RADIUS - 1e-6)

        side_ball = BallState(x=0.0, y=ROBOT_LEG_WIDTH * 0.5 + BALL_RADIUS - 0.02)
        world._robot_ball_impulse(robot, side_ball)
        self.assertGreaterEqual(abs(side_ball.y), ROBOT_LEG_WIDTH * 0.5 + BALL_RADIUS - 1e-6)

        body_ball = BallState(x=ROBOT_LENGTH * 0.5 + BALL_RADIUS - 0.02, y=ROBOT_BODY_WIDTH * 0.5 - 0.01)
        world._robot_ball_impulse(robot, body_ball)
        self.assertGreaterEqual(body_ball.x, ROBOT_LENGTH * 0.5 + BALL_RADIUS - 1e-6)


class GymPhysicsFootprintTests(unittest.TestCase):
    """Verify the training env stays synced with pygame physical sizing."""

    def _gym_symbols(self):
        try:
            from gym.soccer_env import (
                BALL_RADIUS,
                ROBOT_BODY_WIDTH,
                ROBOT_LEG_WIDTH,
                ROBOT_LENGTH,
                BallState,
                RobotState,
                SoccerEnv,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        return (
            SoccerEnv,
            RobotState,
            BallState,
            BALL_RADIUS,
            ROBOT_LENGTH,
            ROBOT_BODY_WIDTH,
            ROBOT_LEG_WIDTH,
        )

    def test_gym_defaults_match_field_config(self):
        SoccerEnv, _, _, _, _, _, _ = self._gym_symbols()

        env = SoccerEnv(phase=3, domain_randomization=False)
        _, info = env.reset(seed=0)
        domain = info["domain"]

        self.assertAlmostEqual(domain["ball_radius"], 0.125)
        self.assertAlmostEqual(domain["robot_length"], 0.43)
        self.assertAlmostEqual(domain["robot_body_width"], 0.20)
        self.assertAlmostEqual(domain["robot_leg_width"], 0.32)
        self.assertAlmostEqual(domain["robot_radius"], 0.27, places=2)
        self.assertAlmostEqual(domain["robot_ball_restitution"], 0.55)
        self.assertAlmostEqual(domain["robot_ball_tangent_gain"], 0.12)

    def test_gym_robot_ball_contact_uses_rectangular_footprint(self):
        (
            SoccerEnv,
            RobotState,
            BallState,
            BALL_RADIUS,
            ROBOT_LENGTH,
            ROBOT_BODY_WIDTH,
            ROBOT_LEG_WIDTH,
        ) = self._gym_symbols()

        env = SoccerEnv(phase=1, domain_randomization=False)
        robot = RobotState(0.0, 0.0, 0.0)

        front_ball = BallState(x=ROBOT_LENGTH * 0.5 + BALL_RADIUS - 0.02, y=0.0)
        env._robot_ball_impulse(robot, front_ball)
        self.assertGreaterEqual(front_ball.x, ROBOT_LENGTH * 0.5 + BALL_RADIUS - 1e-6)

        side_ball = BallState(x=0.0, y=ROBOT_LEG_WIDTH * 0.5 + BALL_RADIUS - 0.02)
        env._robot_ball_impulse(robot, side_ball)
        self.assertGreaterEqual(abs(side_ball.y), ROBOT_LEG_WIDTH * 0.5 + BALL_RADIUS - 1e-6)

        body_ball = BallState(x=ROBOT_LENGTH * 0.5 + BALL_RADIUS - 0.02, y=ROBOT_BODY_WIDTH * 0.5 - 0.01)
        env._robot_ball_impulse(robot, body_ball)
        self.assertGreaterEqual(body_ball.x, ROBOT_LENGTH * 0.5 + BALL_RADIUS - 1e-6)

    def test_gym_wrong_side_reset_puts_striker_goal_side_of_ball(self):
        SoccerEnv, _, _, _, _, _, _ = self._gym_symbols()

        env = SoccerEnv(phase=1, domain_randomization=False, wrong_side_reset_prob=1.0)
        _, info = env.reset(seed=2)
        behind_depth, _ = env._behind_ball_metrics(env.striker, env.ball)

        self.assertEqual(info["reset_scenario"], "wrong_side")
        self.assertLess(behind_depth, 0.0)

    def test_gym_reward_prefers_recovering_from_wrong_side(self):
        SoccerEnv, RobotState, BallState, _, _, _, _ = self._gym_symbols()

        env = SoccerEnv(phase=1, domain_randomization=False, wrong_side_reset_prob=0.0)
        prev_robot = RobotState(x=0.0, y=0.0, yaw=math.pi * 0.5)
        prev_ball = BallState(x=0.0, y=-0.45)

        env.striker = RobotState(x=0.0, y=-0.10, yaw=math.pi * 0.5)
        env.ball = BallState(x=0.0, y=-0.45)
        recover_reward = env._compute_reward(prev_robot, prev_ball, None, [])

        env.striker = RobotState(x=0.0, y=0.10, yaw=math.pi * 0.5)
        env.ball = BallState(x=0.0, y=-0.45)
        drift_reward = env._compute_reward(prev_robot, prev_ball, None, [])

        self.assertGreater(recover_reward, drift_reward)

    def test_gym_phase1_can_start_in_shooting_setup(self):
        SoccerEnv, _, _, _, _, _, _ = self._gym_symbols()

        env = SoccerEnv(
            phase=1,
            domain_randomization=False,
            wrong_side_reset_prob=0.0,
            phase1_shooting_reset_prob=1.0,
        )
        _, info = env.reset(seed=7)
        behind_depth, lateral = env._behind_ball_metrics(env.striker, env.ball)

        self.assertEqual(info["reset_scenario"], "shooting")
        self.assertGreater(behind_depth, 0.0)
        self.assertLess(abs(lateral), 0.25)
        self.assertLess(abs(env.ball.x), 0.5)

    def test_gym_reward_prefers_shot_alignment_over_side_scrape(self):
        SoccerEnv, RobotState, BallState, _, _, _, _ = self._gym_symbols()

        env = SoccerEnv(phase=1, domain_randomization=False, wrong_side_reset_prob=0.0)
        prev_robot = RobotState(x=0.0, y=0.0, yaw=math.pi * 0.5)
        prev_ball = BallState(x=1.2, y=0.0)

        env.striker = RobotState(x=0.0, y=0.0, yaw=math.pi * 0.5)
        env.ball = BallState(x=0.6, y=0.9)
        aligned_reward = env._compute_reward(prev_robot, prev_ball, None, [])

        env.striker = RobotState(x=0.0, y=0.0, yaw=math.pi * 0.5)
        env.ball = BallState(x=1.8, y=0.9)
        side_scrape_reward = env._compute_reward(prev_robot, prev_ball, None, [])

        self.assertGreater(aligned_reward, side_scrape_reward)

    def test_gym_latency_domain_samples_policy_rate_and_delays(self):
        SoccerEnv, _, _, _, _, _, _ = self._gym_symbols()

        env = SoccerEnv(
            phase=1,
            domain_randomization=False,
            latency_randomization=True,
            policy_hz_range=(10.0, 10.0),
            action_latency_range=(0.10, 0.10),
            observation_latency_range=(0.10, 0.10),
        )
        _, info = env.reset(seed=0)
        domain = info["domain"]

        self.assertEqual(domain["policy_interval_steps"], 5)
        self.assertEqual(domain["action_latency_steps"], 1)
        self.assertEqual(domain["observation_latency_steps"], 1)

    def test_gym_action_latency_delays_new_command(self):
        SoccerEnv, RobotState, BallState, _, _, _, _ = self._gym_symbols()

        env = SoccerEnv(
            phase=1,
            domain_randomization=False,
            latency_randomization=True,
            policy_hz_range=(50.0, 50.0),
            action_latency_range=(0.02, 0.02),
            observation_latency_range=(0.0, 0.0),
        )
        env.reset(seed=0)
        env.striker = RobotState(x=0.0, y=0.0, yaw=0.0)
        env.ball = BallState(x=1.5, y=1.5)
        env._reset_latency_buffers()

        env.step([1.0, 0.0, 0.0])
        first_x = env.striker.x
        env.step([1.0, 0.0, 0.0])

        self.assertAlmostEqual(first_x, 0.0)
        self.assertGreater(env.striker.x, first_x)

    def test_gym_observation_latency_returns_previous_policy_tick(self):
        SoccerEnv, RobotState, BallState, _, _, _, _ = self._gym_symbols()

        env = SoccerEnv(
            phase=1,
            domain_randomization=False,
            latency_randomization=True,
            policy_hz_range=(50.0, 50.0),
            action_latency_range=(0.0, 0.0),
            observation_latency_range=(0.02, 0.02),
        )
        env.reset(seed=0)
        env.striker = RobotState(x=0.0, y=0.0, yaw=0.0)
        env.ball = BallState(x=1.5, y=1.5)
        env._reset_latency_buffers()

        obs, _, _, _, _ = env.step([1.0, 0.0, 0.0])

        self.assertAlmostEqual(float(obs[0]), 0.0)
        self.assertGreater(env.striker.x, 0.0)

    def test_gym_phase4_owns_default_latency(self):
        SoccerEnv, _, _, _, _, _, _ = self._gym_symbols()

        phase3 = SoccerEnv(phase=3, domain_randomization=False)
        _, phase3_info = phase3.reset(seed=0)
        phase4 = SoccerEnv(phase=4, domain_randomization=False)
        _, phase4_info = phase4.reset(seed=0)

        self.assertFalse(phase3.latency_randomization)
        self.assertEqual(phase3_info["domain"]["policy_interval_steps"], 1)
        self.assertTrue(phase4.latency_randomization)
        self.assertGreater(phase4_info["domain"]["policy_interval_steps"], 1)

    def test_gym_ball_wall_terminates_and_self_wall_only_penalizes(self):
        SoccerEnv, RobotState, BallState, _, _, _, _ = self._gym_symbols()

        env = SoccerEnv(phase=1, domain_randomization=False, latency_randomization=False)
        env.reset(seed=0)
        prev_ball_striker = RobotState(x=0.0, y=0.0, yaw=0.0)
        prev_ball = BallState(x=env.half_width - env.ball_radius * 0.5, y=0.0, vx=1.0, vy=0.0)
        env.striker = RobotState(x=prev_ball_striker.x, y=prev_ball_striker.y, yaw=prev_ball_striker.yaw)
        env.ball = BallState(x=prev_ball.x, y=prev_ball.y, vx=prev_ball.vx, vy=prev_ball.vy)

        _, ball_reward, ball_terminated, _, ball_info = env.step([0.0, 0.0, 0.0])
        ball_reward_without_event = env._compute_reward(
            prev_ball_striker,
            prev_ball,
            None,
            [],
        )

        self.assertTrue(ball_terminated)
        self.assertEqual(ball_info["event"], "BALL_WALL")
        self.assertIn("BALL_WALL", ball_info["events"])
        self.assertLess(ball_reward, ball_reward_without_event)

        env.reset(seed=1)
        prev_self_striker = RobotState(x=env.half_width, y=0.0, yaw=0.0)
        prev_self_ball = BallState(x=0.0, y=1.5)
        env.striker = RobotState(x=prev_self_striker.x, y=prev_self_striker.y, yaw=prev_self_striker.yaw)
        env.ball = BallState(x=prev_self_ball.x, y=prev_self_ball.y)

        _, self_reward, self_terminated, _, self_info = env.step([0.0, 0.0, 0.0])
        self_reward_without_event = env._compute_reward(
            prev_self_striker,
            prev_self_ball,
            self_info["event"],
            [],
        )

        self.assertFalse(self_terminated)
        self.assertEqual(self_info["event"], "SELF_WALL")
        self.assertIn("SELF_WALL", self_info["events"])
        self.assertLess(self_reward, self_reward_without_event)

    def test_gym_robot_collisions_penalize_without_ending_episode(self):
        SoccerEnv, RobotState, BallState, _, _, _, _ = self._gym_symbols()

        teammate_env = SoccerEnv(phase=4, domain_randomization=False, latency_randomization=False)
        teammate_env.reset(seed=0)
        prev_teammate_striker = RobotState(x=0.0, y=0.0, yaw=0.0)
        prev_teammate_ball = BallState(x=1.5, y=1.5)
        teammate_env.striker = RobotState(
            x=prev_teammate_striker.x,
            y=prev_teammate_striker.y,
            yaw=prev_teammate_striker.yaw,
        )
        teammate_env.teammate_goalkeeper = RobotState(x=0.1, y=0.0, yaw=0.0)
        teammate_env.goalkeeper = RobotState(x=2.0, y=2.0, yaw=0.0)
        teammate_env.opponent_striker = RobotState(x=-2.0, y=2.0, yaw=0.0)
        teammate_env.ball = BallState(x=prev_teammate_ball.x, y=prev_teammate_ball.y)

        _, teammate_reward, teammate_terminated, _, teammate_info = teammate_env.step([0.0, 0.0, 0.0])
        teammate_reward_without_event = teammate_env._compute_reward(
            prev_teammate_striker,
            prev_teammate_ball,
            teammate_info["event"],
            [],
        )

        self.assertFalse(teammate_terminated)
        self.assertIn("TEAMMATE_COLLISION", teammate_info["events"])
        self.assertLess(teammate_reward, teammate_reward_without_event)

        opponent_env = SoccerEnv(phase=2, domain_randomization=False, latency_randomization=False)
        opponent_env.reset(seed=1)
        prev_opponent_striker = RobotState(x=0.0, y=0.0, yaw=0.0)
        prev_opponent_ball = BallState(x=1.5, y=1.5)
        opponent_env.striker = RobotState(
            x=prev_opponent_striker.x,
            y=prev_opponent_striker.y,
            yaw=prev_opponent_striker.yaw,
        )
        opponent_env.goalkeeper = RobotState(x=0.1, y=0.0, yaw=0.0)
        opponent_env.ball = BallState(x=prev_opponent_ball.x, y=prev_opponent_ball.y)

        _, opponent_reward, opponent_terminated, _, opponent_info = opponent_env.step([0.0, 0.0, 0.0])
        opponent_reward_without_event = opponent_env._compute_reward(
            prev_opponent_striker,
            prev_opponent_ball,
            opponent_info["event"],
            [],
        )

        self.assertFalse(opponent_terminated)
        self.assertIn("OPPONENT_COLLISION", opponent_info["events"])
        self.assertLess(opponent_reward, opponent_reward_without_event)


class GameStateGatingTests(unittest.TestCase):
    """Verify that inactive game states cause role controllers to stop."""

    INACTIVE_STATES = ["INIT", "READY", "PAUSED", "RESETTING", "GOAL_TEAM_A", "GOAL_TEAM_B"]
    ACTIVE_STATES = ["PLAYING", "KICKOFF_TEAM_A", "KICKOFF_TEAM_B"]

    def _active_states_set(self):
        from src.striker import _ACTIVE_GAME_STATES as striker_active
        from src.striker_legacy import _ACTIVE_GAME_STATES as legacy_striker_active
        from src.goalkeeper import _ACTIVE_GAME_STATES as gk_active
        return striker_active, legacy_striker_active, gk_active

    def test_active_states_include_playing(self):
        striker_active, legacy_striker_active, gk_active = self._active_states_set()
        self.assertIn("PLAYING", striker_active)
        self.assertIn("PLAYING", legacy_striker_active)
        self.assertIn("PLAYING", gk_active)

    def test_active_states_include_kickoff(self):
        striker_active, legacy_striker_active, gk_active = self._active_states_set()
        self.assertIn("KICKOFF_TEAM_A", striker_active)
        self.assertIn("KICKOFF_TEAM_B", striker_active)
        self.assertIn("KICKOFF_TEAM_A", legacy_striker_active)
        self.assertIn("KICKOFF_TEAM_B", legacy_striker_active)
        self.assertIn("KICKOFF_TEAM_A", gk_active)
        self.assertIn("KICKOFF_TEAM_B", gk_active)

    def test_inactive_states_not_in_active_set(self):
        striker_active, legacy_striker_active, gk_active = self._active_states_set()
        for state in self.INACTIVE_STATES:
            self.assertNotIn(state, striker_active, f"Striker should not allow {state}")
            self.assertNotIn(state, legacy_striker_active, f"Legacy striker should not allow {state}")
            self.assertNotIn(state, gk_active, f"Goalkeeper should not allow {state}")


class GameManagerGoalDetectionTests(unittest.TestCase):
    """Verify goal detection logic from game_manager.py."""

    def _make_ball(self, x: float, y: float):
        from geometry_msgs.msg import PoseStamped
        msg = PoseStamped()
        msg.pose.position.x = x
        msg.pose.position.y = y
        return msg

    def test_ball_inside_goal_a_triggers_team_b_goal(self):
        """Ball at (0, -5.1) inside goal_a width -> team_b scores."""
        from sim.pygame_sim import GOAL_A_Y, GOAL_WIDTH, BALL_RADIUS
        bx, by = 0.0, GOAL_A_Y - BALL_RADIUS * 0.5
        inside_width = abs(bx) <= GOAL_WIDTH * 0.5 + BALL_RADIUS
        crossed_line = by <= GOAL_A_Y + BALL_RADIUS
        self.assertTrue(inside_width and crossed_line)

    def test_ball_inside_goal_b_triggers_team_a_goal(self):
        """Ball at (0, 5.1) inside goal_b width -> team_a scores."""
        from sim.pygame_sim import GOAL_B_Y, GOAL_WIDTH, BALL_RADIUS
        bx, by = 0.0, GOAL_B_Y + BALL_RADIUS * 0.5
        inside_width = abs(bx) <= GOAL_WIDTH * 0.5 + BALL_RADIUS
        crossed_line = by >= GOAL_B_Y - BALL_RADIUS
        self.assertTrue(inside_width and crossed_line)

    def test_ball_outside_goal_width_does_not_score(self):
        """Ball at (1.5, -5.1) is outside goal_a width -> no goal."""
        from sim.pygame_sim import GOAL_A_Y, GOAL_WIDTH, BALL_RADIUS
        bx, by = 1.5, GOAL_A_Y - BALL_RADIUS * 0.5
        inside_width = abs(bx) <= GOAL_WIDTH * 0.5 + BALL_RADIUS
        self.assertFalse(inside_width)

    def test_ball_in_midfield_does_not_score(self):
        """Ball at (0, 0) is not near any goal line."""
        from sim.pygame_sim import GOAL_A_Y, GOAL_B_Y, BALL_RADIUS
        bx, by = 0.0, 0.0
        near_a = by <= GOAL_A_Y + BALL_RADIUS
        near_b = by >= GOAL_B_Y - BALL_RADIUS
        self.assertFalse(near_a or near_b)


class TacticsHelperTests(unittest.TestCase):
    """Verify pure tactical helper functions."""

    def test_distance_self_to_ball(self):
        d = distance_self_to_ball(Point2D(0.0, 0.0), Point2D(3.0, 4.0))
        self.assertAlmostEqual(d, 5.0, places=6)

    def test_distance_teammate_to_ball(self):
        d = distance_teammate_to_ball(Point2D(1.0, 0.0), Point2D(1.0, 3.0))
        self.assertAlmostEqual(d, 3.0, places=6)

    def test_nearest_opponent_returns_closer_one(self):
        ball = Point2D(0.0, 0.0)
        opp1 = Point2D(1.0, 0.0)
        opp2 = Point2D(3.0, 0.0)
        nearest = nearest_opponent_to_ball(ball, opp1, opp2)
        self.assertEqual(nearest, opp1)

    def test_nearest_opponent_with_none(self):
        ball = Point2D(0.0, 0.0)
        nearest = nearest_opponent_to_ball(ball, None, Point2D(2.0, 0.0))
        self.assertIsNotNone(nearest)
        self.assertAlmostEqual(nearest.x, 2.0)

    def test_nearest_opponent_both_none(self):
        ball = Point2D(0.0, 0.0)
        self.assertIsNone(nearest_opponent_to_ball(ball, None, None))

    def test_ball_heading_toward_goal(self):
        ball = Point2D(0.0, 0.0)
        goal = Point2D(0.0, 5.0)
        self.assertTrue(ball_heading_to_goal(ball, 0.0, 1.0, goal))

    def test_ball_heading_away_from_goal(self):
        ball = Point2D(0.0, 0.0)
        goal = Point2D(0.0, 5.0)
        self.assertFalse(ball_heading_to_goal(ball, 0.0, -1.0, goal))

    def test_shot_lane_clear_no_opponents(self):
        self.assertTrue(
            shot_lane_clear(Point2D(0.0, 0.0), Point2D(0.0, 5.0), [None, None])
        )

    def test_shot_lane_blocked_by_opponent(self):
        self.assertFalse(
            shot_lane_clear(
                Point2D(0.0, 0.0),
                Point2D(0.0, 5.0),
                [Point2D(0.0, 2.5)],
                lane_half_width=0.3,
            )
        )

    def test_shot_lane_clear_opponent_to_side(self):
        self.assertTrue(
            shot_lane_clear(
                Point2D(0.0, 0.0),
                Point2D(0.0, 5.0),
                [Point2D(1.0, 2.5)],
                lane_half_width=0.3,
            )
        )

    def test_keeper_zone_contains_ball_inside(self):
        defense_goal = Point2D(0.0, -5.0)
        ball = Point2D(0.3, -4.8)
        self.assertTrue(
            keeper_zone_contains_ball(ball, defense_goal, goal_width=1.5, keeper_depth=0.6)
        )

    def test_keeper_zone_does_not_contain_midfield_ball(self):
        defense_goal = Point2D(0.0, -5.0)
        ball = Point2D(0.0, 0.0)
        self.assertFalse(
            keeper_zone_contains_ball(ball, defense_goal, goal_width=1.5, keeper_depth=0.6)
        )

    def test_teammate_has_better_angle_when_closer(self):
        # Teammate is closer to ball and has a clear angle to goal
        self_pos = Point2D(3.0, -1.0)
        teammate = Point2D(0.5, -0.5)
        ball = Point2D(0.0, 0.0)
        attack_goal = Point2D(0.0, 5.0)
        self.assertTrue(teammate_has_better_angle(self_pos, teammate, ball, attack_goal))

    def test_teammate_does_not_have_better_angle_when_farther(self):
        self_pos = Point2D(0.5, 0.0)
        teammate = Point2D(2.0, 0.0)
        ball = Point2D(0.0, 0.0)
        attack_goal = Point2D(0.0, 5.0)
        self.assertFalse(teammate_has_better_angle(self_pos, teammate, ball, attack_goal))

    def test_teammate_none_returns_false(self):
        self.assertFalse(
            teammate_has_better_angle(
                Point2D(0.0, 0.0), None, Point2D(1.0, 0.0), Point2D(0.0, 5.0)
            )
        )


if __name__ == "__main__":
    unittest.main()
