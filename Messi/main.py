"""Process entry point for Cyberdog Soccer.

The robot is intended to start from the repository root with ``python3
main.py``. This script loads YAML configuration, creates the shared lib nodes
and the selected role controller, then runs them in one MultiThreadedExecutor.
"""

import argparse
import sys
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
import sys, site
u = site.getusersitepackages()
if u in sys.path:
    sys.path.remove(u)
    sys.path.append(u)
from src.goalkeeper import GoalkeeperNode
from src.goalkeeper_policy import GoalkeeperPolicyNode
from src.lib.ball_predictor import BallPredictorNode
from src.lib.config import load_config_file
from src.lib.locator import LocatorNode
from src.lib.move import MoveCommander
from src.lib.path_follower import PathFollowerNode
from src.lib.path_planner import PathPlannerNode
from src.lib.topics import global_soccer_topic
from src.striker import StrikerNode
from src.striker_legacy import StrikerLegacyNode
from src.striker_policy import StrikerPolicyNode



REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"
REQUIRED_RIGIDS = ("self", "ball", "attack_goal", "defense_goal")
GOALKEEPER_MODES = {"rule", "hybrid", "policy"}
GAME_STATES = (
    "INIT",
    "READY",
    "KICKOFF_TEAM_A",
    "KICKOFF_TEAM_B",
    "PLAYING",
    "GOAL_TEAM_A",
    "GOAL_TEAM_B",
    "OUT_OF_BOUNDS",
    "PAUSED",
    "RESETTING",
)


def parse_args(argv):
    """Separate this script's flags from ROS remapping arguments."""

    parser = argparse.ArgumentParser(description="Cyberdog Soccer launcher")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to config YAML. Defaults to config/config.yaml.",
    )
    parser.add_argument(
        "--game-state",
        choices=GAME_STATES,
        help=(
            "Continuously publish this value on /soccer/game_state. "
            "Use PLAYING to let role controllers move immediately."
        ),
    )
    parser.add_argument(
        "--game-state-rate",
        type=float,
        default=2.0,
        help="Publish rate in Hz for --game-state. Defaults to 2.0.",
    )
    return parser.parse_known_args(argv)


def load_config(path: str):
    """Load YAML config and validate the rigid bodies required by the refactor."""

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config_file(config_path)

    rigids = config.get("rigids", {})
    missing = [name for name in REQUIRED_RIGIDS if name not in rigids]
    if missing:
        raise ValueError(f"Missing required rigids in {config_path}: {', '.join(missing)}")
    if config.get("role") not in ("striker", "legacy_striker", "policy_striker", "goalkeeper"):
        raise ValueError(
            "config role must be one of: striker, legacy_striker, policy_striker, goalkeeper"
        )
    return config


def resolve_goalkeeper_mode(config):
    """Return the configured goalkeeper controller mode.

    Backward compatibility: old configs used ``goalkeeper_policy.enabled`` plus
    ``hybrid_clear``. New configs should use ``goalkeeper_controller.mode``.
    """

    controller_config = config.get("goalkeeper_controller", {})
    mode = controller_config.get("mode")
    if mode is None:
        policy_config = config.get("goalkeeper_policy", {})
        if policy_config.get("enabled", False):
            mode = "hybrid" if policy_config.get("hybrid_clear", True) else "policy"
        else:
            mode = "rule"
    mode = str(mode).strip().lower()
    if mode not in GOALKEEPER_MODES:
        raise ValueError(
            "goalkeeper_controller.mode must be one of: "
            + ", ".join(sorted(GOALKEEPER_MODES))
        )
    return mode


def build_nodes(config):
    """Create the node set needed for the configured role."""

    nodes = []
    sim = config.get("sim", {})
    node_suffix = "_".join(part for part in (sim.get("team", ""), sim.get("robot_id", "")) if part)

    def node_name(base: str) -> str:
        return f"{base}_{node_suffix}" if node_suffix else base

    locator = LocatorNode(config, name=node_name("locator"))
    move = MoveCommander(config=config, name=node_name("move_commander"))
    nodes.extend([locator, move])

    role = config.get("role")
    if role == "striker":
        path_planner = PathPlannerNode(config, name=node_name("path_planner"))
        path_follower = PathFollowerNode(config, move, name=node_name("path_follower"))
        striker = StrikerNode(
            config,
            move,
            path_follower=path_follower,
            name=node_name("striker"),
        )
        nodes.extend([path_planner, path_follower, striker])
    elif role == "legacy_striker":
        striker = StrikerLegacyNode(config, move, name=node_name("legacy_striker"))
        nodes.append(striker)
    elif role == "policy_striker":
        striker = StrikerPolicyNode(config, move, name=node_name("policy_striker"))
        nodes.append(striker)
    else:
        goalkeeper_mode = resolve_goalkeeper_mode(config)
        goalkeeper_cls = GoalkeeperNode if goalkeeper_mode == "rule" else GoalkeeperPolicyNode
        goalkeeper = goalkeeper_cls(config, move, name=node_name("goalkeeper"))
        if goalkeeper_mode == "rule":
            nodes.append(goalkeeper)
        else:
            ball_predictor = BallPredictorNode(config, name=node_name("ball_predictor"))
            nodes.extend([ball_predictor, goalkeeper])

    return nodes, move


def executor_thread_count(config):
    """Return configured rclpy executor thread count, or None for rclpy default."""

    value = config.get("runtime", {}).get("executor_threads")
    if value is None:
        return None
    return max(int(value), 1)


class GameStatePublisher(Node):
    """Publishes a local match state when no external referee/simulator is used."""

    def __init__(self, state: str, rate_hz: float = 2.0):
        super().__init__("game_state_publisher")
        self._state = state
        self._pub = self.create_publisher(String, global_soccer_topic("game_state"), 10)
        self._timer = self.create_timer(1.0 / max(rate_hz, 0.1), self._publish)
        self._publish()
        self.get_logger().info(f"publishing /soccer/game_state = {state}")

    def _publish(self):
        msg = String()
        msg.data = self._state
        self._pub.publish(msg)


def shutdown_nodes(executor, nodes, move):
    """Publish a stop command and tear down ROS resources in a predictable order."""

    if rclpy.ok():
        try:
            move.stop(use_stop_motion=True)
            move.timer_callback()
        except Exception as exc:  # pragma: no cover - best-effort shutdown path.
            print(f"failed to publish stop command during shutdown: {exc}", file=sys.stderr)

    executor.shutdown()
    for node in reversed(nodes):
        node.destroy_node()


def main(argv=None):
    script_args, ros_args = parse_args(sys.argv[1:] if argv is None else argv)
    config = load_config(script_args.config)

    rclpy.init(args=ros_args)
    executor = MultiThreadedExecutor(num_threads=executor_thread_count(config))
    nodes, move = build_nodes(config)
    if script_args.game_state:
        nodes.append(GameStatePublisher(script_args.game_state, script_args.game_state_rate))
    for node in nodes:
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_nodes(executor, nodes, move)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
