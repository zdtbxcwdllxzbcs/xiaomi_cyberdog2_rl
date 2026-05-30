"""List locally registered Cyberdog2 IsaacLab tasks."""

from pathlib import Path
import argparse
import sys

from isaaclab.app import AppLauncher

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "source" / "cyberdog2_rl_lab"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

parser = argparse.ArgumentParser(description="List registered Cyberdog2 IsaacLab tasks.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _prefer_d3d12_on_windows(args) -> None:
    """Force D3D12 on Windows unless the caller already overrode the graphics backend."""

    if sys.platform != "win32":
        return
    kit_args = getattr(args, "kit_args", "") or ""
    if "/app/vulkan=" in kit_args:
        return
    args.kit_args = (kit_args + " --/app/vulkan=false").strip()


_prefer_d3d12_on_windows(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import cyberdog2_rl_lab.tasks  # noqa: F401


def main():
    matches = [task_spec for task_spec in sorted(gym.registry.values(), key=lambda item: item.id) if task_spec.id.startswith("Cyberdog2-")]
    print("Available Environments in Cyberdog2 RL Lab")
    for index, task_spec in enumerate(matches, start=1):
        print(f"{index}. {task_spec.id}")
        print(f"   entry_point: {task_spec.entry_point}")
        print(f"   config: {task_spec.kwargs['env_cfg_entry_point']}")


if __name__ == "__main__":
    main()
    simulation_app.close()
