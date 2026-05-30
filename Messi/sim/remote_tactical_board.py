"""Render remote CyberDog ROS2 topics on the existing pygame tactical board.

This entrypoint keeps ``pygame_sim.py`` as the renderer/field implementation.
The live ROS2 graph stays on the robot: a small router is started over SSH,
subscribes to remote VRPN pose/twist topics, and streams JSON snapshots back to
this local pygame process.

SSH connection parameters and rigid-body name mappings are read from
``sim/config/remote.yaml`` by default. Fill in that file before running.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_CONFIG = Path(__file__).resolve().parent / "config" / "remote.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pygame  # noqa: E402
import yaml as _yaml  # noqa: E402

from sim import pygame_sim as board  # noqa: E402
from sim.ssh_topic_router import (  # noqa: E402
    DEFAULT_REMOTE_SETUPS,
    RemoteRouterConfig,
    SshRosTopicRouter,
    apply_mapping_overrides,
    build_ssh_command,
    load_default_slot_mapping,
    topic_specs_from_mapping,
)


ROLE_LABELS = {
    "team_a_1": "STK",
    "team_a_2": "GK",
    "team_b_1": "OP1",
    "team_b_2": "OP2",
}


def _load_remote_config() -> dict:
    """Load sim/config/remote.yaml; return empty dict if missing."""
    if REMOTE_CONFIG.exists():
        with REMOTE_CONFIG.open("r", encoding="utf-8") as f:
            return _yaml.safe_load(f) or {}
    return {}


def parse_args() -> argparse.Namespace:
    remote_cfg = _load_remote_config()
    ssh = remote_cfg.get("ssh", {})
    rigids = remote_cfg.get("rigids", {})

    parser = argparse.ArgumentParser(
        description="Draw remote ROS2 VRPN topics on the pygame tactical board."
    )
    parser.add_argument(
        "--host",
        default=ssh.get("host") or "",
        help="Remote robot host (overrides sim/config/remote.yaml).",
    )
    parser.add_argument(
        "--user",
        default=ssh.get("user") or "",
        help="Remote SSH user (overrides sim/config/remote.yaml).",
    )
    parser.add_argument(
        "--password",
        default=ssh.get("password") or "",
        help="SSH password. Leave empty when SSH keys are configured.",
    )
    parser.add_argument(
        "--remote-workspace",
        default=ssh.get("remote_workspace") or "~/cyberdog_soccer",
        help="Remote workspace used as the command working directory.",
    )
    parser.add_argument(
        "--remote-setup",
        action="append",
        default=list(remote_cfg.get("extra_setups") or []),
        help="Extra remote setup.bash to source before starting the router.",
    )
    parser.add_argument(
        "--no-default-setup",
        action="store_true",
        help="Do not source the default CyberDog/Galactic ROS2 setup files.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "config.yaml"),
        help="Local soccer config used to map board slots to remote VRPN rigid names.",
    )
    # Build default --robot overrides from sim/config/remote.yaml rigids section
    _default_robots = [
        f"{slot}={rigid}"
        for slot, rigid in rigids.items()
        if rigid
    ]
    parser.add_argument(
        "--robot",
        action="append",
        default=_default_robots,
        metavar="SLOT=RIGID_OR_TOPIC",
        help=(
            "Override a slot mapping, e.g. team_a_1=SoTaGo1 or "
            "ball=/vrpn/ball/pose. Can be repeated. "
            "Defaults are loaded from sim/config/remote.yaml."
        ),
    )
    parser.add_argument("--hz", type=float, default=5.0, help="Remote snapshot rate.")
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=0.6,
        help="Seconds before a received topic is marked stale in the board labels.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the generated SSH command and exit.",
    )
    return parser.parse_args()


def make_router_config(args: argparse.Namespace) -> RemoteRouterConfig:
    mapping = load_default_slot_mapping(args.config)
    mapping = apply_mapping_overrides(mapping, args.robot)
    setups = []
    if not args.no_default_setup:
        setups.extend(DEFAULT_REMOTE_SETUPS)
    setups.extend(args.remote_setup)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = _yaml.safe_load(f) or {}
    dog_ns = cfg.get("dog_namespace", "Hephaestus_1")

    return RemoteRouterConfig(
        host=args.host,
        user=args.user,
        password=args.password or None,
        remote_workspace=args.remote_workspace,
        remote_setups=tuple(setups),
        hz=args.hz,
        topics=topic_specs_from_mapping(mapping),
        game_state_topic="/soccer/game_state",
        path_topics={"approach": "/soccer/striker/path"},
        state_topics={"striker": "/soccer/striker/state"},
        cmd_topics={"striker": f"/{dog_ns}/motion_servo_cmd"},
    )


def initial_snapshot() -> dict:
    return {
        "robots": {},
        "ball": board.BallState(),
        "game_state": "REMOTE",
        "score": {"team_a": 0, "team_b": 0},
        "intents": {},
        "paths": {},
        "role_states": {},
        "cmd_vels": {},
    }


def update_snapshot(cache: dict, payload: dict, stale_timeout: float) -> dict:
    robots = dict(cache.get("robots", {}))
    for slot, data in payload.get("robots", {}).items():
        previous = robots.get(slot)
        robots[slot] = board.RobotState(
            x=float(data.get("x", previous.x if previous else 0.0)),
            y=float(data.get("y", previous.y if previous else 0.0)),
            yaw=float(data.get("yaw", previous.yaw if previous else 0.0)),
            vx_world=float(data.get("vx", previous.vx_world if previous else 0.0)),
            vy_world=float(data.get("vy", previous.vy_world if previous else 0.0)),
            cmd_wz=float(data.get("wz", previous.cmd_wz if previous else 0.0)),
        )

    ball = cache.get("ball", board.BallState())
    ball_data = payload.get("ball")
    if ball_data is not None:
        ball = board.BallState(
            x=float(ball_data.get("x", ball.x)),
            y=float(ball_data.get("y", ball.y)),
            vx=float(ball_data.get("vx", ball.vx)),
            vy=float(ball_data.get("vy", ball.vy)),
        )

    status = payload.get("status", {})
    intents = {}
    for slot in ("team_a_1", "team_a_2", "team_b_1", "team_b_2"):
        item = status.get(slot, {})
        pose_age = item.get("pose_age")
        if not item.get("pose_seen"):
            state = "MISS"
        elif pose_age is None or pose_age <= stale_timeout:
            state = "LIVE"
        else:
            state = "STALE"
        intents[slot] = {"role": ROLE_LABELS.get(slot, "?"), "state": state}

    return {
        "robots": robots,
        "ball": ball,
        "game_state": payload.get("game_state") or cache.get("game_state", "REMOTE"),
        "score": payload.get("score") or cache.get("score", {"team_a": 0, "team_b": 0}),
        "intents": intents,
        "paths": {
            name: [(p["x"], p["y"]) for p in pts]
            for name, pts in payload.get("paths", {}).items()
            if pts
        } or cache.get("paths", {}),
        "router_status": status,
        "remote_stamp": payload.get("stamp"),
        "role_states": payload.get("role_states") or cache.get("role_states", {}),
        "cmd_vels": payload.get("cmd_vels") or cache.get("cmd_vels", {}),
    }


def configure_board_window() -> tuple[pygame.Surface, board.Renderer]:
    pygame.init()
    screen = pygame.display.set_mode((board.SCREEN_W, board.SCREEN_H))
    pygame.display.set_caption("CyberDog Soccer - Remote Tactical Board")

    available_w = board.SCREEN_W - 2 * board.FIELD_MARGIN - board.SIDEBAR_W
    available_h = board.SCREEN_H - 2 * board.FIELD_MARGIN
    field_aspect = board.FIELD_LENGTH / board.FIELD_WIDTH
    field_w = available_w
    field_h = int(round(field_w / field_aspect))
    if field_h > available_h:
        field_h = available_h
        field_w = int(round(field_h * field_aspect))
    board.FIELD_LEFT = board.FIELD_MARGIN + max((available_w - field_w) // 2, 0)
    board.FIELD_TOP = board.FIELD_MARGIN + max((available_h - field_h) // 2, 0)
    return screen, board.Renderer(screen, field_w, field_h)


def draw_router_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    router: SshRosTopicRouter,
    snap: dict,
    stderr_lines: list[str],
) -> None:
    x = board.SCREEN_W - board.SIDEBAR_W + 5
    y = board.SCREEN_H - 200
    status = "SSH LIVE" if router.running else f"SSH EXIT {router.returncode}"
    color = board.GREEN if router.running else board.ORANGE
    screen.blit(font.render(status, True, color), (x, y))
    y += 16

    router_status = snap.get("router_status", {})
    missing = []
    stale = []
    for slot, item in router_status.items():
        if not item.get("pose_seen"):
            missing.append(slot)
        elif item.get("pose_age") is not None and item["pose_age"] > 0.6:
            stale.append(slot)
    if missing:
        screen.blit(font.render("missing: " + ",".join(missing[:4]), True, board.ORANGE), (x, y))
        y += 14
    if stale:
        screen.blit(font.render("stale: " + ",".join(stale[:4]), True, board.YELLOW), (x, y))
        y += 14
    if not missing and not stale and router_status:
        screen.blit(font.render("topics: ok", True, board.CYAN), (x, y))
        y += 14

    role_states = snap.get("role_states", {})
    striker_state = role_states.get("striker", "")
    if striker_state:
        state_color = {
            "approach": board.CYAN,
            "dribble": board.YELLOW,
            "recover": board.ORANGE,
            "waiting": board.GRAY,
        }.get(striker_state.lower(), board.LIGHT_GRAY)
        screen.blit(font.render(f"state: {striker_state}", True, state_color), (x, y))
        y += 14

    cmd_vels = snap.get("cmd_vels", {})
    cmd = cmd_vels.get("striker")
    if cmd:
        vx = cmd.get("vx", 0.0)
        vy = cmd.get("vy", 0.0)
        wz = cmd.get("wz", 0.0)
        screen.blit(font.render(f"vx:{vx:+.2f} vy:{vy:+.2f}", True, board.LIGHT_GRAY), (x, y))
        y += 14
        screen.blit(font.render(f"wz:{wz:+.2f}", True, board.LIGHT_GRAY), (x, y))
        y += 14

    for line in stderr_lines[-3:]:
        text = line[:34]
        screen.blit(font.render(text, True, board.GRAY), (x, y))
        y += 14


def main() -> int:
    args = parse_args()
    config = make_router_config(args)
    if args.print_command:
        print(" ".join(build_ssh_command(config)))
        return 0

    router = SshRosTopicRouter(config)
    router.start()

    screen, renderer = configure_board_window()
    overlay_font = pygame.font.SysFont("monospace", 11)
    clock = pygame.time.Clock()
    snap = initial_snapshot()
    stderr_lines: list[str] = []
    next_restart_at = 0.0
    running = True

    try:
        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_v:
                        renderer.show_vel = not renderer.show_vel
                    elif event.key == pygame.K_d:
                        renderer.show_debug = not renderer.show_debug
                    elif event.key == pygame.K_r:
                        router.stop()
                        router.start()

            latest = router.latest_snapshot()
            if latest is not None:
                snap = update_snapshot(snap, latest, args.stale_timeout)
            new_stderr = router.latest_stderr()
            if new_stderr:
                stderr_lines = (stderr_lines + new_stderr)[-8:]

            now = time.monotonic()
            if not router.running and now >= next_restart_at:
                next_restart_at = now + 3.0
                try:
                    router.start()
                except Exception as exc:
                    stderr_lines = (stderr_lines + [str(exc)])[-8:]

            renderer.draw(snap, match_time=0.0)
            draw_router_overlay(screen, overlay_font, router, snap, stderr_lines)
            pygame.display.flip()
    finally:
        router.stop()
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
