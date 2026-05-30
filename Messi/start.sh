#!/usr/bin/env bash
# Convenience launcher for the robot. Keep the real startup logic in main.py so
# the same entry point works from shell scripts, IDEs and direct SSH sessions.

set -euo pipefail

python3 main.py --game-state PLAYING "$@" --config config/config_local.yaml
