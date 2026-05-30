"""Register all Cyberdog2 IsaacLab tasks."""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = []
import_packages(__name__, _BLACKLIST_PKGS)
