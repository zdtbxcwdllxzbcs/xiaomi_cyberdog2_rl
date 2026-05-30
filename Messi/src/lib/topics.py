"""Topic-name helpers for global and per-robot soccer channels."""


def _clean_part(part) -> str:
    return str(part).strip("/")


def soccer_prefix(config) -> str:
    """Return the configured soccer topic prefix, defaulting to ``/soccer``."""

    topics = (config or {}).get("topics", {})
    prefix = topics.get("prefix") or topics.get("soccer_prefix") or "/soccer"
    prefix = "/" + _clean_part(prefix)
    return prefix if prefix != "/" else "/soccer"


def soccer_topic(config, *parts) -> str:
    """Join the configured soccer prefix with one or more topic path parts."""

    suffix = "/".join(_clean_part(part) for part in parts if str(part).strip("/"))
    prefix = soccer_prefix(config)
    return f"{prefix}/{suffix}" if suffix else prefix


def global_soccer_topic(*parts) -> str:
    """Build a topic under the shared match-level ``/soccer`` namespace."""

    suffix = "/".join(_clean_part(part) for part in parts if str(part).strip("/"))
    return f"/soccer/{suffix}" if suffix else "/soccer"
