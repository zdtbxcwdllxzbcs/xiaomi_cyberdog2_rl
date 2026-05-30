"""Compatibility wrapper for the old single-rigid subscriber name.

New role code should use ``LocatorNode`` from ``src/lib/locator.py``. This file
keeps older imports working while the refactor settles, and maps a single VRPN
rigid into the same locator publishing pipeline.
"""

from .locator import LocatorNode, ObjectState


class LocationSubscriber(LocatorNode):
    """Deprecated single-rigid adapter backed by ``LocatorNode``."""

    def __init__(self, name="location_subscriber", rigid=None, object_name="object", config=None):
        if config is None:
            rigid_name = rigid or object_name
            config = {
                "rigids": {object_name: rigid_name},
                "locator": {"required_objects": []},
            }
        self.default_object = object_name
        super().__init__(config=config, name=name)

    def get_pose(self, object_name=None):
        return super().get_pose(object_name or self.default_object)

    def get_twist(self, object_name=None):
        return super().get_twist(object_name or self.default_object)


__all__ = ["LocationSubscriber", "LocatorNode", "ObjectState"]
