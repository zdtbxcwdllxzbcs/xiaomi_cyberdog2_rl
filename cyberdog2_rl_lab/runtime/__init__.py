"""Runtime bridges for Cyberdog2 deployment."""

__all__ = [
    "OfficialCommandPublisher",
    "OfficialLocomotionObservationBuilder",
    "OfficialLocomotionPolicyRunner",
    "OfficialLocomotionPolicyRunnerConfig",
    "OfficialLocomotionRuntimeState",
    "OfficialPublisherConfig",
    "RigidBodyState",
    "SoccerObservationBuilder",
    "SoccerPolicyRunner",
    "SoccerPolicyRunnerConfig",
    "SoccerWorldState",
    "VrpnStateReceiver",
    "VrpnStateReceiverConfig",
]


def __getattr__(name: str):
    if name in {"OfficialCommandPublisher", "OfficialPublisherConfig"}:
        from .official_cmd_publisher import OfficialCommandPublisher, OfficialPublisherConfig

        values = {
            "OfficialCommandPublisher": OfficialCommandPublisher,
            "OfficialPublisherConfig": OfficialPublisherConfig,
        }
    elif name in {
        "OfficialLocomotionObservationBuilder",
        "OfficialLocomotionPolicyRunner",
        "OfficialLocomotionPolicyRunnerConfig",
        "OfficialLocomotionRuntimeState",
    }:
        from .official_locomotion_policy_runner import (
            OfficialLocomotionObservationBuilder,
            OfficialLocomotionPolicyRunner,
            OfficialLocomotionPolicyRunnerConfig,
            OfficialLocomotionRuntimeState,
        )

        values = {
            "OfficialLocomotionObservationBuilder": OfficialLocomotionObservationBuilder,
            "OfficialLocomotionPolicyRunner": OfficialLocomotionPolicyRunner,
            "OfficialLocomotionPolicyRunnerConfig": OfficialLocomotionPolicyRunnerConfig,
            "OfficialLocomotionRuntimeState": OfficialLocomotionRuntimeState,
        }
    elif name in {"SoccerObservationBuilder", "SoccerPolicyRunner", "SoccerPolicyRunnerConfig"}:
        from .soccer_policy_runner import SoccerObservationBuilder, SoccerPolicyRunner, SoccerPolicyRunnerConfig

        values = {
            "SoccerObservationBuilder": SoccerObservationBuilder,
            "SoccerPolicyRunner": SoccerPolicyRunner,
            "SoccerPolicyRunnerConfig": SoccerPolicyRunnerConfig,
        }
    elif name in {"RigidBodyState", "SoccerWorldState", "VrpnStateReceiver", "VrpnStateReceiverConfig"}:
        from .vrpn_state_receiver import RigidBodyState, SoccerWorldState, VrpnStateReceiver, VrpnStateReceiverConfig

        values = {
            "RigidBodyState": RigidBodyState,
            "SoccerWorldState": SoccerWorldState,
            "VrpnStateReceiver": VrpnStateReceiver,
            "VrpnStateReceiverConfig": VrpnStateReceiverConfig,
        }
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals().update(values)
    return globals()[name]
