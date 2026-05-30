import unittest

import yaml

from sim.ssh_topic_router import (
    RemoteRouterConfig,
    TopicSpec,
    apply_mapping_overrides,
    build_remote_shell_command,
    load_default_slot_mapping,
    topic_specs_from_mapping,
)


class SshTopicRouterTests(unittest.TestCase):
    def test_load_default_slot_mapping_from_runtime_config(self):
        mapping = load_default_slot_mapping("config/config.yaml")

        with open("config/config.yaml", encoding="utf-8") as stream:
            rigids = yaml.safe_load(stream)["rigids"]

        self.assertEqual(mapping["team_a_1"], rigids["self"])
        self.assertEqual(mapping["team_a_2"], rigids["teammate"])
        self.assertEqual(mapping["team_b_1"], rigids["opponent_1"])
        self.assertEqual(mapping["team_b_2"], rigids["opponent_2"])
        self.assertEqual(mapping["ball"], rigids["ball"])

    def test_mapping_overrides_accept_rigid_or_full_topic(self):
        mapping = apply_mapping_overrides(
            {"team_a_1": "betago2", "ball": "greenball"},
            ["team_a_1=SoTaGo1", "ball=/vrpn/ball/pose"],
        )
        specs = {spec.slot: spec for spec in topic_specs_from_mapping(mapping)}

        self.assertEqual(specs["team_a_1"].pose_topic, "/vrpn/SoTaGo1/pose")
        self.assertEqual(specs["team_a_1"].twist_topic, "/vrpn/SoTaGo1/twist")
        self.assertEqual(specs["ball"].pose_topic, "/vrpn/ball/pose")
        self.assertEqual(specs["ball"].twist_topic, "/vrpn/ball/twist")
        self.assertEqual(specs["ball"].kind, "ball")

    def test_remote_shell_sources_setup_and_runs_router(self):
        command = build_remote_shell_command(
            RemoteRouterConfig(
                remote_workspace="~/cyberdog_soccer",
                remote_setups=("/opt/ros2/cyberdog/setup.bash",),
                topics=(TopicSpec("team_a_1", "/vrpn/SoTaGo1/pose"),),
                hz=12.5,
            )
        )

        self.assertIn("source /opt/ros2/cyberdog/setup.bash", command)
        self.assertIn("cd $HOME/cyberdog_soccer", command)
        self.assertIn("exec python3 -u -", command)
        self.assertIn("--hz 12.5", command)


if __name__ == "__main__":
    unittest.main()
