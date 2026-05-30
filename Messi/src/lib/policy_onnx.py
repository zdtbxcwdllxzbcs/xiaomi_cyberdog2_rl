"""Lightweight ONNX-based policy inference for robot deployment.

Provides a drop-in replacement for the SB3 PPO model on systems where
stable-baselines3 / PyTorch cannot be installed (e.g. glibc 2.27 + Python 3.6).

Generate the .onnx file on the dev machine:
    python scripts/export_policy_onnx.py

Install on robot (Python 3.6, glibc 2.27):
    pip3 install onnxruntime==1.10.0
"""

import numpy as np


class OnnxPolicy:
    """Drop-in replacement for SB3 PPO.predict() using onnxruntime."""

    def __init__(self, model_path):
        # type: (str) -> None
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, obs, deterministic=True):
        # type: (np.ndarray, bool) -> tuple
        obs_f32 = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action = self._session.run(None, {self._input_name: obs_f32})[0]
        return action.flatten(), None  # matches SB3 (action, state) tuple
