"""Export a trained RSL-RL soccer policy to ONNX for Flamez deployment.

Usage:
    python scripts/export_soccer_policy.py --checkpoint <path_to_model.pt> --output <output.onnx>

Example:
    python scripts/export_soccer_policy.py \
        --checkpoint logs/rsl_rl/cyberdog2_soccer_2v2/2026-05-26_16-14-27_phase1_skill2/model_16199.pt \
        --output f:/xiaomidog-RL-main/Flamez_upload/checkpoints/striker_rl/phase1/best/best_model.onnx
"""

import argparse
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Pure MLP policy without obs_normalizer to avoid double normalization.

    Training env already normalizes observations, so we only export the MLP part.
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(x))


def export_policy(checkpoint_path: str, output_path: str):
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    sd = ckpt.get('actor_state_dict', ckpt)

    # Print normalizer stats for debugging
    if 'obs_normalizer._mean' in sd:
        print(f"[INFO] obs_mean: {sd['obs_normalizer._mean']}")
        print(f"[INFO] obs_std: {sd['obs_normalizer._std']}")

    model = PolicyNet()
    mlp_sd = {}
    for k, v in sd.items():
        if k.startswith('mlp.'):
            mlp_sd[k[4:]] = v
    model.mlp.load_state_dict(mlp_sd, strict=True)
    model.eval()
    print("[INFO] MLP weights loaded successfully")

    # Verify output is not constant
    test_inputs = torch.randn(5, 12)
    with torch.no_grad():
        out = model(test_inputs)
    print(f"[INFO] Test outputs (should vary):\n{out}")

    # Export to ONNX
    dummy = torch.zeros(1, 12)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=15,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch'}, 'action': {0: 'batch'}}
    )
    print(f"[INFO] Exported ONNX opset 15 to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export soccer policy to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--output', required=True, help='Output .onnx path')
    args = parser.parse_args()

    export_policy(args.checkpoint, args.output)
