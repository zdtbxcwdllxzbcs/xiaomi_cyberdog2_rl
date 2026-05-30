"""Export a trained locomotion policy to TorchScript and ONNX for the soccer task."""

import argparse
import os
import torch
import torch.nn as nn


def export(checkpoint_path: str, output_dir: str):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    sd = ckpt["actor_state_dict"]

    # RSL-RL MLP actor structure (observed from checkpoint):
    #   mlp.0: Linear(in_dim, 512)
    #   mlp.2: Linear(512, 256)
    #   mlp.4: Linear(256, 128)
    #   mlp.6: Linear(128, action_dim)
    # Each Linear (except last) has an implicit ELU activation following it.
    in_dim = sd["mlp.0.weight"].shape[1]
    action_dim = sd["mlp.6.weight"].shape[0]

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, action_dim),
            )

        def forward(self, x):
            return self.mlp(x)

    model = PolicyNet()

    layer_keys = ["mlp.0", "mlp.2", "mlp.4", "mlp.6"]
    for i, key in enumerate(layer_keys):
        idx = i * 2  # each block: Linear (idx*2), ELU (idx*2+1)
        model.mlp[idx].weight.data = sd[f"{key}.weight"].clone()
        model.mlp[idx].bias.data = sd[f"{key}.bias"].clone()

    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    pt_path = os.path.join(output_dir, "policy.pt")
    onnx_path = os.path.join(output_dir, "policy.onnx")

    scripted = torch.jit.script(model)
    scripted.save(pt_path)
    print(f"Exported TorchScript: {pt_path}")

    dummy = torch.zeros(1, in_dim)
    torch.onnx.export(
        model, dummy, onnx_path,
        export_params=True,
        opset_version=15,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"Exported ONNX: {onnx_path}")

    test_out = model(dummy)
    print(f"  Input dim: {in_dim}, Action dim: {action_dim}")
    print(f"  Output sample: {test_out[0, :3].tolist()} ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model_X.pt checkpoint")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: <checkpoint_dir>/exported)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint), "exported")

    export(args.checkpoint, args.output_dir)
