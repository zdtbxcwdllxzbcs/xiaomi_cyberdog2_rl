import torch, yaml

ckpt_path = r'f:\dog\cyberdog2_rl_lab\logs\rsl_rl\cyberdog2_soccer_2v2\2026-05-28_08-06-07_phase1_skill1\model_9999.pt'
out_path = r'f:\xiaomidog-RL-main\Flamez_upload\checkpoints\striker_rl\phase1\best\best_model.onnx'

ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
sd = ckpt['actor_state_dict']

class PolicyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_mean = torch.nn.Parameter(sd['obs_normalizer._mean'].clone())
        self.obs_std = torch.nn.Parameter(sd['obs_normalizer._std'].clone().clamp_min(1e-5))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(12, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 3),
        )

    def forward(self, x):
        x = (x - self.obs_mean) / self.obs_std
        return torch.tanh(self.mlp(x))

model = PolicyNet()
# Strip 'mlp.' prefix
mlp_sd = {}
for k, v in sd.items():
    if k.startswith('mlp.'):
        mlp_sd[k[4:]] = v
model.mlp.load_state_dict(mlp_sd, strict=True)
model.eval()

dummy = torch.zeros(1, 12)
torch.onnx.export(model, dummy, out_path, export_params=True, opset_version=15,
    input_names=['observation'], output_names=['action'],
    dynamic_axes={'observation': {0: 'batch'}, 'action': {0: 'batch'}})
print(f'Exported opset 15 to {out_path}')

# Verify
import onnxruntime as ort
sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
out = sess.run(None, {'observation': torch.zeros(1, 12).numpy()})
print(f'Inference test: {out[0]}')