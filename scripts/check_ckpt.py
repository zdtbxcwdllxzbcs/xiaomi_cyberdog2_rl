import torch, os
ckpt_path = r"F:\dog\cyberdog2_rl_lab\logs\rsl_rl\cyberdog2_velocity_flat\2026-05-27_05-04-31\model_25400.pt"
ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
sd = ckpt['actor_state_dict']
print('Keys:', list(sd.keys())[:20])
for k in sorted(sd.keys()):
    if 'weight' in k or 'bias' in k:
        print(f'  {k}: {sd[k].shape}')
