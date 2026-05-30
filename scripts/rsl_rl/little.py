# 仅创建环境，不训练，看看能否正常 reset
python -c "
from isaaclab.app import AppLauncher
import gymnasium as gym
import cli_args
# 简化启动
app = AppLauncher({})
sim = app.app
env = gym.make('Cyberdog2-Soccer-2v2-v0', scene={'num_envs': 4})
obs = env.reset()
print('Observation shape:', obs.shape)
env.close()
"