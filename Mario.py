import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
# 创建超级马里奥兄弟游戏环境，版本2
env = gym_super_mario_bros.make('SuperMarioBros-v2')
# 用简易动作空间包装环境（只有左右、跳跃等基本动作）
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#灰度处理
env=GrayScaleObservation(env,keep_dim=True)
# 创建PPO模型，使用卷积神经网络策略，verbose=1表示打印训练日志
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log='log')
# 开始训练，总共训练1000步
model.learn(total_timesteps=1000)
model.save("ppo_mario")
