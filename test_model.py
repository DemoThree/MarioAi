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

model=PPO.load('ppo_mario.zip')

# 初始化环境，获取第一帧观察值
obs = env.reset()

# 循环执行10000次游戏步骤
for i in range(10000):
    # 复制当前观察值（用于修复负步长问题）
    obs = obs.copy()

    # 使用模型预测最佳动作，deterministic=True表示每次都选最优动作（不随机探索）
    action, _state = model.predict(obs, deterministic=True)

    # 执行动作，获取新状态：
    # obs   - 新的观察值（游戏画面）
    # reward - 获得的奖励分数
    # done   - 游戏是否结束（马里奥死亡或过关）
    # info   - 额外信息（如当前关卡、分数等）
    obs, reward, done, info = env.step(action)

    # 渲染游戏画面（显示窗口）
    env.render()

    # 如果游戏结束，重置环境重新开始
    if done:
        obs = env.reset()