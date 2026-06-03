import gym
from stable_baselines3 import A2C
import torch
import time

# 1. 检查 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# 2. 训练
env = gym.make("CartPole-v1")
model = A2C("MlpPolicy", env, verbose=1, device=device)
print("🏋️ 开始训练...")
model.learn(total_timesteps=1000)
model.save("a2c_cartpole_gpu")
print("✅ 训练完成！")

# 3. 测试并显示画面
print("\n🎬 开始测试，显示画面...")
test_env = gym.make("CartPole-v1")

# 重置环境
obs = test_env.reset()
done = False
total_reward = 0

while not done:
    # 渲染画面
    test_env.render()

    # 预测动作
    action, _ = model.predict(obs, deterministic=True)

    # 执行动作
    obs, reward, done, info = test_env.step(action)
    total_reward += reward

    # 控制速度（太快看不清）
    time.sleep(0.05)

print(f"📊 测试得分: {total_reward}")
test_env.close()