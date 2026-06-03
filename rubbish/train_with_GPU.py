import gym
from stable_baselines3 import A2C
import torch


def main():
    # 1. 检测 GPU
    print("=" * 50)
    print("GPU 检测:")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"  显存大小: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = 'cuda'
    else:
        print("  GPU 不可用，使用 CPU")
        device = 'cpu'

    print(f"\n✅ 将使用设备: {device.upper()}")
    print("=" * 50)

    # 2. 创建环境
    env = gym.make("CartPole-v1")

    # 3. 创建模型 - 强制使用 GPU
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,  # 关键设置
        tensorboard_log='logs',
        learning_rate=7e-4
    )

    # 验证模型是否在正确的设备上
    print(f"\n模型实际设备: {model.device}")
    if model.device.type == 'cuda':
        print("🎉 模型已成功加载到 GPU！")
    else:
        print("⚠️ 警告：模型仍在 CPU 上")

    # 4. 训练
    print("\n开始训练...")
    model.learn(total_timesteps=10000)  # 先测试 10000 步
    print("训练完成！")


if __name__ == "__main__":
    main()