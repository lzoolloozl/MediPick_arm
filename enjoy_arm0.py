"""
MediPick Arm0 可视化/演示脚本

用于加载训练好的模型并可视化执行效果

使用方法:
    python enjoy_arm0.py                              # 使用最佳模型
    python enjoy_arm0.py models/medipick_arm0/xxx.zip # 使用指定模型
"""
import sys
import numpy as np
from stable_baselines3 import PPO
from envs.medipick_arm_0 import MediPickArm0Env


def enjoy(model_path=None, num_episodes=3, max_steps=1000):
    """
    加载模型并可视化执行
    
    参数:
        model_path: 模型文件路径,如果不提供则尝试加载best_model
        num_episodes: 执行多少个episode
        max_steps: 每个episode最多多少步
    """
    # ==================== 1. 加载模型 ====================
    if model_path is None:
        # 尝试默认路径
        default_paths = [
            "models/medipick_arm0/best_model/best_model.zip",
            "models/medipick_arm0/medipick_arm0_final.zip",
            "models/medipick_arm0/medipick_arm0_final"
        ]
        
        for path in default_paths:
            try:
                model = PPO.load(path)
                print(f"成功加载模型: {path}")
                model_path = path
                break
            except:
                continue
        else:
            print("错误: 无法找到训练好的模型!")
            print("请先运行训练: python train_arm0.py")
            return
    else:
        try:
            model = PPO.load(model_path)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            return

    # ==================== 2. 创建环境 ====================
    print("\n创建MuJoCo环境...")
    env = MediPickArm0Env(model_path="models/scene.xml", render_mode="human")
    
    print(f"观测空间: {env.observation_space.shape}")
    print(f"动作空间: {env.action_space.shape}")

    # ==================== 3. 执行演示 ====================
    print("\n" + "=" * 50)
    print("开始演示!")
    print("=" * 50)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while not done and episode_steps < max_steps:
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            # 渲染
            env.render()
            
            # 显示实时信息
            if episode_steps % 50 == 0:
                contact_dist = info.get('contact_dist', -1)
                print(f"步骤: {episode_steps}, 奖励: {reward:.2f}, 接触距离: {contact_dist:.4f}m")
        
        # 显示episode结果
        success = info.get('success', False)
        print(f"\nEpisode {episode + 1} 完成:")
        print(f"  总步数: {episode_steps}")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  成功: {'✓' if success else '✗'}")
        
        if success:
            print("  🎉 成功接触到目标!")
    
    print("\n" + "=" * 50)
    print("演示结束!")
    print("=" * 50)
    
    env.close()


if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"使用指定模型: {model_path}")
    else:
        model_path = None
    
    enjoy(model_path=model_path, num_episodes=3)
