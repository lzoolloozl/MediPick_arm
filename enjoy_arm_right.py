"""
MediPick 右臂可视化/演示脚本

用于加载训练好的模型并可视化执行效果

使用方法:
    python enjoy_arm_right.py                              # 使用最佳模型
    python enjoy_arm_right.py models/medipick_arm_right_xxx.zip # 使用指定模型
    python enjoy_arm_right.py --record                     # 录制视频
    python enjoy_arm_right.py --record models/xxx.zip     # 录制视频并指定模型
"""
import sys
import glob
import os
import numpy as np
import click
from stable_baselines3 import PPO
from envs.medipick_arm_right import MediPickArmRightEnv


def enjoy(model_path=None, record=False, num_episodes=3, max_steps=1000):
    """
    加载模型并可视化执行
    
    参数:
        model_path: 模型文件路径,如果不提供则尝试加载best_model
        record: 是否录制视频
        num_episodes: 执行多少个episode
        max_steps: 每个episode最多多少步
    """
    # ==================== 1. 加载模型 ====================
    if model_path is None:
        # 尝试默认路径 - 按时间戳排序找最新的
        model_dirs = sorted(glob.glob("models/medipick_arm_right_*/"), reverse=True)
        
        # 默认路径列表
        default_paths = []
        for d in model_dirs:
            # 添加 best_model
            best = os.path.join(d, "best_model", "best_model.zip")
            if os.path.exists(best):
                default_paths.append(best)
            # 添加 final_model
            final = os.path.join(d, "final_model.zip")
            if os.path.exists(final):
                default_paths.append(final)
        
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
            print("请先运行训练: python train_arm_right.py")
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
    
    if record:
        # 录制视频模式 - 使用固定摄像头
        env = MediPickArmRightEnv(model_path="models/scene.xml", render_mode="rgb_array")
        
        # 使用 gymnasium 的 RecordVideo
        from gymnasium.wrappers import RecordVideo
        video_folder = "videos"
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=lambda x: True,
            video_length=max_steps,
            name_prefix="medipick_right"
        )
        print(f"视频将保存到: {video_folder}/")
    else:
        # 交互模式 - 使用固定摄像头
        env = MediPickArmRightEnv(model_path="models/scene.xml", render_mode="human")
    
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


@click.command()
@click.option('--record', is_flag=True, help='录制视频')
@click.argument('model_path', required=False, default=None)
def main(record, model_path):
    """MediPick 右臂模型演示"""
    enjoy(model_path=model_path, record=record, num_episodes=3)


if __name__ == "__main__":
    main()
