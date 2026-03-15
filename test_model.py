"""
测试训练好的模型 - 自动列出所有模型供选择
"""
import os
import numpy as np
from stable_baselines3 import PPO
from envs.medipick_arm_0 import MediPickArm0Env
import imageio
import glob


def list_models():
    """列出所有训练好的模型，按时间排序"""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("没有找到 models 目录")
        return []
    
    # 查找所有模型目录
    model_dirs = []
    for d in os.listdir(models_dir):
        full_path = os.path.join(models_dir, d)
        if os.path.isdir(full_path):
            # 检查是否有模型文件
            for f in ["final_model.zip", "best_model.zip"]:
                model_file = os.path.join(full_path, f)
                if os.path.exists(model_file):
                    model_dirs.append({
                        "name": d,
                        "path": model_file,
                        "time": os.path.getmtime(full_path)
                    })
                    break
    
    # 按时间排序（最新的在前）
    model_dirs.sort(key=lambda x: x["time"], reverse=True)
    
    return model_dirs


def select_model():
    """让用户选择模型"""
    models = list_models()
    
    if not models:
        print("没有找到训练好的模型")
        return None
    
    print("\n" + "=" * 60)
    print("可用的模型列表（按时间排序）:")
    print("=" * 60)
    
    for i, m in enumerate(models):
        import datetime
        time_str = datetime.datetime.fromtimestamp(m["time"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{i+1}] {m['name']} ({time_str})")
    
    print("=" * 60)
    
    while True:
        try:
            choice = input("\n选择模型编号 (直接回车选择第一个): ").strip()
            if not choice:
                return models[0]["path"]
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]["path"]
            else:
                print("编号无效，请重新选择")
        except ValueError:
            print("请输入有效的编号")


def test_model(model_path, video_path=None, episodes=3, render_mode='human'):
    """测试模型并可选录制视频"""
    
    print(f"\n加载模型: {model_path}")
    
    # 创建环境
    env = MediPickArm0Env(model_path="models/scene.xml", render_mode=render_mode)
    
    # 加载模型
    model = PPO.load(model_path)
    
    if video_path:
        # 录制视频
        print(f"录制视频到: {video_path}")
        frames = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1
            
            print(f"Episode {episode+1}: reward={episode_reward:.2f}, steps={step}")
        
        # 保存视频
        if frames:
            imageio.mimwrite(video_path, frames, fps=30)
            print(f"视频已保存: {video_path}")
    else:
        # 交互式显示
        print("\n开始测试（按 Ctrl+C 退出）...")
        try:
            for episode in range(episodes):
                obs, _ = env.reset()
                episode_reward = 0
                done = False
                step = 0
                
                while not done:
                    env.render()
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    step += 1
                
                print(f"Episode {episode+1}: reward={episode_reward:.2f}, steps={step}")
        except KeyboardInterrupt:
            print("\n退出")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试训练好的模型")
    parser.add_argument('--model', type=str, default=None, help='模型路径（不指定则让用户选择）')
    parser.add_argument('--video', type=str, default=None, help='视频输出路径')
    parser.add_argument('--episodes', type=int, default=3, help='测试回合数')
    parser.add_argument('--no-display', action='store_true', help='不显示，只录制')
    parser.add_argument('--latest', action='store_true', help='自动选择最新的模型')
    
    args = parser.parse_args()
    
    # 选择模型
    if args.model:
        model_path = args.model
    elif args.latest:
        models = list_models()
        if models:
            model_path = models[0]["path"]
            print(f"自动选择最新模型: {models[0]['name']}")
        else:
            print("没有找到模型")
            exit(1)
    else:
        model_path = select_model()
    
    if not model_path:
        print("没有选择模型")
        exit(1)
    
    # 自动检测是否可以使用 human 模式
    if args.video:
        render_mode = 'rgb_array'
    else:
        # 尝试检测是否可以使用 human 模式
        try:
            import glfw
            render_mode = 'human'
        except:
            print("警告: 无法初始化显示窗口，使用 rgb_array 模式并保存视频")
            render_mode = 'rgb_array'
            args.video = 'test_output.mp4'
    
    test_model(model_path, args.video, args.episodes, render_mode)
