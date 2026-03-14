"""
MediPick Arm0 PPO训练脚本

任务目标: 让吸盘末端(sucker_tip)与药盒前中心点(box_front_center)垂直接触

使用方法:
    python train_arm0.py

查看训练过程:
    tensorboard --logdir logs/tensorboard/arm0

视频录像:
    训练过程中会自动录制评估视频到 videos/ 目录
"""
import os
import torch
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from envs.medipick_arm_0 import MediPickArm0Env


class TensorboardRewardCallback(BaseCallback):
    """
    自定义回调 - 将评估奖励记录到tensorboard
    用于解决EvalCallback不自动记录评估奖励到tensorboard的问题
    """
    def __init__(self, eval_callback, verbose=0):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # 在每个评估步骤后获取奖励并记录到tensorboard
        if hasattr(self.eval_callback, 'last_mean_reward'):
            # 记录评估奖励
            self.logger.record("rollout/ep_rew_mean", self.eval_callback.last_mean_reward)
        return True


class VideoRecorderCallback(BaseCallback):
    """
    自定义视频录制回调 - 每个eval_freq录制一次评估视频
    """
    def __init__(self, eval_env, video_dir, eval_freq=2000, video_length=500, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_dir = video_dir
        self.eval_freq = eval_freq
        self.video_length = video_length
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # 每eval_freq步录制一次视频
        if self.n_calls % self.eval_freq == 0:
            self._record_video()
        return True
    
    def _record_video(self):
        """录制一段评估视频"""
        video_path = os.path.join(self.video_dir, f"video_{self.n_calls:08d}.mp4")
        os.makedirs(self.video_dir, exist_ok=True)
        
        # 创建临时环境用于录制
        env = MediPickArm0Env(model_path="models/scene.xml")
        
        # 重置环境
        obs, _ = env.reset()
        
        frames = []
        total_reward = 0
        done = False
        step_count = 0
        
        # 运行一个episode或达到视频长度
        while not done and step_count < self.video_length:
            # 渲染一帧
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # 使用模型预测动作
            action, _ = self.model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
        
        # 确保至少有最后一帧
        if len(frames) > 0:
            # 录制最后几帧以确保视频完整
            for _ in range(10):
                frames.append(env.render())
        
        # 保存视频
        if len(frames) > 0:
            try:
                imageio.mimwrite(video_path, frames, fps=30, codec='libx264', quality=8)
                if self.verbose > 0:
                    print(f"Video saved to {video_path}, reward: {total_reward:.2f}, steps: {step_count}")
            except Exception as e:
                print(f"Error saving video: {e}")
        
        env.close()


def make_env():
    """创建并包装环境"""
    # 使用models/scene.xml作为模型文件
    env = MediPickArm0Env(model_path="models/scene.xml")
    # Monitor用于记录episode统计信息
    env = Monitor(env)
    return env


def train():
    """PPO训练主函数"""
    # ==================== 1. 路径与环境配置 ====================
    model_dir = "models/medipick_arm0"
    log_dir = "logs/tensorboard/arm0"
    video_dir = "videos"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    print("=" * 50)
    print("MediPick Arm0 PPO 训练")
    print("任务: 吸盘末端触碰药盒前中心")
    print("=" * 50)

    # 创建环境
    env = DummyVecEnv([make_env])
    
    # 获取环境信息
    obs_space = env.observation_space
    action_space = env.action_space
    print(f"观测空间维度: {obs_space.shape}")
    print(f"动作空间维度: {action_space.shape}")
    print(f"动作范围: [{action_space.low[0]:.2f}, {action_space.high[0]:.2f}]")

    # ==================== 2. 神经网络架构配置 ====================
    # 针对机器人控制设计的多层感知机
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,  # ReLU激活函数
        # 策略网络: 2层256神经元
        net_arch=dict(
            pi=[256, 256],    # 策略网络
            vf=[256, 256]     # 值函数网络
        )
    )

    # ==================== 3. PPO超参数 ====================
    hyperparams = {
        "n_steps": 2048,           # 每次更新前收集的样本数
        "batch_size": 64,          # 优化时的批大小
        "gamma": 0.99,             # 折扣因子(未来奖励的衰减)
        "learning_rate": 3e-4,      # 学习率
        "ent_coef": 0.01,          # 熵系数(鼓励探索)
        "clip_range": 0.2,         # PPO裁剪范围
        "clip_range_vf": None,     # 值函数裁剪(不使用)
        "n_epochs": 10,            # 每次更新时的迭代次数
        "gae_lambda": 0.95,        # GAE参数
        "max_grad_norm": 0.5,      # 梯度裁剪
        "vf_coef": 0.5,           # 值函数损失系数
        "device": "cpu",           # 使用CPU训练MLP策略更稳定
        "verbose": 1
    }

    print(f"\n使用设备: {hyperparams['device']}")
    print(f"总步数: {hyperparams['n_steps']}")
    print(f"批大小: {hyperparams['batch_size']}")
    print(f"学习率: {hyperparams['learning_rate']}")

    # ==================== 4. 初始化PPO模型 ====================
    model = PPO(
        "MlpPolicy",               # 使用多层感知机策略
        env,                       # 环境
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        **hyperparams
    )

    # ==================== 5. 设置回调函数 ====================
    # 检查点回调: 每10000步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="arm0_step"
    )

    # 评估回调: 每2000步评估一次并保存最佳模型
    eval_callback = EvalCallback(
        env, 
        best_model_save_path=f"{model_dir}/best_model",
        log_path=f"{model_dir}/eval_results", 
        eval_freq=2000,
        deterministic=True, 
        render=False,
        n_eval_episodes=10
    )

    # 视频录制回调: 每2000步录制一次评估视频
    video_callback = VideoRecorderCallback(
        eval_env=env,
        video_dir=video_dir,
        eval_freq=2000,
        video_length=500,
        verbose=1
    )

    # tensorboard奖励记录回调: 将评估奖励记录到tensorboard
    tb_reward_callback = TensorboardRewardCallback(
        eval_callback=eval_callback,
        verbose=0
    )

    # ==================== 6. 开始训练 ====================
    total_timesteps = 2000000  # 总训练步数
    
    print(f"\n开始训练! 总步数: {total_timesteps}")
    print(f"日志目录: {log_dir}")
    print(f"视频目录: {video_dir}")
    print("提示: 运行 'tensorboard --logdir logs/tensorboard/arm0' 查看实时数据")
    print("提示: 可以在tensorboard中查看 'rollout/ep_rew_mean' 曲线评估奖励")
    print("提示: 视频每2000步录制一次，保存在 videos/ 目录")
    print("-" * 50)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, video_callback, tb_reward_callback],
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\n训练被手动停止，正在保存当前模型...")
    finally:
        # 保存最终模型
        final_path = f"{model_dir}/medipick_arm0_final"
        model.save(final_path)
        print(f"模型已保存到: {final_path}")
        
    print("\n训练完成!")


def evaluate(num_episodes=5):
    """评估训练好的模型"""
    from stable_baselines3.common.evaluation import evaluate_policy
    
    model_dir = "models/medipick_arm0/best_model"
    
    print("=" * 50)
    print("开始评估模型")
    print("=" * 50)
    
    # 加载最佳模型
    try:
        model = PPO.load(model_dir)
        print(f"已加载模型: {model_dir}")
    except:
        # 如果没有最佳模型,尝试加载最终模型
        model = PPO.load(f"{model_dir.replace('best_model', 'medipick_arm0_final')}")
        print(f"已加载模型: {model_dir.replace('best_model', 'medipick_arm0_final')}")
    
    # 创建评估环境
    env = MediPickArm0Env(model_path="models/scene.xml")
    
    # 评估
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=num_episodes,
        render=True,
        deterministic=True
    )
    
    print(f"\n评估结果 (共{num_episodes}个episode):")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "eval":
            # 评估模式
            n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            evaluate(n_eval)
        else:
            print("用法:")
            print("  python train_arm0.py        # 训练")
            print("  python train_arm0.py eval    # 评估(5个episode)")
            print("  python train_arm0.py eval 10 # 评估(10个episode)")
    else:
        # 训练模式
        train()
