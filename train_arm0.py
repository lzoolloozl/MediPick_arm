"""
MediPick Arm0 PPO训练脚本 - 参考 so100-mujoco-rl 项目
"""
import os
import torch
import numpy as np
from datetime import datetime
import click
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.medipick_arm_0 import MediPickArm0Env

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# 目录配置
MODEL_DIR = "models"
LOG_DIR = "logs"
RECORDING_DIR = "videos"  # 视频保存目录


class TensorboardRewardCallback(BaseCallback):
    """将评估奖励记录到tensorboard"""
    def __init__(self, eval_callback, verbose=0):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        
    def _on_step(self) -> bool:
        if hasattr(self.eval_callback, 'last_mean_reward'):
            reward = self.eval_callback.last_mean_reward
            if reward == float('-inf'):
                reward = -1000.0
            self.logger.record("rollout/ep_rew_mean", reward)
        return True


def make_env(render_mode='rgb_array'):
    """创建环境"""
    env = MediPickArm0Env(model_path="models/scene.xml", render_mode=render_mode)
    env = Monitor(env)
    return env


@click.command()
@click.option('--record/--no-record', default=False, help='启用视频录制')
@click.option('--steps', default=10000, type=int, help='训练步数')
def main(record, steps):
    """训练脚本"""
    # 创建目录
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)
    
    # 使用时间戳创建唯一目录
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"{MODEL_DIR}/medipick_arm0_{run_id}"
    log_dir = f"{LOG_DIR}/arm0_run_{run_id}"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 50)
    print(f"MediPick Arm0 PPO 训练")
    print(f"训练ID: {run_id}")
    print(f"训练步数: {steps}")
    print(f"视频录制: {'启用' if record else '禁用'}")
    print("=" * 50)
    
    # 创建环境
    env = make_env(render_mode='rgb_array')
    
    # 添加视频录制（可选）
    if record:
        try:
            import gymnasium as gym
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=RECORDING_DIR,
                video_length=0,
                episode_trigger=lambda x: x % 50 == 0,
                name_prefix=f"medipick_{run_id}"
            )
            print("视频录制已启用")
        except ImportError:
            print("警告: moviepy 未安装，跳过视频录制")
    
    env = DummyVecEnv([lambda: env])
    
    # 创建模型
    model = PPO(
        "MlpPolicy",
        env=env,
        verbose=1,
        device='cpu',
        tensorboard_log=log_dir,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        n_epochs=10,
        gamma=0.99,
    )
    
    print(f"使用设备: {'cpu'}")
    print(f"日志目录: {log_dir}")
    print("=" * 50)
    print("【重要】查看训练数据的命令:")
    print(f"tensorboard --logdir {os.path.abspath(log_dir)}")
    print("=" * 50)
    
    # 评估回调 - 使用较大的 eval_freq 避免阻塞
    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=20000,  # 改成20000，避免频繁评估
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="arm0_step"
    )
    
    # 奖励记录回调
    tb_reward_callback = TensorboardRewardCallback(eval_callback, verbose=0)
    
    # 开始训练
    try:
        model.learn(
            total_timesteps=steps,
            callback=CallbackList([
                checkpoint_callback,
                eval_callback,
                tb_reward_callback
            ]),
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\n训练被手动停止")
    finally:
        # 保存最终模型
        final_path = f"{model_dir}/final_model"
        model.save(final_path)
        print(f"模型已保存到: {final_path}")
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()
