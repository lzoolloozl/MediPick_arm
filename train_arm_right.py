"""
MediPick 右臂 PPO训练脚本 - 只有右臂(r1-r6)和升降杆(raise)可以动
支持无限训练直到成功率接近100%或无法继续学习才停止
支持训练时录制视频

修复版:
- 训练环境使用render_mode=None加快训练速度
- 支持训练时录制视频
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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.medipick_arm_right import MediPickArmRightEnv

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# 目录配置
MODEL_DIR = "models"
LOG_DIR = "logs"
RECORDING_DIR = "videos"  # 视频保存目录

# 训练停止条件
SUCCESS_THRESHOLD = 0.95  # 成功率阈值 (95%)
CONSECUTIVE_SUCCESS = 3  # 连续多少次评估达到阈值就停止
MAX_STEPS = 10000000     # 最大步数限制


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


def make_env(render_mode=None, camera_name='fixed'):
    """创建环境"""
    env = MediPickArmRightEnv(model_path="models/scene.xml", render_mode=render_mode)
    env = Monitor(env)
    return env


def evaluate_model(ppo_model, n_eval_episodes=10):
    """评估模型成功率"""
    # 创建独立的评估环境
    eval_env = MediPickArmRightEnv(model_path="models/scene.xml", render_mode=None)
    
    successes = 0
    total_rewards = []
    
    for _ in range(n_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 500  # 限制每回合最大步数，避免卡住
        
        while not done and step_count < max_steps:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        if info.get('success', False):
            successes += 1
    
    eval_env.close()
    
    success_rate = successes / n_eval_episodes
    mean_reward = np.mean(total_rewards)
    
    return success_rate, mean_reward


@click.command()
@click.option('--steps', default=-1, type=int, help='训练步数 (-1表示无限训练直到成功)')
@click.option('--success-threshold', default=0.95, type=float, help='成功率阈值 (0-1)')
@click.option('--consecutive', default=3, type=int, help='连续达到阈值次数')
@click.option('--record/--no-record', default=False, help='训练时录制视频')
def main(steps, success_threshold, consecutive, record):
    """训练脚本"""
    # 创建目录
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)
    
    # 使用时间戳创建唯一目录
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"{MODEL_DIR}/medipick_arm_right_{run_id}"
    log_dir = f"{LOG_DIR}/arm_right_run_{run_id}"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 确定训练步数
    if steps <= 0:
        steps = MAX_STEPS
        print("=" * 50)
        print(f"MediPick 右臂 PPO 训练 (无限模式)")
        print(f"训练ID: {run_id}")
        print(f"最大步数: {MAX_STEPS:,}")
        print(f"成功率阈值: {success_threshold*100}%")
        print(f"连续成功次数: {consecutive}")
        print(f"视频录制: {'启用' if record else '禁用'}")
    else:
        print("=" * 50)
        print(f"MediPick 右臂 PPO 训练")
        print(f"训练ID: {run_id}")
        print(f"训练步数: {steps:,}")
        print(f"视频录制: {'启用' if record else '禁用'}")
    
    print("=" * 50)
    
    # 训练环境 - 不使用rgb_array渲染加快速度
    env = make_env(render_mode=None)
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
    
    print(f"使用设备: cpu")
    print(f"日志目录: {log_dir}")
    print("=" * 50)
    print("【重要】查看训练数据的命令:")
    print(f"tensorboard --logdir {os.path.abspath(log_dir)}")
    print("=" * 50)
    
    # 评估回调 - render=False
    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=10000,  # 每10000步评估一次
        deterministic=True,
        render=False,  # Bug修复: 不渲染
        n_eval_episodes=10,
        verbose=1
    )
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="arm_right_step"
    )
    
    # 奖励记录回调
    tb_reward_callback = TensorboardRewardCallback(eval_callback, verbose=0)
    
    # 连续成功计数
    consecutive_count = 0
    best_success_rate = 0.0
    
    # 开始训练
    total_steps = 0
    iteration = 0
    
    # 视频录制相关
    video_recorder = None
    if record:
        try:
            from stable_baselines3.common.vec_env import VecVideoRecorder
            # 使用固定摄像头录制
            video_recorder = VecVideoRecorder(
                env, 
                RECORDING_DIR,
                record_video_trigger=lambda x: x % 20000 == 0,  # 每20000步录制一次
                video_length=500,
                name_prefix=f"medipick_right_train_{run_id}"
            )
            print(f"视频录制已启用，视频将保存到: {RECORDING_DIR}")
        except Exception as e:
            print(f"视频录制初始化失败: {e}")
            record = False
    
    try:
        # 直接开始训练，先不评估
        while total_steps < steps:
            iteration += 1
            print(f"\n--- 迭代 {iteration} (已训练: {total_steps:,} 步) ---")
            
            # 训练下一批
            batch_steps = min(10000, steps - total_steps)
            model.learn(
                total_timesteps=batch_steps,
                callback=CallbackList([
                    checkpoint_callback,
                    eval_callback,
                    tb_reward_callback
                ]),
                progress_bar=False
            )
            total_steps += batch_steps
            
            # 训练完成后评估
            print("评估模型...")
            success_rate, mean_reward = evaluate_model(model, n_eval_episodes=10)
            print(f"当前成功率: {success_rate*100:.1f}%")
            print(f"平均奖励: {mean_reward:.2f}")
            
            # 检查是否达到成功阈值
            if success_rate >= success_threshold:
                consecutive_count += 1
                print(f"  达到阈值! 连续次数: {consecutive_count}/{consecutive}")
                if consecutive_count >= consecutive:
                    print(f"\n🎉 连续{consecutive}次评估成功率超过{success_threshold*100}%!")
                    break
            else:
                consecutive_count = 0
            
            # 检查是否有改善
            if success_rate > best_success_rate:
                best_success_rate = success_rate
            
            # 检查是否已经收敛（连续多次没有改善）
            if success_rate < 0.1 and total_steps > 50000:
                print("\n⚠️ 训练似乎没有进展，成功率很低")
                print("继续训练...")
            
    except KeyboardInterrupt:
        print("\n训练被手动停止")
    finally:
        # 最终评估
        print("\n" + "=" * 50)
        print("最终评估:")
        final_success_rate, final_reward = evaluate_model(model, n_eval_episodes=20)
        print(f"最终成功率: {final_success_rate*100:.1f}%")
        print(f"最终平均奖励: {final_reward:.2f}")
        
        # 保存最终模型
        final_path = f"{model_dir}/final_model"
        model.save(final_path)
        print(f"模型已保存到: {final_path}")
        
        # 关闭视频录制器
        if video_recorder is not None:
            try:
                video_recorder.close()
            except:
                pass
        
        print("\n训练完成!")
        print("如需单独录制视频，请使用: python enjoy_arm_right.py --record")


if __name__ == "__main__":
    main()
