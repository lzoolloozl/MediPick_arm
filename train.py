import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.medipick_env import MediPickEnv

def train():
    # --- 1. 路径与环境配置 ---
    model_dir = "models/medipick_v1"
    log_dir = "logs/tensorboard"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 包装环境：Monitor 用于记录 Episode 统计信息，DummyVecEnv 是 SB3 的标准输入格式
    raw_env = MediPickEnv(model_path="assets/robot.xml")
    env = Monitor(raw_env)
    env = DummyVecEnv([lambda: env])

    # --- 2. 超参数配置 (Hyperparameters) ---
    # 这里定义了神经网络的架构和训练的核心参数
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], qf=[256, 256]) # 针对机器人控制设计的双层 256 宽网络
    )

    hyperparams = {
        "n_steps": 2048,           # 每次更新前收集的样本数
        "batch_size": 64,          # 优化时的批大小
        "gamma": 0.99,             # 折扣因子
        "learning_rate": 3e-4,     # 学习率
        "ent_coef": 0.01,          # 熵系数（鼓励探索，防止避障时陷入局部最优）
        "clip_range": 0.2,         # PPO 裁剪范围
        "n_epochs": 10,            # 每次更新时的迭代次数
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # --- 3. 初始化模型 ---
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        **hyperparams
    )

    # --- 4. 设置回调函数 (Callbacks) ---
    # 每 50,000 步保存一次模型
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix="medipick_step"
    )

    # (可选) 每 10,000 步进行一次独立评估并记录最佳模型
    eval_callback = EvalCallback(
        env, 
        best_model_save_path=f"{model_dir}/best_model",
        log_path=f"{model_dir}/eval_results", 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )

    # --- 5. 开始训练 ---
    print(f"训练开始！使用设备: {hyperparams['device']}")
    print(f"提示：运行 'tensorboard --logdir {log_dir}' 查看实时数据")
    
    try:
        model.learn(
            total_timesteps=1000000,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("训练被手动停止，正在保存当前模型...")
    finally:
        model.save(f"{model_dir}/medipick_final")
        print("模型已保存。")

if __name__ == "__main__":
    train()