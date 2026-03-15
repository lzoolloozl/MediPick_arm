"""
MediPick 课程训练系统 - 支持配置文件
5个渐进课程:
1. 只有右臂6关节,无货架,目标点位姿
2. 右臂6关节,2层货架,目标在架子里,碰撞扣分
3. 右臂6关节,4层货架
4. 右臂6关节,6层货架
5. 右臂6关节+升降杆,8层货架(更高)

配置文件: train_config.json
"""
import os
import sys
import json
import glob
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import click

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 目录配置
MODEL_DIR = "assets/checkpoints"
LOG_DIR = "assets/logs"
VIDEO_DIR = "assets/videos"

# 默认课程配置 (如果配置文件不存在时使用)
DEFAULT_COURSES = {
    1: {
        "name": "基础 - 只有右臂,无货架",
        "scene": "assets/models/scene_no_shelf.xml",
        "steps": 500000,
        "reward_threshold": -200,
        "success_consecutive": 3
    },
    2: {
        "name": "初级 - 2层货架",
        "scene": "assets/models/scene_2_layers.xml",
        "steps": 500000,
        "reward_threshold": -150,
        "success_consecutive": 3
    },
    3: {
        "name": "中级 - 4层货架",
        "scene": "assets/models/scene.xml",
        "steps": 500000,
        "reward_threshold": -100,
        "success_consecutive": 3
    },
    4: {
        "name": "高级 - 6层货架",
        "scene": "assets/models/scene_6_layers.xml",
        "steps": 500000,
        "reward_threshold": -50,
        "success_consecutive": 3
    },
    5: {
        "name": "完整 - 升降杆+右臂",
        "scene": "assets/models/scene.xml",
        "steps": 1000000,
        "reward_threshold": 0,
        "success_consecutive": 3
    },
}


def load_config(config_file=None):
    """加载配置文件"""
    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__), "train_config.json")
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return None


def get_courses_from_config(config):
    """从配置获取课程"""
    if config and 'courses' in config:
        return config['courses']
    return DEFAULT_COURSES


def get_training_params(config):
    """从配置获取训练参数"""
    if config and 'training' in config:
        return config['training']
    return {
        "n_steps": 512,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "n_epochs": 10,
        "gamma": 0.99,
        "eval_freq": 10000,
        "checkpoint_freq": 10000
    }


class CurriculumTrainer:
    """课程训练器"""
    
    def __init__(self, config_file=None):
        self.config = load_config(config_file)
        self.courses = get_courses_from_config(self.config)
        self.training_params = get_training_params(self.config)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建课程训练总目录
        self.course_dir = f"{MODEL_DIR}/curriculum_{self.run_id}"
        os.makedirs(self.course_dir, exist_ok=True)
        
        self.current_model_path = None
        
    def get_prev_course_model(self, course_num):
        """获取前一课程的模型"""
        if course_num <= 1:
            return None
        
        prev_course_dir = f"{self.course_dir}/course_{course_num - 1}"
        if not os.path.exists(prev_course_dir):
            return None
        
        # 找最新模型
        best_model = os.path.join(prev_course_dir, "best_model.zip")
        if os.path.exists(best_model):
            return best_model
        
        final_model = os.path.join(prev_course_dir, "final_model.zip")
        if os.path.exists(final_model):
            return final_model
        
        return None
    
    def evaluate_model(self, model, env, n_eval_episodes=10):
        """评估模型"""
        rewards = []
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            max_steps = 500
            
            while not done and step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
            
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        return mean_reward
    
    def train_course(self, course_num, config, record_video=False):
        """训练单个课程"""
        course_dir = f"{self.course_dir}/course_{course_num}"
        os.makedirs(course_dir, exist_ok=True)
        
        log_dir = f"{LOG_DIR}/course{course_num}_{self.run_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        video_dir = f"{VIDEO_DIR}/course{course_num}_{self.run_id}" if record_video else None
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print(f"课程 {course_num}: {config['name']}")
        print(f"场景: {config['scene']}")
        print(f"训练步数: {config['steps']}")
        print(f"奖励阈值: {config.get('reward_threshold', -100)}")
        print("=" * 60)
        
        # 导入环境
        from rl.envs.medipick_arm_right import MediPickArmRightEnv
        
        # 创建环境
        scene_path = config.get('scene', 'assets/models/scene.xml')
        env = MediPickArmRightEnv(
            model_path=scene_path, 
            render_mode=None
        )
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # 加载前一课程模型或创建新模型
        prev_model = self.get_prev_course_model(course_num)
        if prev_model and os.path.exists(prev_model):
            print(f"从上一课程加载模型: {prev_model}")
            model = PPO.load(prev_model, env=env)
        else:
            print("创建新模型...")
            model = PPO(
                "MlpPolicy",
                env=env,
                verbose=1,
                device='cpu',
                tensorboard_log=log_dir,
                n_steps=self.training_params['n_steps'],
                batch_size=self.training_params['batch_size'],
                learning_rate=self.training_params['learning_rate'],
                n_epochs=self.training_params['n_epochs'],
                gamma=self.training_params['gamma'],
            )
        
        # 回调
        eval_callback = EvalCallback(
            env,
            best_model_save_path=course_dir,
            log_path=course_dir,
            eval_freq=self.training_params['eval_freq'],
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=self.training_params['checkpoint_freq'],
            save_path=course_dir,
            name_prefix=f"course{course_num}_step"
        )
        
        # 获取课程特定参数
        reward_threshold = config.get('reward_threshold', -100)
        success_consecutive = config.get('success_consecutive', 3)
        
        # 训练循环 - 检查是否达到奖励阈值
        total_steps = 0
        consecutive_good = 0
        batch_size = 10000
        max_steps = config['steps']
        
        while total_steps < max_steps:
            # 训练一批
            batch_steps = min(batch_size, max_steps - total_steps)
            model.learn(
                total_timesteps=batch_steps,
                callback=CallbackList([checkpoint_callback, eval_callback]),
                progress_bar=False
            )
            total_steps += batch_steps
            
            # 评估
            mean_reward = self.evaluate_model(model, env, n_eval_episodes=5)
            print(f"  步数: {total_steps}, 平均奖励: {mean_reward:.2f}")
            
            # 检查是否达到阈值
            if mean_reward >= reward_threshold:
                consecutive_good += 1
                print(f"  ✓ 达到阈值! 连续: {consecutive_good}/{success_consecutive}")
                if consecutive_good >= success_consecutive:
                    print(f"\n  🎉 课程 {course_num} 训练完成!")
                    break
            else:
                consecutive_good = 0
        
        # 保存最终模型
        final_path = f"{course_dir}/final_model"
        model.save(final_path)
        
        self.current_model_path = final_path + ".zip"
        
        print(f"\n课程 {course_num} 完成!")
        print(f"模型保存到: {course_dir}")
        
        env.close()
        
        return course_dir


def print_courses(courses):
    """列出所有课程"""
    print("\n" + "=" * 60)
    print("课程列表")
    print("=" * 60)
    for num, config in courses.items():
        print(f"\n课程 {num}: {config['name']}")
        print(f"  场景: {config['scene']}")
        print(f"  训练步数: {config['steps']:,}")
        print(f"  奖励阈值: {config.get('reward_threshold', '-')}")
    print("\n" + "=" * 60)


@click.command()
@click.option('--course', default=None, type=int, help='训练指定课程 (1-5)')
@click.option('--config', default=None, type=str, help='配置文件路径')
@click.option('--record/--no-record', default=False, help='录制视频')
def main(course, config, record):
    """课程训练系统"""
    # 加载配置
    config_data = load_config(config)
    courses = get_courses_from_config(config_data)
    
    if course:
        # 单独训练指定课程
        if course not in courses:
            print(f"错误: 无效课程编号 {course}")
            return
        config = courses[course]
        trainer = CurriculumTrainer(config)
        trainer.train_course(course, config, record_video=record)
    else:
        # 训练所有课程
        trainer = CurriculumTrainer(config)
        
        for course_num in range(1, 6):
            if course_num not in courses:
                continue
            config = courses[course_num]
            trainer.train_course(course_num, config, record_video=record)
        
        print("\n" + "=" * 60)
        print("所有课程训练完成!")
        print(f"总目录: {trainer.course_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
