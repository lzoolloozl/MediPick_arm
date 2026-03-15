"""
分析训练日志并生成奖励记录表格
"""
import os
import glob
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def find_latest_log():
    """找到最新的日志目录"""
    log_dirs = sorted(glob.glob("assets/logs/arm_right_run_*"), reverse=True)
    if not log_dirs:
        return None
    return log_dirs[0]

def read_tensorboard_log(log_dir):
    """读取TensorBoard日志"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    return ea

def get_reward_data(log_dir):
    """获取奖励数据"""
    ea = read_tensorboard_log(log_dir)
    
    data = {
        'timestep': [],
        'train_reward': [],
        'eval_reward': [],
        'success_rate': []
    }
    
    # 获取训练奖励
    if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
        train_data = ea.Scalars('rollout/ep_rew_mean')
        for e in train_data:
            data['timestep'].append(e.step)
            data['train_reward'].append(e.value)
    
    # 获取评估奖励
    if 'eval/mean_reward' in ea.Tags()['scalars']:
        eval_data = ea.Scalars('eval/mean_reward')
        for e in eval_data:
            data['eval_reward'].append(e.value)
    
    return data

def create_reward_table():
    """创建奖励记录表格"""
    log_dir = find_latest_log()
    
    if log_dir is None:
        print("未找到训练日志")
        return
    
    print(f"分析日志: {log_dir}")
    
    # 获取子日志目录
    sub_dirs = glob.glob(f"{log_dir}/PPO_*")
    if not sub_dirs:
        print("未找到PPO日志子目录")
        return
    
    all_data = {
        'timestep': [],
        'reward': [],
        'source': []
    }
    
    for sub_dir in sub_dirs:
        try:
            ea = event_accumulator.EventAccumulator(sub_dir)
            ea.Reload()
            
            # 训练奖励
            if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
                for e in ea.Scalars('rollout/ep_rew_mean'):
                    all_data['timestep'].append(e.step)
                    all_data['reward'].append(e.value)
                    all_data['source'].append('train')
            
            # 评估奖励
            if 'eval/mean_reward' in ea.Tags()['scalars']:
                for e in ea.Scalars('eval/mean_reward'):
                    all_data['timestep'].append(e.step)
                    all_data['reward'].append(e.value)
                    all_data['source'].append('eval')
                    
        except Exception as e:
            print(f"读取{sub_dir}失败: {e}")
            continue
    
    if not all_data['timestep']:
        print("未找到奖励数据")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    df = df.sort_values('timestep')
    
    # 打印表格
    print("\n" + "=" * 80)
    print("奖励记录表")
    print("=" * 80)
    print(f"{'步数':<12} {'训练奖励':<15} {'评估奖励':<15}")
    print("-" * 80)
    
    # 按时间步聚合显示
    timesteps = sorted(df['timestep'].unique())
    for ts in timesteps:
        ts_data = df[df['timestep'] == ts]
        train_r = ts_data[ts_data['source'] == 'train']['reward'].mean()
        eval_r = ts_data[ts_data['source'] == 'eval']['reward'].mean()
        
        train_str = f"{train_r:.2f}" if not np.isnan(train_r) else "-"
        eval_str = f"{eval_r:.2f}" if not np.isnan(eval_r) else "-"
        
        print(f"{ts:<12} {train_str:<15} {eval_str:<15}")
    
    # 保存到CSV
    csv_path = f"{log_dir}/reward_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n数据已保存到: {csv_path}")
    
    # 打印统计摘要
    print("\n统计摘要:")
    print(f"  总训练步数: {max(timesteps) if timesteps else 0}")
    print(f"  最高训练奖励: {df[df['source']=='train']['reward'].max():.2f}")
    print(f"  最高评估奖励: {df[df['source']=='eval']['reward'].max():.2f}")


def list_all_runs():
    """列出所有训练记录"""
    log_dirs = sorted(glob.glob("assets/logs/arm_right_run_*"), reverse=True)
    
    if not log_dirs:
        print("未找到训练日志")
        return
    
    print("\n" + "=" * 80)
    print("所有训练记录")
    print("=" * 80)
    print(f"{'训练ID':<30} {'步数':<12} {'最高奖励':<15}")
    print("-" * 80)
    
    for log_dir in log_dirs:
        run_id = os.path.basename(log_dir).replace("arm_right_run_", "")
        
        # 尝试获取步数和奖励
        max_step = 0
        max_reward = float('-inf')
        
        sub_dirs = glob.glob(f"{log_dir}/PPO_*")
        for sub_dir in sub_dirs:
            try:
                ea = event_accumulator.EventAccumulator(sub_dir)
                ea.Reload()
                
                if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
                    for e in ea.Scalars('rollout/ep_rew_mean'):
                        if e.step > max_step:
                            max_step = e.step
                        if e.value > max_reward:
                            max_reward = e.value
                            
                if 'eval/mean_reward' in ea.Tags()['scalars']:
                    for e in ea.Scalars('eval/mean_reward'):
                        if e.value > max_reward:
                            max_reward = e.value
                            
            except:
                continue
        
        reward_str = f"{max_reward:.2f}" if max_reward != float('-inf') else "-"
        print(f"{run_id:<30} {max_step:<12} {reward_str:<15}")
    
    print("-" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_all_runs()
        else:
            create_reward_table()
    else:
        create_reward_table()
