#!/usr/bin/env python3
"""
MediPick 交互式菜单
"""
import os
import sys
import subprocess

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    print("=" * 60)
    print("       MediPick 机械臂控制系统 - 交互式菜单")
    print("=" * 60)
    print()

def print_menu():
    print("请选择功能:")
    print()
    print("  [1] 单次训练 (PPO)")
    print("      - 训练右臂6关节 + 升降杆")
    print("      - 支持视频录制")
    print("      - 支持命令行参数")
    print()
    print("  [2] 课程训练 (5个渐进课程)")
    print("      - 课程1: 基础 - 只有右臂,无货架")
    print("      - 课程2: 初级 - 2层货架")
    print("      - 课程3: 中级 - 4层货架")
    print("      - 课程4: 高级 - 6层货架")
    print("      - 课程5: 完整 - 8层货架+升降杆")
    print()
    print("  [3] 查看训练日志/奖励表")
    print("      - 分析TensorBoard日志")
    print("      - 生成奖励记录表格")
    print()
    print("  [4] 模型演示")
    print("      - 观看训练好的模型效果")
    print("      - 支持视频录制")
    print()
    print("  [5] 查看训练视频")
    print("      - 打开视频保存目录")
    print()
    print("  [6] 查看帮助文档")
    print()
    print("  [0] 退出")
    print()

def run_single_train():
    """运行单次训练"""
    print("\n单次训练选项:")
    print("  [1] 使用默认参数训练")
    print("  [2] 训练 + 录制视频")
    print("  [3] 指定训练步数")
    print("  [4] 指定训练步数 + 录制视频")
    print("  [0] 返回")
    
    choice = input("\n请选择: ").strip()
    
    if choice == "1":
        os.system("python rl/train_arm_right.py")
    elif choice == "2":
        os.system("python rl/train_arm_right.py --record")
    elif choice == "3":
        steps = input("请输入训练步数 (例如 100000): ").strip()
        if steps.isdigit():
            os.system(f"python rl/train_arm_right.py --steps {steps}")
    elif choice == "4":
        steps = input("请输入训练步数 (例如 100000): ").strip()
        if steps.isdigit():
            os.system(f"python rl/train_arm_right.py --steps {steps} --record")

def run_curriculum_train():
    """运行课程训练"""
    print("\n课程训练选项:")
    print("  [1] 训练所有课程")
    print("  [2] 从指定课程开始")
    print("  [3] 单独训练某个课程")
    print("  [4] 查看课程列表")
    print("  [0] 返回")
    
    choice = input("\n请选择: ").strip()
    
    if choice == "1":
        os.system("python rl/curriculum.py")
    elif choice == "2":
        start = input("请输入起始课程 (1-5): ").strip()
        if start.isdigit() and 1 <= int(start) <= 5:
            os.system(f"python rl/curriculum.py --start {start}")
    elif choice == "3":
        course = input("请输入课程编号 (1-5): ").strip()
        if course.isdigit() and 1 <= int(course) <= 5:
            os.system(f"python rl/curriculum.py --course {course}")
    elif choice == "4":
        os.system("python rl/curriculum.py --list")

def show_logs():
    """查看训练日志"""
    print("\n日志查看选项:")
    print("  [1] 查看最新训练的奖励表")
    print("  [2] 列出所有训练记录")
    print("  [3] 启动TensorBoard")
    print("  [0] 返回")
    
    choice = input("\n请选择: ").strip()
    
    if choice == "1":
        os.system("python rl/analyze_logs.py")
    elif choice == "2":
        os.system("python rl/analyze_logs.py --list")
    elif choice == "3":
        print("\n启动TensorBoard... (按Ctrl+C退出)")
        print("然后在浏览器打开: http://localhost:6006")
        os.system("tensorboard --logdir assets/logs/")

def run_demo():
    """模型演示"""
    print("\n演示选项:")
    print("  [1] 使用最新模型演示")
    print("  [2] 演示 + 录制视频")
    print("  [0] 返回")
    
    choice = input("\n请选择: ").strip()
    
    if choice == "1":
        os.system("python rl/enjoy_arm_right.py")
    elif choice == "2":
        os.system("python rl/enjoy_arm_right.py --record")

def show_videos():
    """查看视频"""
    print("\n视频目录: assets/videos/")
    os.system("ls -la assets/videos/")

def show_help():
    """显示帮助"""
    print("""
================================================================================
                        MediPick 帮助文档
================================================================================

1. 单次训练
   python rl/train_arm_right.py --steps 100000 --record

2. 课程训练
   python rl/curriculum.py --list        # 查看课程列表
   python rl/curriculum.py --course 3    # 单独训练课程3
   python rl/curriculum.py --start 2     # 从课程2开始

3. 查看日志
   python rl/analyze_logs.py             # 最新训练奖励表
   python rl/analyze_logs.py --list      # 所有训练记录
   tensorboard --logdir assets/logs/     # TensorBoard

4. 模型演示
   python rl/enjoy_arm_right.py
   python rl/enjoy_arm_right.py --record

5. 规划器测试
   python planning/rrt.py

================================================================================
""")

def main():
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("请输入选项: ").strip()
        
        if choice == "0":
            print("\n再见!")
            break
        elif choice == "1":
            run_single_train()
        elif choice == "2":
            run_curriculum_train()
        elif choice == "3":
            show_logs()
        elif choice == "4":
            run_demo()
        elif choice == "5":
            show_videos()
        elif choice == "6":
            show_help()
        else:
            print("\n无效选项，请重试")
        
        if choice != "0":
            input("\n按回车继续...")

if __name__ == "__main__":
    main()
