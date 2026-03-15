# MediPick 机械臂控制系统

MediPick是一个用于药品分拣的机械臂控制系统，支持两种控制方式：

1. **强化学习 (RL)** - 使用PPO算法
2. **规划器** - 使用RRT等路径规划算法

## 项目结构

```
MediPick_arm/
├── assets/                  # 资源文件
│   ├── checkpoints/        # 训练保存的模型
│   ├── logs/              # TensorBoard日志
│   ├── meshes/            # 3D模型文件
│   ├── models/            # MuJoCo模型文件
│   │   ├── scene.xml     # 场景文件
│   │   ├── shelves/      # 货架模型
│   │   └── simple3/      # 机器人模型
│   └── videos/            # 训练/演示视频
│
├── planning/               # 规划模块
│   ├── __init__.py
│   └── rrt.py            # RRT路径规划器
│
├── rl/                    # 强化学习模块
│   ├── train_arm_right.py # PPO训练脚本
│   ├── enjoy_arm_right.py # 模型演示脚本
│   └── envs/             # RL环境
│       ├── medipick_arm_right.py
│       └── utils.py
│
└── README.md
```

## 快速开始

### 强化学习训练

```bash
# 激活虚拟环境
source .venv/bin/activate

# 训练模型 (默认无限训练直到成功率达标)
python rl/train_arm_right.py

# 指定训练步数
python rl/train_arm_right.py --steps 100000

# 训练时录制视频
python rl/train_arm_right.py --record
```

### 查看训练进度

```bash
# 启动TensorBoard
tensorboard --logdir assets/logs/

# 然后在浏览器打开 http://localhost:6006
```

### 模型演示

```bash
# 演示训练好的模型
python rl/enjoy_arm_right.py

# 演示并录制视频
python rl/enjoy_arm_right.py --record
```

### 规划器

```bash
# 测试RRT规划器
python planning/rrt.py
```

## 机械臂关节说明

| 关节名 | 类型 | 范围 | 描述 |
|--------|------|------|------|
| raise_joint | 线性 | 0 - 0.8m | 升降关节 |
| r1_joint | 旋转 | -0.6 - 4 rad | 右臂关节1 |
| r2_joint | 旋转 | -1.98 - 1.96 rad | 右臂关节2 |
| r3_joint | 旋转 | -1.6 - 2.4 rad | 右臂关节3 |
| r4_joint | 旋转 | -2.1 - 2.4 rad | 右臂关节4 |
| r5_joint | 旋转 | -3.06 - 2.01 rad | 右臂关节5 |
| r6_joint | 旋转 | -1.88 - 1.92 rad | 右臂关节6 |

## 环境要求

- Python 3.10+
- MuJoCo
- Stable-Baselines3
- Gymnasium
- PyTorch

## 训练参数

- **SUCCESS_THRESHOLD**: 95% 成功率阈值
- **CONSECUTIVE_SUCCESS**: 连续3次达到阈值停止训练
- **MAX_STEPS**: 最大1000万步

## 注意事项

1. 训练时会自动评估模型并保存最佳模型
2. 每10000步保存一次检查点
3. 视频录制会占用较多磁盘空间，建议仅在需要时使用
