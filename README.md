# MediPick 机械臂控制系统

MediPick是一个用于药品分拣的机械臂控制系统，使用强化学习(PPO)算法训练机械臂完成抓取任务。

## 项目结构

```
MediPick_arm/
├── assets/                  # 资源文件
│   ├── checkpoints/        # 训练保存的模型
│   ├── logs/              # TensorBoard日志
│   ├── models/            # MuJoCo场景文件
│   └── videos/            # 训练视频
├── planning/               # 规划模块
│   └── rrt.py            # RRT路径规划器
├── rl/                    # 强化学习模块
│   ├── train_config.json  # 训练配置文件
│   ├── train_arm_right.py # 单次训练脚本
│   ├── curriculum.py      # 课程训练系统
│   ├── analyze_logs.py   # 日志分析工具
│   ├── enjoy_arm_right.py # 模型演示
│   ├── menu.py           # 交互式菜单
│   └── envs/             # RL环境
│       ├── medipick_arm_right.py  # 训练环境
│       └── utils.py       # 工具函数
└── README.md
```

## 快速开始

### 1. 交互式菜单 (推荐)
```bash
python rl/menu.py
```

### 2. 命令行训练
```bash
# 单次训练
python rl/train_arm_right.py --steps 50000 --record

# 课程训练
python rl/curriculum.py --course 1
python rl/curriculum.py --course 2
```

### 3. 查看训练结果
```bash
# 查看奖励表
python rl/analyze_logs.py

# 启动TensorBoard
tensorboard --logdir assets/logs/

# 模型演示
python rl/enjoy_arm_right.py
```

---

## 训练配置文件说明

配置文件: `rl/train_config.json`

```json
{
    "curriculum": {
        "start_course": 1,        // 起始课程
        "end_course": 5,          // 结束课程
        "auto_advance": true,     // 自动进入下一课程
        "reward_threshold": -100,  // 奖励阈值
        "consecutive_success": 3   // 连续成功次数
    },
    "courses": {
        "1": {
            "name": "基础 - 只有右臂,无货架",
            "scene": "assets/models/scene_no_shelf.xml",
            "steps": 500000,       // 训练步数
            "reward_threshold": -200,  // 奖励阈值
            "success_consecutive": 3    // 连续成功次数
        },
        ...
    },
    "training": {
        "n_steps": 512,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "n_epochs": 10,
        "gamma": 0.99,
        "eval_freq": 10000,
        "checkpoint_freq": 10000
    }
}
```

### 可调参数说明:

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `steps` | 每个课程训练步数 | 500000 |
| `reward_threshold` | 自动进入下一课程的奖励阈值 | -200 ~ 0 |
| `success_consecutive` | 连续达到阈值次数 | 3 |
| `learning_rate` | 学习率 | 0.0003 |
| `batch_size` | 批大小 | 64 |

---

## 奖励函数说明

奖励函数定义在: `rl/envs/medipick_arm_right.py` 的 `_get_rew` 方法

### 奖励组成:

| 奖励项 | 数值 | 说明 |
|--------|------|------|
| 距离惩罚 | `-distance * 10` | 越近越好 |
| 接近奖励 (<0.10m) | +5 | 接近目标 |
| 接近奖励 (<0.05m) | +15 | 很接近 |
| 接触奖励 (<0.02m) | +50 | 接触目标 |
| 成功接触 (<0.01m) | +100 | 成功接触 |
| **成功奖励** | **+500** | **额外大额奖励!** |
| 障碍物惩罚 (<0.05m) | -10 | 太靠近障碍物 |
| 碰撞惩罚 | -150 | 发生碰撞 |

### 调整奖励:

修改 `rl/envs/medipick_arm_right.py` 中的 `_get_rew` 方法:

```python
def _get_rew(self, obs):
    rel_pos = obs[29:32]
    distance = np.linalg.norm(rel_pos)
    
    # 1. 距离惩罚
    reward = -distance * 10.0
    
    # 2. 接近分级奖励
    contact_dist = self._check_contact_with_box_front()
    if contact_dist < 0.10:
        reward += 5.0
    if contact_dist < 0.05:
        reward += 15.0
    if contact_dist < 0.02:
        reward += 50.0
    if contact_dist < 0.01:
        reward += 100.0
        
    # 3. 成功奖励 (额外)
    if contact_dist < 0.01:
        reward += 500.0  # 修改这个值!
        
    # 4. 障碍物惩罚
    d_suction = obs[28]
    if d_suction < 0.05:
        reward -= 10.0 * (0.05 - d_suction)
            
    return reward
```

---

## 课程系统

5个渐进课程:

| 课程 | 名称 | 场景 | 描述 |
|------|------|------|------|
| 1 | 基础 | scene_no_shelf.xml | 只有右臂,无货架 |
| 2 | 初级 | scene_2_layers.xml | 2层货架 |
| 3 | 中级 | scene.xml | 4层货架 |
| 4 | 高级 | scene_6_layers.xml | 6层货架 |
| 5 | 完整 | scene.xml | 8层货架+升降杆 |

### 使用课程训练:

```bash
# 训练指定课程
python rl/curriculum.py --course 1
python rl/curriculum.py --course 2

# 训练所有课程
python rl/curriculum.py

# 指定配置文件
python rl/curriculum.py --config my_config.json
```

---

## 场景文件

场景文件位于 `assets/models/`:

| 文件 | 说明 |
|------|------|
| scene.xml | 默认4层货架场景 |
| scene_no_shelf.xml | 无货架场景 |
| scene_2_layers.xml | 2层货架场景 |
| scene_6_layers.xml | 6层货架场景 |

### 场景组成:
- **机器人**: simple3机械臂 (r1-r6关节 + raise升降杆)
- **货架**: 可配置层数
- **目标点**: 可视化盒子,无物理属性

---

## 常见问题

### Q: 如何调整训练难度?
A: 修改 `rl/train_config.json` 中的 `reward_threshold`，值越大越容易升级

### Q: 如何增加成功奖励?
A: 修改 `rl/envs/medipick_arm_right.py` 中的 `reward += 500.0`

### Q: 如何只训练单个课程?
A: 使用 `--course` 参数，如 `python rl/curriculum.py --course 1`

### Q: 如何查看训练进度?
A: 使用TensorBoard: `tensorboard --logdir assets/logs/`

---

## 环境要求

- Python 3.10+
- MuJoCo
- Stable-Baselines3
- Gymnasium
- PyTorch
