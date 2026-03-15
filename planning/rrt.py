"""
RRT 路径规划器
用于机械臂路径规划
"""
import numpy as np
import random


class RRT:
    """快速随机树路径规划器"""
    
    def __init__(self, joint_limits, step_size=0.1, max_iter=1000):
        """
        初始化RRT规划器
        
        参数:
            joint_limits: 关节限制 [(min, max), ...]
            step_size: 步长
            max_iter: 最大迭代次数
        """
        self.joint_limits = np.array(joint_limits)
        self.step_size = step_size
        self.max_iter = max_iter
        
    def plan(self, start, goal):
        """
        规划路径
        
        参数:
            start: 起始关节位置
            goal: 目标关节位置
            
        返回:
            path: 路径点列表
        """
        start = np.array(start)
        goal = np.array(goal)
        
        # 树节点: (position, parent_index)
        tree = [(start, None)]
        
        for _ in range(self.max_iter):
            # 随机采样
            if random.random() < 0.1:
                # 10%概率采样目标
                random_point = goal.copy()
            else:
                random_point = self._random_sample()
            
            # 找到最近的节点
            nearest_idx = self._nearest(tree, random_point)
            nearest = tree[nearest_idx][0]
            
            # 扩展树
            new_point = self._steer(nearest, random_point)
            
            # 检查是否有效
            if self._is_valid(new_point):
                tree.append((new_point, nearest_idx))
                
                # 检查是否到达目标
                if np.linalg.norm(new_point - goal) < self.step_size:
                    # 重建路径
                    return self._reconstruct_path(tree, len(tree) - 1)
        
        return None
    
    def _random_sample(self):
        """随机采样关节空间"""
        return np.array([
            random.uniform(lo, hi) 
            for lo, hi in self.joint_limits
        ])
    
    def _nearest(self, tree, point):
        """找到最近的节点"""
        min_dist = float('inf')
        min_idx = 0
        
        for i, (node_pos, _) in enumerate(tree):
            dist = np.linalg.norm(node_pos - point)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
                
        return min_idx
    
    def _steer(self, from_pos, to_pos):
        """从from_pos向to_pos扩展"""
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        
        if distance < self.step_size:
            return to_pos.copy()
        
        direction = direction / distance
        return from_pos + direction * self.step_size
    
    def _is_valid(self, position):
        """检查位置是否有效"""
        for i, (pos, (lo, hi)) in enumerate(zip(position, self.joint_limits)):
            if pos < lo or pos > hi:
                return False
        return True
    
    def _reconstruct_path(self, tree, goal_idx):
        """重建路径"""
        path = []
        current_idx = goal_idx
        
        while current_idx is not None:
            node_pos, parent_idx = tree[current_idx]
            path.append(node_pos.copy())
            current_idx = parent_idx
            
        return path[::-1]


def main():
    """测试RRT规划器"""
    # 7个关节的限制
    joint_limits = [
        (0, 0.8),      # raise
        (-0.6, 4),     # r1
        (-1.98, 1.96), # r2
        (-1.6, 2.4),   # r3
        (-2.1, 2.4),   # r4
        (-3.06, 2.01), # r5
        (-1.88, 1.92), # r6
    ]
    
    rrt = RRT(joint_limits, step_size=0.1, max_iter=1000)
    
    start = [0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    goal = [0.6, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5]
    
    print("开始规划...")
    path = rrt.plan(start, goal)
    
    if path:
        print(f"规划成功! 路径长度: {len(path)}")
        for i, point in enumerate(path[:5]):
            print(f"  步骤{i}: {point}")
    else:
        print("规划失败")


if __name__ == "__main__":
    main()
