"""
MediPick 右臂训练环境 - 只有右臂(r1-r6)和升降杆(raise)可以动

奖励函数参数从配置文件读取
"""

import os
import json
import random
import numpy as np
import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from .utils import get_entry_plane_data, get_layer_bounds, get_slam_style_dist


# 默认奖励参数
DEFAULT_REWARD = {
    "distance_weight": 10.0,
    "approach_dist_1": 0.10,
    "approach_reward_1": 5.0,
    "approach_dist_2": 0.05,
    "approach_reward_2": 15.0,
    "approach_dist_3": 0.02,
    "approach_reward_3": 50.0,
    "approach_dist_4": 0.01,
    "approach_reward_4": 100.0,
    "success_reward": 500.0,
    "obstacle_dist": 0.05,
    "obstacle_penalty": 10.0,
    "collision_penalty": 150.0
}


def load_reward_config():
    """从配置文件加载奖励参数"""
    # 先检查当前目录
    config_path = os.path.join(os.path.dirname(__file__), "train_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'reward' in config:
                    return config['reward']
        except:
            pass
    
    # 再检查上级目录
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'reward' in config:
                    return config['reward']
        except:
            pass
    
    return DEFAULT_REWARD


class MediPickArmRightEnv(MujocoEnv, utils.EzPickle):
    """
    MediPick右臂训练环境 - 修复版
    
    动作空间: 只使用r1-r6关节(旋转)和raise_joint(线性移动)
    只有这7个执行器会被控制,其他关节(h1-h2, l1-l5, sucker)保持静止.
    """
    
    def __init__(self, model_path="../models/scene.xml", **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        
        # 加载奖励配置
        self._reward_config = load_reward_config()
        
        # 观测空间维度: 7 + 7 + 3 + 3 + 6 + 2 + 1 + 3 = 32
        observation_space = Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

        # 先创建MujocoEnv，让它设置默认的action_space（会被覆盖，但需要先调用）
        MujocoEnv.__init__(
            self, os.path.abspath(model_path), 5,
            observation_space=observation_space, **kwargs
        )
        
        # Bug修复1: 在MujocoEnv.__init__之后重新设置action_space
        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # Bug修复2: 获取7个活动关节的执行器信息
        # 使用正确的执行器名称
        self._joint_names = ['raise_joint', 'r1_joint', 'r2_joint', 'r3_joint', 'r4_joint', 'r5_joint', 'r6_joint']
        self._actuator_names = ['act_raise', 'act_r1', 'act_r2', 'act_r3', 'act_r4', 'act_r5', 'act_r6']
        
        self._joint_ids = np.array([self.model.joint(name).id for name in self._joint_names])
        
        # 获取执行器ID和真实控制范围
        self._actuator_ids = []
        self._ctrl_ranges = []
        
        for name in self._actuator_names:
            act = self.model.actuator(name)
            act_id = act.id
            # 获取ctrlrange
            ctrlrange = (act.ctrlrange[0], act.ctrlrange[1])
            self._actuator_ids.append(act_id)
            self._ctrl_ranges.append(ctrlrange)
        
        self._actuator_ids = np.array(self._actuator_ids)
        self._ctrl_ranges = np.array(self._ctrl_ranges)
        
        # 获取需要固定的关节id (h1-h2, l1-l5)
        self._fixed_joint_names = ['h1_joint', 'h2_joint', 'l1_joint', 'l2_joint', 'l3_joint', 'l4_joint', 'l5_joint']
        self._fixed_joint_ids = np.array([self.model.joint(name).id for name in self._fixed_joint_names])
        
        # sucker关节ID（用于锁定）
        self._sucker_joint_id = self.model.joint("sucker_joint").id
        
        # 目标点body ID（用于获取位置）
        self._box_body_id = self.model.body("target_point").id
        
        # 最大步数限制
        self._max_steps = 500
        self._step_count = 0
        
    def _apply_action(self, action):
        """
        Bug修复3: 手动设置ctrl数组
        - 先将所有执行器设为当前位置（保持不变）
        - 只更新7个活动执行器的控制值
        - 将[-1,1]反归一化到真实控制范围
        """
        # 先将所有ctrl设为当前位置（保持不变）
        for i in range(self.model.nu):
            # 获取执行器对应的关节ID
            trn_id = self.model.actuator_trnid[i][0]  # joint ID
            self.data.ctrl[i] = self.data.qpos[trn_id]
        
        # 只更新7个活动执行器
        for i, (act_id, ctrl_range) in enumerate(zip(self._actuator_ids, self._ctrl_ranges)):
            lo, hi = ctrl_range
            # 将[-1,1]反归一化到真实范围
            self.data.ctrl[act_id] = lo + (action[i] + 1.0) / 2.0 * (hi - lo)
    
    def _get_active_joint_data(self):
        """提取活动关节的位置和速度数据"""
        active_qpos = np.array([self.data.qpos[jid] for jid in self._joint_ids], dtype=np.float32)
        active_qvel = np.array([self.data.qvel[jid] for jid in self._joint_ids], dtype=np.float32)
        return active_qpos, active_qvel
        
    def _check_contact_with_box_front(self):
        """检查吸盘末端是否与box_front_center接触"""
        sucker_pos = self.data.site("sucker_tip").xpos
        box_front_pos = self._get_box_front_center_world()
        dist = np.linalg.norm(sucker_pos - box_front_pos)
        return dist

    def _get_box_front_center_world(self):
        """获取box_front_center site的世界坐标"""
        box_front_site_local = np.array([0, -0.01, 0])
        box_rot = self.data.body("target_point").xmat.reshape(3, 3)
        box_front_world = self.data.body("target_point").xpos + box_rot @ box_front_site_local
        return box_front_world

    def _get_obs(self):
        """构建观测向量"""
        # 1. 机器人活动关节的位置和速度
        active_qpos, active_qvel = self._get_active_joint_data()
        
        # 2. 吸盘末端位置
        sucker_pos = self.data.site("sucker_tip").xpos.astype(np.float32)
        
        # 3. 药盒front_center世界坐标
        box_front_pos = self._get_box_front_center_world().astype(np.float32)
        
        # 4. 侵入面信息
        entry_point, entry_normal = get_entry_plane_data(self.data, self.model)
        entry_point = entry_point.astype(np.float32)
        entry_normal = entry_normal.astype(np.float32)
        
        # 5. 上下层边界
        z_upper, z_lower = get_layer_bounds(self.data)
        layer_bounds = np.array([z_upper, z_lower], dtype=np.float32)
        
        # 6. 距离传感器
        d_suction = get_slam_style_dist(
            self.model, self.data, 
            "sucker_tip", 
            ["sucker", "r6_link", "r5_link", "r4_link", "r3_link", "r2_link", "r1_link"],
            max_dist=0.5
        )
        dist_sensors = np.array([d_suction], dtype=np.float32)
        
        # 7. 相对位置
        rel_pos = box_front_pos - sucker_pos
        
        # 拼接所有观测
        obs = np.concatenate([
            active_qpos,        # 7
            active_qvel,        # 7
            sucker_pos,         # 3
            box_front_pos,      # 3
            entry_point,        # 3
            entry_normal,       # 3
            layer_bounds,       # 2
            dist_sensors,       # 1
            rel_pos             # 3
        ]).astype(np.float32)
        
        return obs

    def _check_collisions(self):
        """检测非法碰撞"""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1 = self.model.geom(c.geom1).name if c.geom1 >= 0 else None
            n2 = self.model.geom(c.geom2).name if c.geom2 >= 0 else None
            
            if n1 is None or n2 is None:
                continue
                
            if "floor" in n1.lower() or "floor" in n2.lower():
                continue
                
            if "Shelf" in n1 or "Shelf" in n2:
                continue
                
            if "box_geom" in n1 or "box_geom" in n2:
                continue
                
            robot_parts = ['base', 'raise', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 
                          'h1', 'h2', 'l1', 'l2', 'l3', 'l4', 'l5', 'sucker']
            
            is_robot_1 = any(part in n1.lower() for part in robot_parts)
            is_robot_2 = any(part in n2.lower() for part in robot_parts)
            
            if is_robot_1 or is_robot_2:
                is_box_1 = "box" in n1.lower()
                is_box_2 = "box" in n2.lower()
                
                if not (is_box_1 or is_box_2):
                    return True
                    
        return False

    def _get_rew(self, obs):
        """
        奖励函数 - 使用配置文件参数
        """
        cfg = self._reward_config
        rel_pos = obs[29:32]
        distance = np.linalg.norm(rel_pos)
        
        # 1. 距离惩罚
        reward = -distance * cfg['distance_weight']
        
        # 2. 接近分级奖励
        contact_dist = self._check_contact_with_box_front()
        if contact_dist < cfg['approach_dist_1']:
            reward += cfg['approach_reward_1']
        if contact_dist < cfg['approach_dist_2']:
            reward += cfg['approach_reward_2']
        if contact_dist < cfg['approach_dist_3']:
            reward += cfg['approach_reward_3']
        if contact_dist < cfg['approach_dist_4']:
            reward += cfg['approach_reward_4']
            
        # 3. 成功奖励
        if contact_dist < cfg['approach_dist_4']:
            reward += cfg['success_reward']
            
        # 4. 障碍物距离惩罚
        d_suction = obs[28]
        if d_suction < cfg['obstacle_dist']:
            reward -= cfg['obstacle_penalty'] * (cfg['obstacle_dist'] - d_suction)
                
        return reward

    def step(self, action):
        """执行一步仿真"""
        # Bug修复5: 使用_apply_action并循环frame_skip次
        self._apply_action(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # 增加步数计数
        self._step_count += 1
        
        # 获取观测
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._get_rew(obs)
        
        # 检测碰撞
        collided = self._check_collisions()
        if collided:
            reward -= self._reward_config['collision_penalty']
            
        # 检查终止条件
        contact_dist = self._check_contact_with_box_front()
        
        # Bug修复6: 添加truncated处理
        terminated = collided or (contact_dist < 0.01)
        truncated = self._step_count >= self._max_steps
        
        info = {
            "collided": collided,
            "contact_dist": contact_dist,
            "success": contact_dist < 0.01
        }
        
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        """重置环境"""
        init_qpos = self.data.qpos.copy()
        init_qvel = self.data.qvel.copy()
        
        # 将活动关节设为0
        for jid in self._joint_ids:
            if jid < len(init_qpos):
                init_qpos[jid] = 0.0
            if jid < len(init_qvel):
                init_qvel[jid] = 0.0
        
        # 将固定关节也设为0
        for jid in self._fixed_joint_ids:
            if jid < len(init_qpos):
                init_qpos[jid] = 0.0
            if jid < len(init_qvel):
                init_qvel[jid] = 0.0
        
        # 锁定sucker_joint
        if self._sucker_joint_id < len(init_qpos):
            init_qpos[self._sucker_joint_id] = 0.0
        if self._sucker_joint_id < len(init_qvel):
            init_qvel[self._sucker_joint_id] = 0.0
        
        # 重置步数计数
        self._step_count = 0
        
        self.set_state(init_qpos, init_qvel)
        return self._get_obs()
