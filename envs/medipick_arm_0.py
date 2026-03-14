import os
import numpy as np
import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from .utils import get_entry_plane_data, get_layer_bounds, get_slam_style_dist


class MediPickArm0Env(MujocoEnv, utils.EzPickle):
    """
    MediPick机械臂训练环境.
    
    动作空间: 只使用r1-r6关节(旋转)和raise_joint(线性移动)
    只有这7个执行器会被控制,其他关节(h1-h2, l1-l5)保持静止.
    
    观测空间包含:
    - 机器人qpos和qvel(仅活动关节: raise + r1-r6)
    - 吸盘末端位置(末端执行器)
    - 药盒front_center位置
    - 侵入面信息(点和法向量)
    - 上下层边界(货架高度信息)
    - 避障距离传感器
    
    目标: 让末端执行器与box_front_center垂直接触
    """
    
    def __init__(self, model_path="../models/scene.xml", **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        
        # 观测空间维度:
        # 活动关节qpos: 7 (raise + r1-r6)
        # 活动关节qvel: 7 (raise + r1-r6)
        # 吸盘末端位置: 3
        # 药盒front_center位置: 3
        # 侵入面: 点(3) + 法向量(3) = 6
        # 上下层边界: 上层z + 下层z = 2
        # 距离传感器: 1 (吸盘末端到障碍物)
        # 相对位置(box_front - sucker): 3
        # 总计: 7 + 7 + 3 + 3 + 6 + 2 + 1 + 3 = 32
        observation_space = Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

        # 动作空间: 7 (raise + r1-r6)
        # 先定义动作空间,避免MujocoEnv._set_action_space设置错误的值
        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        MujocoEnv.__init__(
            self, os.path.abspath(model_path), 5,
            observation_space=observation_space, **kwargs
        )
        
        # 获取r关节和raise_joint的关节id(只有这些会被控制)
        self._joint_names = ['raise_joint', 'r1_joint', 'r2_joint', 'r3_joint', 'r4_joint', 'r5_joint', 'r6_joint']
        self._joint_ids = np.array([self.model.joint(name).id for name in self._joint_names])
        
    def _get_active_joint_data(self):
        """提取活动关节的位置和速度数据"""
        active_qpos = np.array([self.data.qpos[jid] for jid in self._joint_ids], dtype=np.float32)
        active_qvel = np.array([self.data.qvel[jid] for jid in self._joint_ids], dtype=np.float32)
        return active_qpos, active_qvel
        
    def _check_contact_with_box_front(self):
        """
        检查吸盘末端是否与box_front_center接触.
        目标是垂直接触(吸盘末端触碰药盒前部中心).
        """
        # 获取吸盘末端的世界坐标
        sucker_pos = self.data.site("sucker_tip").xpos
        
        # 获取box_front_center的世界坐标
        # 根据scene.xml中的定义,box_front_center相对于药盒中心的位置是[0, -0.01, 0]
        box_front_site_local = np.array([0, -0.01, 0])
        # 获取药盒的旋转矩阵
        box_rot = self.data.body("pill_box").xmat.reshape(3, 3)
        # 计算box_front_center的世界坐标 = 药盒位置 + 旋转后的局部坐标
        box_front_world = self.data.body("pill_box").xpos + box_rot @ box_front_site_local
        
        # 计算吸盘末端到box_front_center的距离
        dist = np.linalg.norm(sucker_pos - box_front_world)
        
        return dist

    def _get_box_front_center_world(self):
        """获取box_front_center site的世界坐标"""
        box_front_site_local = np.array([0, -0.01, 0])
        box_rot = self.data.body("pill_box").xmat.reshape(3, 3)
        box_front_world = self.data.body("pill_box").xpos + box_rot @ box_front_site_local
        return box_front_world

    def _get_obs(self):
        """
        构建观测向量
        
        观测向量组成:
        - [0:7]   活动关节位置 (raise + r1-r6)
        - [7:14]  活动关节速度 (raise + r1-r6)  
        - [14:17] 吸盘末端位置 (sucker_tip)
        - [17:20] 药盒front_center世界坐标
        - [20:23] 侵入面位置点P
        - [23:26] 侵入面法向量n
        - [26:28] 上下层边界 (z_upper, z_lower)
        - [28:29] 距离传感器 (吸盘到障碍物的距离)
        - [29:32] 相对位置 (box_front_center - sucker_tip)
        """
        # 1. 机器人活动关节的位置和速度(只有raise + r1-r6)
        active_qpos, active_qvel = self._get_active_joint_data()
        
        # 2. 吸盘末端位置(末端执行器)
        sucker_pos = self.data.site("sucker_tip").xpos.astype(np.float32)
        
        # 3. 药盒front_center世界坐标
        box_front_pos = self._get_box_front_center_world().astype(np.float32)
        
        # 4. 侵入面信息(点和法向量)
        # 从utils.py中的get_entry_plane_data函数获取
        entry_point, entry_normal = get_entry_plane_data(self.data, self.model)
        entry_point = entry_point.astype(np.float32)
        entry_normal = entry_normal.astype(np.float32)
        
        # 5. 上下层边界(货架高度限制)
        # 从utils.py中的get_layer_bounds函数获取
        z_upper, z_lower = get_layer_bounds(self.data)
        layer_bounds = np.array([z_upper, z_lower], dtype=np.float32)
        
        # 6. 吸盘末端的距离传感器(用于避障)
        # 计算吸盘末端到最近障碍物的距离,排除机器人自身
        d_suction = get_slam_style_dist(
            self.model, self.data, 
            "sucker_tip", 
            ["sucker", "r6_link", "r5_link", "r4_link", "r3_link", "r2_link", "r1_link"],
            max_dist=0.5
        )
        dist_sensors = np.array([d_suction], dtype=np.float32)
        
        # 7. 相对位置: box_front_center - sucker_tip
        # 这是主要的目标信号,用于奖励函数
        rel_pos = box_front_pos - sucker_pos
        
        # 拼接所有观测
        obs = np.concatenate([
            active_qpos,        # 7个关节位置
            active_qvel,        # 7个关节速度
            sucker_pos,         # 3 吸盘位置
            box_front_pos,      # 3 药盒前端中心位置
            entry_point,        # 3 侵入面点
            entry_normal,       # 3 侵入面法向量
            layer_bounds,       # 2 上下层边界
            dist_sensors,       # 1 距离传感器
            rel_pos             # 3 相对位置
        ]).astype(np.float32)
        
        return obs

    def _check_collisions(self):
        """
        检测非法碰撞
        
        合法接触:
        - 机器人接触地面(floor)
        - 机器人接触货架(Shelf)
        - 机器人接触药盒(box_geom) - 抓取动作
        
        非法接触:
        - 机器人其他部位接触环境(除了上述情况)
        """
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1 = self.model.geom(c.geom1).name if c.geom1 >= 0 else None
            n2 = self.model.geom(c.geom2).name if c.geom2 >= 0 else None
            
            # 跳过无效的几何体
            if n1 is None or n2 is None:
                continue
                
            # 跳过地面接触(合法)
            if "floor" in n1.lower() or "floor" in n2.lower():
                continue
                
            # 跳过货架接触(合法 - 药盒可以接触货架)
            if "Shelf" in n1 or "Shelf" in n2:
                continue
                
            # 跳过药盒自身接触(合法)
            if "box_geom" in n1 or "box_geom" in n2:
                continue
                
            # 检查是否是机器人部件接触
            # 如果机器人碰到除了floor/shelf/box以外的东西,就是碰撞
            robot_parts = ['base', 'raise', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 
                          'h1', 'h2', 'l1', 'l2', 'l3', 'l4', 'l5', 'sucker']
            
            is_robot_1 = any(part in n1.lower() for part in robot_parts)
            is_robot_2 = any(part in n2.lower() for part in robot_parts)
            
            if is_robot_1 or is_robot_2:
                # 机器人接触到东西了,检查是否合法
                is_box_1 = "box" in n1.lower()
                is_box_2 = "box" in n2.lower()
                
                # 机器人碰到非药盒的东西 -> 碰撞
                if not (is_box_1 or is_box_2):
                    return True
                    
        return False

    def _get_rew(self, obs):
        """
        计算奖励:
        - 主要目标: 最小化sucker_tip到box_front_center的距离
        - 碰撞惩罚
        - 障碍物接近惩罚
        """
        # 提取相对位置(最后3个元素)
        rel_pos = obs[29:32]
        distance = np.linalg.norm(rel_pos)
        
        # 距离奖励(主要目标) - 使用较大权重加速学习
        reward = -distance * 20.0
        
        # 检查是否成功接触
        contact_dist = self._check_contact_with_box_front()
        if contact_dist < 0.02:  # 2cm以内视为接触
            reward += 100.0  # 成功接触的大奖励
            
        # 距离传感器惩罚(避障)
        d_suction = obs[28]
        if d_suction < 0.05:  # 离障碍物小于5cm
            reward -= 10.0 * (0.05 - d_suction)
                
        return reward

    def step(self, action):
        """
        执行一步仿真.
        
        参数:
            action: 活动关节的控制值 (raise + r1-r6, 共7个)
        
        返回:
            obs: 观测向量
            reward: 奖励值
            terminated: 是否终止
            truncated: 是否截断(总是False)
            info: 额外信息字典
        """
        # 执行仿真
        self.do_simulation(action, self.frame_skip)
        
        # 获取观测
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._get_rew(obs)
        
        # 检测碰撞
        collided = self._check_collisions()
        if collided:
            reward -= 150.0  # 碰撞的重罚
            
        # 检查终止条件
        contact_dist = self._check_contact_with_box_front()
        
        # 终止条件: 发生碰撞 或 成功接触(距离<1cm)
        terminated = collided or (contact_dist < 0.01)
        
        # 额外信息
        info = {
            "collided": collided,
            "contact_dist": contact_dist,
            "success": contact_dist < 0.01
        }
        
        return obs, reward, terminated, False, info

    def reset_model(self):
        """
        重置环境到初始状态
        
        将所有活动关节(position和velocity)设为0
        """
        init_qpos = self.data.qpos.copy()
        init_qvel = self.data.qvel.copy()
        
        # 将活动关节设为0
        for jid in self._joint_ids:
            if jid < len(init_qpos):
                init_qpos[jid] = 0.0
            if jid < len(init_qvel):
                init_qvel[jid] = 0.0
                
        self.set_state(init_qpos, init_qvel)
        return self._get_obs()
