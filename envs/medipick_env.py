import os
import numpy as np
import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from .utils import get_slam_style_dist

class MediPickEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, model_path="../assets/robot.xml", **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # 维度计算: qpos(10) + qvel(10) + rel_pos(3) + 测距点(5) + 抓取状态(1) = 29
        observation_space = Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        MujocoEnv.__init__(
            self, os.path.abspath(model_path), 5,
            observation_space=observation_space, **kwargs
        )

    def _check_is_grasped(self):
        """ 判断药盒是否已经被吸盘成功吸附 """
        suction_pos = self.data.site("suction_site").xpos
        box_pos = self.data.body("medicine_box").xpos
        dist = np.linalg.norm(box_pos - suction_pos)
        return dist < 0.01

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        
        suction_pos = self.data.site("suction_site").xpos
        box_pos = self.data.body("medicine_box").xpos
        rel_pos = box_pos - suction_pos
        is_grasped = self._check_is_grasped()
        
        # 1. 机械臂本体全方位测距
        d_upper = get_slam_style_dist(self.model, self.data, "upper_arm_site", ["upper_arm_link"])
        d_elbow = get_slam_style_dist(self.model, self.data, "elbow_site", ["elbow_link"])
        d_forearm = get_slam_style_dist(self.model, self.data, "forearm_site", ["forearm_link"])
        d_suction = get_slam_style_dist(self.model, self.data, "suction_site", ["suction_link"])
        
        # 2. 动态药盒测距 (核心改动)
        if is_grasped:
            # 如果抓住了，药盒成为了手臂一部分，测距时必须同时屏蔽药盒自己和抓着它的吸盘
            d_box = get_slam_style_dist(
                self.model, self.data, 
                "box_center_site", # 需在药盒 body 里定义这个 site
                ["medicine_box", "suction_link"] 
            )
        else:
            # 没抓住时，药盒的测距不参与避障（返回安全最大值）
            d_box = 0.5 
            
        return np.concatenate([
            qpos, qvel, rel_pos, 
            [d_upper, d_elbow, d_forearm, d_suction, d_box, float(is_grasped)]
        ]).astype(np.float32)

    def _check_collisions(self, is_grasped):
        """ 动态接触检测：严格区分合法接触与非法碰撞 """
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1 = self.model.geom(c.geom1).name
            n2 = self.model.geom(c.geom2).name
            
            # 合法 1：吸盘与药盒的接触（这是抓取动作）
            if ("suction" in n1 and "box" in n2) or ("box" in n1 and "suction" in n2):
                continue
                
            # 合法 2：如果还没抓起来，药盒安静地躺在药架上是正常的
            if not is_grasped:
                if ("box" in n1 and "shelf" in n2) or ("shelf" in n1 and "box" in n2):
                    continue
                    
            # 只要走到这里，说明发生了绝对的非法碰撞（比如手臂撞墙、抓着药盒撞架子）
            return True
        return False

    def _get_rew(self, obs, is_grasped):
        rel_pos = obs[20:23]
        d_sensors = obs[23:28] # 提取 5 个测距器的值
        
        # 引导奖励
        reward = -np.linalg.norm(rel_pos) * 2.0
        if is_grasped:
            reward += 10.0 # 给予抓取成功的阶段性奖励
            
        # 空间压迫预警（全节点保护）
        for d in d_sensors:
            if d < 0.05: # 任何部位离障碍物小于 5cm，开始线性扣分
                reward -= 5.0 * (0.05 - d)
                
        return reward

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        is_grasped = self._check_is_grasped()
        obs = self._get_obs()
        reward = self._get_rew(obs, is_grasped)
        
        # 碰撞检测
        collided = self._check_collisions(is_grasped)
        if collided:
            reward -= 150.0 # 严厉惩罚
            
        # 终止条件：抓取成功后将药盒抬起（或者你定义的其他完成条件），或发生碰撞
        terminated = collided or (is_grasped and self.data.body("medicine_box").xpos[2] > 0.8)
        
        return obs, reward, terminated, False, {"collided": collided}