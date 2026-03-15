import numpy as np
import mujoco

# 1. 获取侵入面的点和法向量
def get_entry_plane_data(data, model):
    # 点 P: site 的全局坐标
    point_p = data.site('entry_plane').xpos
    
    # 法向量 n: site 的旋转矩阵的第二列 (对应 Y 轴)
    # xmat 是 3x3 旋转矩阵，flatten 后是长度为 9 的数组
    rot_matrix = data.site('entry_plane').xmat.reshape(3, 3)
    normal_n = rot_matrix[:, 1] # 拿到该 site 的 Y 轴指向
    
    return point_p, normal_n

# 2. 获取上下层 Z 轴高度 (标量)
def get_layer_bounds(data):
    """获取上下层边界，如果没有货架则返回默认值"""
    try:
        z_upper = data.site('upper_limit').xpos[2]
        z_lower = data.site('lower_limit').xpos[2]
    except KeyError:
        # 没有货架时返回默认边界
        z_upper = 1.2  # 默认上层高度
        z_lower = 0.0   # 默认下层高度
    return z_upper, z_lower
def get_slam_style_dist(model, data, site_name, body_names_to_exclude, max_dist=0.5):
    """
    计算 Site 到场景中最近障碍物表面的欧式距离。
    body_names_to_exclude: 列表格式，例如 ["medicine_box", "suction_link"]
    """
    try:
        site_id = model.site(site_name).id
        site_pos = data.site_xpos[site_id]
        # 将传入的 body 名字全部转换为 id
        exclude_body_ids = [model.body(name).id for name in body_names_to_exclude]
    except KeyError as e:
        print(f"XML 中找不到对应的 site 或 body: {e}")
        return max_dist

    min_dist = max_dist
    
    # 使用 fromto 参数 (线段起点和终点)
    # 线段从 site_pos 到 site_pos (长度为0,即点)
    fromto = np.array([site_pos[0], site_pos[1], site_pos[2],
                       site_pos[0], site_pos[1], site_pos[2]], dtype=np.float64)
    
    for i in range(model.ngeom):
        geom_body_id = model.geom_bodyid[i]
        
        # 1. 过滤：如果这个几何体属于需要屏蔽的本体，直接跳过
        if geom_body_id in exclude_body_ids:
            continue
            
        # 2. 核心计算：计算点到几何体表面的距离
        # 新版本 mujoco 使用 fromto 参数
        dist = mujoco.mj_geomDistance(
            model, data, 
            geom1=i, geom2=-1, 
            distmax=min_dist, 
            fromto=fromto
        )
        
        if dist < min_dist:
            min_dist = max(0.0, dist)
            
    return min_dist

