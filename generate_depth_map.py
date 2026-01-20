#!/usr/bin/env python3
"""
LiDAR点云到图像深度图投影脚本

功能：
1. 读取LiDAR LAS文件
2. 读取图像文件
3. 读取相机内参和外参
4. 将LiDAR点云投影到图像平面
5. 生成深度图并保存

使用方法：
1. 打开脚本文件
2. 在main函数的"配置参数"部分修改文件路径
3. 运行脚本：python generate_depth_map.py
"""

import os
import numpy as np

# 设置JAX在CPU上运行
import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import jit, vmap
from PIL import Image
import laspy
import logging
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ========== 旋转矩阵计算 (来自fast_proj.py) ==========

def rot_x(o):
    """绕X轴旋转矩阵"""
    return jnp.array([
        [1, 0, 0],
        [0, jnp.cos(o), jnp.sin(o)],
        [0, -jnp.sin(o), jnp.cos(o)]])


def rot_y(p):
    """绕Y轴旋转矩阵"""
    return jnp.array([
        [jnp.cos(p), 0, -jnp.sin(p)],
        [0, 1, 0],
        [jnp.sin(p), 0, jnp.cos(p)]])


def rot_z(k):
    """绕Z轴旋转矩阵"""
    return jnp.array([
        [jnp.cos(k), jnp.sin(k), 0],
        [-jnp.sin(k), jnp.cos(k), 0],
        [0, 0, 1]])


def rot_zyx(o, p, k):
    """Z-Y-X顺序的复合旋转矩阵"""
    return rot_z(k) @ rot_y(p) @ rot_x(o)


# ========== 相机投影计算 (来自fast_proj.py) ==========

def rms_dict(_s):
    """将3D点从世界坐标系转换到相机坐标系"""
    R = rot_zyx(_s['omega'], _s['phi'], _s['kappa'])
    M = jnp.array([_s['X'], _s['Y'], _s['Z']])
    S = jnp.array([_s['Xs'], _s['Ys'], _s['Zs']])
    RMS = R @ (M - S)
    return RMS


def xy_frame(_s):
    """将相机坐标系中的3D点投影到归一化图像平面"""
    RMS = rms_dict(_s)
    m = - RMS / RMS[2]
    x, y, _ = m
    z = -RMS[2]
    return x, y, z


def corr_dist_agi(x, y, _s):
    """应用相机畸变校正并计算最终图像坐标"""
    rc = x ** 2 + y ** 2
    dr = 1 + _s['k1'] * rc + _s['k2'] * rc ** 2 + _s['k3'] * rc ** 3 + _s['k4'] * rc ** 4 + _s['k5'] * rc ** 5
    drx = x * dr
    dry = y * dr
    
    dtx = _s['p1'] * (rc + 2 * x ** 2) + 2 * _s['p2'] * x * y * (1 + _s['p3'] * rc + _s['p4'] * rc ** 2)
    dty = _s['p2'] * (rc + 2 * y ** 2) + 2 * _s['p1'] * x * y * (1 + _s['p3'] * rc + _s['p4'] * rc ** 2)
    xp = drx + dtx
    yp = dry + dty
    
    fx = _s['width'] * 0.5 + _s['cx'] + xp * _s['f'] + xp * _s['b1'] + yp * _s['b2']
    fy = _s['height'] * 0.5 + _s['cy'] + yp * _s['f']
    return fx, fy


def f_frame_agi(_s):
    """完整的Agisoft相机投影流程"""
    x, y, z = xy_frame(_s)
    w_2 = _s['width'] / _s['f'] / 2
    h_2 = _s['height'] / _s['f'] / 2
    ins = (x >= -w_2) & (x < w_2) & (y >= -h_2) & (y < h_2) & (z > 0)
    y = -y  # 调整y坐标以匹配Agisoft约定
    fx, fy = corr_dist_agi(x, y, _s)
    return fx, fy, z, ins


def compute_depth_map(i, j, z, depth_map, buffer_size=4, threshold=0.05):
    """计算深度图并进行可见性过滤"""
    height, width = depth_map.shape
    offsets = jnp.arange(-buffer_size, buffer_size + 1)
    di, dj = jnp.meshgrid(offsets, offsets, indexing='ij')
    di = di.ravel()
    dj = dj.ravel()
    
    neighbor_i = (i[:, None] + di).clip(0, width - 1)
    neighbor_j = (j[:, None] + dj).clip(0, height - 1)
    neighbor_depths = jnp.repeat(z[:, None], len(di), axis=1)
    
    neighbor_i = neighbor_i.ravel()
    neighbor_j = neighbor_j.ravel()
    neighbor_depths = neighbor_depths.ravel()
    
    depth_map = depth_map.at[neighbor_j, neighbor_i].min(neighbor_depths)
    visibility = jnp.abs(depth_map[j, i] - z) <= threshold
    
    return depth_map, visibility


# ========== 相机参数解析 (来自fast_proj.py) ==========

def parse_calibration_xml(file_path):
    """解析相机内参XML文件"""
    tree = ET.parse(file_path)
    root = tree.getroot()

    calibration_data = {
        'width': 0,
        'height': 0,
        'f': 0.0,
        'cx': 0.0,
        'cy': 0.0,
        'k1': 0.0,
        'k2': 0.0,
        'k3': 0.0,
        'k4': 0.0,
        'k5': 0.0,
        'p1': 0.0,
        'p2': 0.0,
        'p3': 0.0,
        'p4': 0.0,
        'b1': 0.0,
        'b2': 0.0,
    }

    for element in root:
        if element.tag in calibration_data:
            if element.tag in ['projection', 'date']:
                continue
            elif element.tag in ['width', 'height']:
                calibration_data[element.tag] = int(element.text)
            else:
                calibration_data[element.tag] = float(element.text)

    return calibration_data


def read_camera_pose(pose_file, image_name=None):
    """读取相机外参文件"""
    data = {}
    with open(pose_file, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            values = line.strip().split()
            if len(values) < 7:
                continue
            photo_id = values[0]
            
            # 读取必要的参数
            Xs = float(values[1])
            Ys = float(values[2])
            Zs = float(values[3])
            omega = np.radians(float(values[4]))
            phi = np.radians(float(values[5]))
            kappa = np.radians(float(values[6]))
            
            # 计算旋转矩阵
            # 绕X轴旋转omega
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(omega), -np.sin(omega)],
                [0, np.sin(omega), np.cos(omega)]
            ])
            
            # 绕Y轴旋转phi
            Ry = np.array([
                [np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)]
            ])
            
            # 绕Z轴旋转kappa
            Rz = np.array([
                [np.cos(kappa), -np.sin(kappa), 0],
                [np.sin(kappa), np.cos(kappa), 0],
                [0, 0, 1]
            ])
            
            # 计算复合旋转矩阵 R = Rz * Ry * Rx
            R = Rz @ Ry @ Rx
            
            data[photo_id] = {
                "Xs": Xs,
                "Ys": Ys,
                "Zs": Zs,
                "omega": omega,
                "phi": phi,
                "kappa": kappa,
                "R": R  # 保存旋转矩阵
            }
    
    # 如果指定了图像名称，返回对应相机参数
    if image_name:
        # 如果image_name是完整路径，则提取文件名（不带扩展名）
        if os.path.exists(image_name) or "./" in image_name or "/" in image_name or "\\" in image_name:
            image_id = os.path.splitext(os.path.basename(image_name))[0]
        else:
            # 否则直接使用image_name作为image_id
            image_id = image_name
            
        if image_id in data:
            logging.info(f"找到图像 {image_id} 的相机参数")
            return data[image_id]
        else:
            logging.warning(f"未找到图像 {image_id} 的相机参数，返回第一个相机的参数")
            return next(iter(data.values())) if data else None
    
    return data


# ========== 批量处理相关函数 ==========

def read_image_list(image_list_file):
    """
    读取image_list.json文件，获取所有照片信息
    
    参数：
        image_list_file: image_list.json文件路径
    
    返回：
        image_list: 照片信息列表
    """
    import json
    import os
    
    try:
        with open(image_list_file, 'r') as f:
            image_list = json.load(f)
        
        logging.info(f"成功读取image_list.json文件，共包含 {len(image_list)} 张照片")
        return image_list
    except Exception as e:
        logging.error(f"读取image_list.json文件失败：{e}")
        return []


# ========== 主函数 ==========

def lidar_to_depth_map(
    las_file, 
    image_file, 
    calib_file, 
    pose_file, 
    output_file,
    buffer_size=4,
    threshold=1.0,
    batch_size=1000000  # 调整为CPU适合的批处理大小
):
    """
    将LiDAR点云投影到图像平面，生成深度图
    
    参数：
        las_file: LAS文件路径
        image_file: 图像文件路径
        calib_file: 相机内参XML文件路径
        pose_file: 相机外参文件路径
        output_file: 深度图输出路径
        buffer_size: 深度图计算的缓冲区大小
        threshold: 可见性过滤的深度阈值
        batch_size: 批处理大小
    """
    logging.info(f"开始生成深度图")
    logging.info(f"LAS文件: {las_file}")
    logging.info(f"图像文件: {image_file}")
    logging.info(f"内参文件: {calib_file}")
    logging.info(f"外参文件: {pose_file}")
    logging.info(f"输出文件: {output_file}")
    
    # 1. 读取LiDAR文件
    logging.info(f"读取LiDAR文件...")
    las = laspy.read(las_file)
    X, Y, Z = las.x, las.y, las.z
    nb_pts = X.shape[0]
    logging.info(f"点云点数: {nb_pts:,}")
    
    # 2. 读取图像或使用image_list.json中的尺寸信息
    logging.info(f"读取图像或使用image_list.json中的尺寸信息...")
    image_width, image_height = None, None
    origin_filename = None  # 用于存储原始文件名，用于查找相机参数
    
    # 尝试从image_list.json中获取尺寸信息和原始文件名
    try:
        import json
        import os
        
        # 尝试查找image_list.json文件
        image_list_file = "image_list.json"
        if os.path.exists(image_list_file):
            with open(image_list_file, 'r') as f:
                image_list = json.load(f)
            
            # 查找当前图像
            current_image_id = os.path.splitext(os.path.basename(image_file))[0]
            for image_info in image_list:
                # 使用image_list中的id字段进行匹配
                if image_info["id"] == current_image_id:
                    image_width = image_info.get("width", 5280)  # 默认值
                    image_height = image_info.get("height", 3956)  # 默认值
                    # 获取原始文件名（不带扩展名）
                    origin_filename = os.path.splitext(os.path.basename(image_info["origin_path"]))[0]
                    logging.info(f"从image_list.json获取图像尺寸: {image_width}x{image_height}")
                    logging.info(f"从image_list.json获取原始文件名: {origin_filename}")
                    break
    except Exception as e:
        logging.warning(f"从image_list.json获取图像信息失败: {e}")
    
    # 如果没有从image_list.json获取到尺寸信息，尝试打开图像文件
    if image_width is None or image_height is None:
        try:
            image = Image.open(image_file)
            image_width, image_height = image.size
            logging.info(f"从图像文件获取尺寸: {image_width}x{image_height}")
        except Exception as e:
            logging.error(f"无法获取图像尺寸: {e}")
            return
    
    # 3. 读取相机参数
    logging.info(f"读取相机内参...")
    calib_data = parse_calibration_xml(calib_file)
    logging.info(f"读取相机外参...")
    
    # 使用原始文件名查找相机参数，如果没有则使用image_file
    camera_pose = read_camera_pose(pose_file, origin_filename)
    if not camera_pose:
        logging.error("无法读取相机外参")
        return
    
    # 4. 合并相机参数
    params_values = {
        **calib_data,
        **camera_pose
    }
    
    # 5. 坐标偏移，提高计算精度
    offset = [int(X[0]), int(Y[0]), int(Z[0])]
    X_offset = X - offset[0]
    Y_offset = Y - offset[1]
    Z_offset = Z - offset[2]
    params_values["Xs"] -= offset[0]
    params_values["Ys"] -= offset[1]
    params_values["Zs"] -= offset[2]
    
    # 6. 初始化深度图
    depth_map = jnp.full((image_height, image_width), jnp.inf)
    
    # 7. 批处理投影
    logging.info(f"开始投影计算...")
    f_proj = None
    
    for start in tqdm(range(0, nb_pts, batch_size), desc="处理点云批次"):
        end = min(start + batch_size, nb_pts)
        
        # 设置点云数据（在CPU上运行）
        params_values["X"] = X_offset[start:end]
        params_values["Y"] = Y_offset[start:end]
        params_values["Z"] = Z_offset[start:end]
        
        # 延迟初始化投影函数
        if f_proj is None:
            in_axes_dict = {key: None for key in params_values}
            for v in ['X', 'Y', 'Z']:
                in_axes_dict[v] = 0
            f_proj = jit(vmap(f_frame_agi, in_axes=(in_axes_dict,)))
        
        # 执行投影
        x, y, z, in_bounds = f_proj(params_values)
        
        # 转换为像素坐标
        i, j = x.astype(int), y.astype(int)
        
        # 只保留在图像边界内的点
        in_bounds = in_bounds & (i >= 0) & (i < image_width) & (j >= 0) & (j < image_height)
        
        if jnp.sum(in_bounds) == 0:
            continue
        
        i_in_bounds = i[in_bounds]
        j_in_bounds = j[in_bounds]
        z_in_bounds = z[in_bounds]
        
        # 计算深度图
        depth_map, _ = compute_depth_map(
            i_in_bounds, j_in_bounds, z_in_bounds, depth_map,
            buffer_size=buffer_size, threshold=threshold
        )
    
    # 8. 转换深度图格式
    logging.info(f"处理深度图...")
    depth_map_np = np.array(depth_map)
    
    # 将无穷大值替换为0
    depth_map_np[depth_map_np == np.inf] = 0
    
    # 确保数据类型为uint8并归一化
    if depth_map_np.dtype in [np.float32, np.float64, np.float16]:
        if np.max(depth_map_np) > 0:
            # 归一化到0-1范围
            depth_map_np = depth_map_np / np.max(depth_map_np)
            # 转换到0-255范围并转换为uint8
            depth_map_np = (depth_map_np * 255).astype(np.uint8)
        else:
            # 如果所有值都是0，直接转换为uint8
            depth_map_np = np.zeros_like(depth_map_np, dtype=np.uint8)
    elif depth_map_np.dtype != np.uint8:
        # 如果是其他非8位格式，转换为uint8
        depth_map_np = depth_map_np.astype(np.uint8)
    
    # 9. 保存深度图
    logging.info(f"保存深度图到: {output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"创建输出目录: {output_dir}")
    
    # 创建图像并保存为PNG
    depth_image = Image.fromarray(depth_map_np)
    # 确保图像模式为'L'（8位灰度）
    if depth_image.mode != 'L':
        depth_image = depth_image.convert('L')
    depth_image.save(output_file)
    
    logging.info("深度图生成完成！")


def batch_generate_depth_maps(
    las_file,
    image_list_file,
    calib_file,
    pose_file,
    output_dir,
    buffer_size=4,
    threshold=1.0,
    batch_size=1000000
):
    """
    批量生成深度图
    
    参数：
        las_file: LAS文件路径
        image_list_file: image_list.json文件路径
        calib_file: 相机内参XML文件路径
        pose_file: 相机外参文件路径
        output_dir: 深度图输出目录
        buffer_size: 深度图计算的缓冲区大小
        threshold: 可见性过滤的深度阈值
        batch_size: 批处理大小
    """
    import os
    
    # 1. 读取image_list.json文件
    image_list = read_image_list(image_list_file)
    if not image_list:
        logging.error("没有找到照片信息，批量处理失败")
        return
    
    # 2. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"创建输出目录: {output_dir}")
    
    # 3. 批量处理每张照片
    logging.info(f"开始批量生成深度图，共 {len(image_list)} 张照片")
    
    for i, image_info in enumerate(image_list):
        logging.info(f"\n处理第 {i+1}/{len(image_list)} 张照片")
        
        # 获取照片信息
        origin_filename = image_info["origin_path"].split("/")[-1]
        image_id = os.path.splitext(origin_filename)[0]
        
        # 即使照片文件不存在，也使用假的文件路径继续处理
        image_path = image_info["path"]
        
        logging.info(f"  照片路径: {image_path}")
        logging.info(f"  原始文件名: {origin_filename}")
        
        # 4. 生成深度图输出路径
        output_file = os.path.join(output_dir, f"{image_id}_depth_map.png")
        
        # 5. 生成深度图
        lidar_to_depth_map(
            las_file,
            image_path,  # 即使文件不存在，也传递路径以便从image_list.json获取尺寸
            calib_file,
            pose_file,
            output_file,
            buffer_size,
            threshold,
            batch_size
        )
    
    logging.info(f"\n批量生成深度图完成，共处理了 {len(image_list)} 张照片")


def main():
    """主函数"""
    # ==========================================
    # 配置参数 - 直接在此处修改文件路径
    # ==========================================
    # 处理模式：single（单张）或batch（批量）
    process_mode = "batch"  # single 或 batch
    
    # 通用参数
    las_file = "/Users/oldshen/Desktop/实训场测试/cloud2sxc.las"        # LiDAR LAS文件路径
    calib_file = "/Users/oldshen/Desktop/ImageVote_GridNet-HD_baseline_main/camera_calibration_generated.xml"    # 相机内参XML文件路径
    pose_file = "/Users/oldshen/Desktop/ImageVote_GridNet-HD_baseline_main/Cameras.txt"     # 相机外参文件路径
    
    # 可选参数
    buffer_size = 4       # 深度图计算的缓冲区大小
    threshold = 1.0       # 可见性过滤的深度阈值
    batch_size = 1000000  # 批处理大小（CPU环境建议：500,000-2,000,000）
    
    if process_mode == "single":
        # 单张处理参数
        image_file = "/Users/oldshen/Desktop/实训场测试/DJI_20250424164644_0002_D.JPG"           # 图像文件路径
        output_file = "depth_map.png"     # 深度图输出路径（当前目录）
        
        # 执行单张处理
        lidar_to_depth_map(
            las_file,
            image_file,
            calib_file,
            pose_file,
            output_file,
            buffer_size,
            threshold,
            batch_size
        )
    elif process_mode == "batch":
        # 批量处理参数
        image_list_file = "/Users/oldshen/Desktop/ImageVote_GridNet-HD_baseline_main/image_list.json" 
        image_file = "/Users/oldshen/Desktop/实训场测试/image" 
        output_dir = "/Users/oldshen/Desktop/ImageVote_GridNet-HD_baseline_main/depth_maps"  # 深度图输出目录
        
        # 执行批量处理
        batch_generate_depth_maps(
            las_file,
            image_list_file,
            calib_file,
            pose_file,
            output_dir,
            buffer_size,
            threshold,
            batch_size
        )
    else:
        logging.error(f"未知的处理模式：{process_mode}，请选择 'single' 或 'batch'")


if __name__ == '__main__':
    main()