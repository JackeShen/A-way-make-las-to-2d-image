import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import math
import os
from PIL import Image
import pyproj


def parse_xml_file(xml_file):
    """
    解析BlocksExchangeUndistortAT_WithoutTiePoints.xml文件
    提取所有Photo元素的ImagePath和对应OPK参数
    
    参数:
    xml_file: XML文件路径
    
    返回:
    photo_data: 字典，键为图像文件名或Photo的Id，值为包含OPK参数的字典
    """
    photo_data = {}
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 找到所有Photo元素
        for photo in root.findall('.//Photo'):
            # 提取Photo的Id
            photo_id = photo.find('Id').text
            
            # 提取ImagePath
            image_path = photo.find('ImagePath').text
            if not image_path:
                continue
                
            # 提取旋转参数
            pose = photo.find('Pose')
            rotation = pose.find('Rotation')
            
            # 提取OPK参数
            omega = float(rotation.find('Omega').text)
            phi = float(rotation.find('Phi').text)
            kappa = float(rotation.find('Kappa').text)
            
            # 应用坐标变换
            omega = omega - 180
            phi = -phi
            kappa = -kappa
            
            # 提取相机中心点坐标
            center = pose.find('Center')
            x = float(center.find('x').text)
            y = float(center.find('y').text)
            z = float(center.find('z').text)
            
            # 提取图像文件名（从完整路径中）
            # 处理不同操作系统的路径分隔符
            if '\\' in image_path:
                filename = image_path.split('\\')[-1]
            else:
                filename = os.path.basename(image_path)
            
            # 存储数据，使用文件名和Photo的Id作为键
            photo_data[filename] = {
                'omega': omega,
                'phi': phi,
                'kappa': kappa,
                'x': x,
                'y': y,
                'z': z
            }
            # 使用Photo的Id作为额外的键
            photo_data[f"photo_{photo_id}"] = {
                'omega': omega,
                'phi': phi,
                'kappa': kappa,
                'x': x,
                'y': y,
                'z': z
            }
            # 存储完整路径
            photo_data[f"path_{photo_id}"] = image_path
            
        print(f"成功从XML文件解析了 {len(photo_data)} 张照片的参数")
        
    except Exception as e:
        print(f"解析XML文件时出错: {e}")
        
    return photo_data


def get_sign_of(chifre):
    """
    获取数字的符号
    """
    if chifre >= 0:
        return 1
    else:
        return -1


def hrp2opk(roll, pitch, heading):
    """
    将DJI姿态角（Roll, Pitch, Yaw）转换为摄影测量OPK角度
    
    参数:
    roll: 滚动角（度）
    pitch: 俯仰角（度）
    heading: 航向角（度）
    
    返回:
    omega_deg, phi_deg, kappa_deg: 摄影测量OPK角度（度）
    """
    # 转换为弧度
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    heading_rad = np.deg2rad(heading)

    # 计算三角函数值
    A_SINH = np.sin(heading_rad)
    A_SINR = np.sin(roll_rad)
    A_SINP = np.sin(pitch_rad)

    A_COSH = np.cos(heading_rad)
    A_COSR = np.cos(roll_rad)
    A_COSP = np.cos(pitch_rad)

    # 构建旋转矩阵
    MX = np.zeros((3, 3))
    MX[0][0] = (A_COSH * A_COSR) + (A_SINH * A_SINP * A_SINR)
    MX[0][1] = (-A_SINH * A_COSR) + (A_COSH * A_SINP * A_SINR)
    MX[0][2] = -A_COSP * A_SINR

    MX[1][0] = A_SINH * A_COSP
    MX[1][1] = A_COSH * A_COSP
    MX[1][2] = A_SINP

    MX[2][0] = (A_COSH * A_SINR) - (A_SINH * A_SINP * A_COSR)
    MX[2][1] = (-A_SINH * A_SINR) - (A_COSH * A_SINP * A_COSR)
    MX[2][2] = A_COSP * A_COSR

    # 矩阵转置
    P = np.zeros((3, 3))
    P[0][0] = MX[0][0]
    P[0][1] = MX[1][0]
    P[0][2] = MX[2][0]
    
    P[1][0] = MX[0][1]
    P[1][1] = MX[1][1]
    P[1][2] = MX[2][1]
    
    P[2][0] = MX[2][0]
    P[2][1] = MX[1][2]
    P[2][2] = MX[2][2]

    # 计算欧拉角
    omega = np.arctan(-P[2][1] / P[2][2])
    phi = np.arcsin(P[2][2])
    kappa = np.arctan(-P[1][0] / P[0][0])

    # 修正符号处理
    phi = np.arcsin(P[2][0])  # 移除abs，直接使用arcsin的符号
    omega = np.arccos((P[2][2] / np.cos(phi)))
    omega = omega * (get_sign_of(P[2][1] / P[2][2]))  # 移除*-1
    kappa = np.arccos(P[0][0] / np.cos(phi))

    # 直接取反Kappa符号以匹配摄影测量标准
    kappa = -kappa

    # 转换为角度
    omega_deg = np.rad2deg(omega)
    phi_deg = np.rad2deg(-phi)  # 修正Phi符号
    kappa_deg = np.rad2deg(-kappa)  # 修正Kappa符号

    return omega_deg, phi_deg, kappa_deg


def opk2matrix(omega_deg, phi_deg, kappa_deg):
    """
    将OPK角度转换为旋转矩阵元素
    
    参数:
    omega_deg: Omega角度（度）
    phi_deg: Phi角度（度）
    kappa_deg: Kappa角度（度）
    
    返回:
    r11, r12, r13, r21, r22, r23, r31, r32, r33: 旋转矩阵元素
    """
    # 转换为弧度
    omega = math.radians(omega_deg)
    phi = math.radians(phi_deg)
    kappa = math.radians(kappa_deg)

    # 使用负角度以匹配正确的矩阵结果（摄影测量标准）
    # X轴旋转（Omega）使用负角度
    Rx_neg = np.array([
        [1, 0, 0],
        [0, math.cos(-omega), -math.sin(-omega)],
        [0, math.sin(-omega), math.cos(-omega)]
    ])

    # Y轴旋转（Phi）使用负角度
    Ry_neg = np.array([
        [math.cos(-phi), 0, math.sin(-phi)],
        [0, 1, 0],
        [-math.sin(-phi), 0, math.cos(-phi)]
    ])

    # Z轴旋转（Kappa）使用负角度
    Rz_neg = np.array([
        [math.cos(-kappa), -math.sin(-kappa), 0],
        [math.sin(-kappa), math.cos(-kappa), 0],
        [0, 0, 1]
    ])

    # 总旋转矩阵（Z-Y-X顺序，使用负角度）
    R = Rz_neg @ Ry_neg @ Rx_neg

    # 提取旋转矩阵元素
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    return r11, r12, r13, r21, r22, r23, r31, r32, r33


def process_image(jpg_file, xml_file=None, photo_id=None):
    """
    处理单张图像，提取元数据并生成Cameras.txt和相机内参XML
    
    参数:
    jpg_file: JPG图像文件路径
    xml_file: BlocksExchangeUndistortAT_WithoutTiePoints.xml文件路径（可选）
    photo_id: 要使用的Photo的Id（可选）
    """
    print(f"处理图像: {jpg_file}")
    
    # 图像文件名
    jpg_filename = os.path.basename(jpg_file)
    
    # 初始化OPK参数和相机位置
    omega_deg = None
    phi_deg = None
    kappa_deg = None
    camera_position = None
    
    # -----------------------------# 1. 从XML文件中提取OPK参数（如果提供）# -----------------------------
    if xml_file:
        print(f"从XML文件读取OPK参数: {xml_file}")
        photo_data = parse_xml_file(xml_file)
        
        # 优先使用photo_id匹配
        if photo_id is not None:
            photo_key = f"photo_{photo_id}"
            if photo_key in photo_data:
                data = photo_data[photo_key]
                omega_deg = data['omega']
                phi_deg = data['phi']
                kappa_deg = data['kappa']
                
                camera_position = np.array([data['x'], data['y'], data['z']])
                print(f"  从XML获取的OPK角度（度）: Omega={omega_deg}, Phi={phi_deg}, Kappa={kappa_deg}")
                # 如果有完整路径，打印出来
                path_key = f"path_{photo_id}"
                if path_key in photo_data:
                    print(f"  使用的XML图像路径: {photo_data[path_key]}")
        
        # 如果没有使用photo_id或者没有匹配到，尝试使用文件名匹配
        if omega_deg is None:
            # 查找当前图像对应的参数
            if jpg_filename in photo_data:
                data = photo_data[jpg_filename]
                omega_deg = data['omega']
                phi_deg = data['phi']
                kappa_deg = data['kappa']
                
                camera_position = np.array([data['x'], data['y'], data['z']])
                print(f"  从XML获取的OPK角度（度）: Omega={omega_deg}, Phi={phi_deg}, Kappa={kappa_deg}")
            else:
                # 如果找不到对应的文件名，尝试匹配没有扩展名的部分
                base_name = os.path.splitext(jpg_filename)[0]
                found = False
                for filename, data in photo_data.items():
                    if isinstance(data, dict) and 'omega' in data:
                        # 只检查值为字典且包含omega键的项
                        pass
                    elif base_name in filename:
                        # 检查键是否包含基础文件名
                        data = photo_data[filename]
                        if isinstance(data, dict) and 'omega' in data:
                            omega_deg = data['omega']
                            phi_deg = data['phi']
                            kappa_deg = data['kappa']
                            
                            camera_position = np.array([data['x'], data['y'], data['z']])
                            print(f"  从XML获取的OPK角度（度）: Omega={omega_deg}, Phi={phi_deg}, Kappa={kappa_deg}")
                            print(f"  匹配的XML文件名: {filename}")
                            found = True
                            break
                if not found:
                    print(f"警告：在XML文件中未找到图像 {jpg_filename} 对应的参数")
    # 使用PIL打开图像
    img = Image.open(jpg_file)
    
    # 初始化校准数据
    calibration_data = None
    
    # 如果没有从XML获取OPK参数，尝试从XMP元数据获取
    if omega_deg is None:
        print("未从XML获取OPK参数，尝试从XMP元数据获取")
        
        # 提取XMP元数据
        xmp_data = img.info.get('xmp')
        if not xmp_data:
            print("未找到XMP元数据")
            return

        # 将XMP数据解析为XML
        tree = ET.ElementTree(ET.fromstring(xmp_data))
        root = tree.getroot()

        # DJI 命名空间
        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'drone': 'http://www.dji.com/drone-dji/1.0/'
        }

        rdf_desc = root.find('.//rdf:RDF/rdf:Description', ns)

        # GPS/高度
        lat = float(rdf_desc.attrib['{http://www.dji.com/drone-dji/1.0/}GpsLatitude'])
        lon = float(rdf_desc.attrib['{http://www.dji.com/drone-dji/1.0/}GpsLongitude'])
        alt = float(rdf_desc.attrib['{http://www.dji.com/drone-dji/1.0/}AbsoluteAltitude'])

        # 飞行姿态
        flight_roll = float(rdf_desc.attrib['{http://www.dji.com/drone-dji/1.0/}FlightRollDegree'])
        flight_pitch = float(rdf_desc.attrib['{http://www.dji.com/drone-dji/1.0/}FlightPitchDegree'])
        flight_yaw = float(rdf_desc.attrib['{http://www.dji.com/drone-dji/1.0/}FlightYawDegree'])

        # 云台姿态
        gimbal_roll = float(rdf_desc.attrib['{http://www.dji.com/drone-dji/1.0/}GimbalRollDegree'])
        gimbal_pitch = float(rdf_desc.attrib['{http://www.dji.com/drone-dji/1.0/}GimbalPitchDegree'])
        gimbal_yaw = float(rdf_desc.attrib['{http://www.dji.com/drone-dji/1.0/}GimbalYawDegree'])

        # 调试输出：姿态参数
        print(f"提取的姿态参数：")
        print(f"  飞机姿态 - Roll: {flight_roll}, Pitch: {flight_pitch}, Yaw: {flight_yaw}")
        print(f"  云台姿态 - Roll: {gimbal_roll}, Pitch: {gimbal_pitch}, Yaw: {gimbal_yaw}")
        
        # 计算组合姿态
        total_roll = flight_roll + gimbal_roll
        total_pitch = flight_pitch + gimbal_pitch
        total_yaw = flight_yaw + gimbal_yaw
    
    # 打印组合姿态（仅当从XMP获取参数时）
    if 'total_roll' in locals():
        print(f"  组合姿态 - Roll: {total_roll}, Pitch: {total_pitch}, Yaw: {total_yaw}")

    # DewarpData（相机内参，可选）
    dewarp = ""
    if 'rdf_desc' in locals():
        dewarp = rdf_desc.attrib.get('{http://www.dji.com/drone-dji/1.0/}DewarpData', '')
    
    # 解析DewarpData（格式：日期;f_x,f_y,cx,cy,b1,k1,k2,k3）
    if dewarp:
        print(f"DewarpData: {dewarp}")
        dewarp_parts = dewarp.split(';')
        if len(dewarp_parts) == 2:
            date_str, params_str = dewarp_parts
            params = list(map(float, params_str.split(',')))
            if len(params) >= 9:
                # 从图像获取实际尺寸
                img_width, img_height = img.size
                calibration_data = {
                    'date': date_str,
                    'width': img_width,  # 使用图像实际宽度
                    'height': img_height,  # 使用图像实际高度
                    'f': params[0],  # 焦距
                    'cx': params[2],  # x方向主点偏移
                    'cy': params[3],  # y方向主点偏移
                    'b1': params[4],  # 相机倾斜参数
                    'k1': params[5],  # 径向畸变系数k1
                    'k2': params[6],  # 径向畸变系数k2
                    'k3': params[7],  # 径向畸变系数k3
                    'k4': params[8],  # 径向畸变系数k4
                    'p1': 0.0,        # 切向畸变系数p1（默认0）
                    'p2': 0.0         # 切向畸变系数p2（默认0）
                }
            else:
                print("DewarpData参数数量不足")
        else:
            print("DewarpData格式不正确")
    else:
        print("未找到DewarpData")

    # -----------------------------# 3. 坐标系转换（WGS84 -> UTM）# -----------------------------
    # 如果没有从XML获取相机位置，使用GPS坐标转换
    if camera_position is None:
        if 'lat' in locals() and 'lon' in locals() and 'alt' in locals():
            wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
            # 根据经度自动选择UTM zone
            utm_zone = int((lon + 180) / 6) + 1
            utm_proj = pyproj.Proj(proj="utm", zone=utm_zone, ellps="WGS84")
            x, y = pyproj.transform(wgs84, utm_proj, lon, lat)
            camera_position = np.array([x, y, alt])  # 相机世界坐标
        else:
            print("错误：无法获取相机位置")
            return

    # -----------------------------# 4. Omega/Phi/Kappa 计算# -----------------------------
    # 如果没有从XML获取OPK参数，计算OPK角度
    if omega_deg is None:
        if 'gimbal_roll' in locals() and 'gimbal_pitch' in locals() and 'gimbal_yaw' in locals():
            # 只使用云台姿态计算OPK角度
            omega_deg, phi_deg, kappa_deg = hrp2opk(gimbal_roll, gimbal_pitch, gimbal_yaw)
            print(f"  仅使用云台姿态计算OPK角度（度）: Omega={omega_deg}, Phi={phi_deg}, Kappa={kappa_deg}")
        else:
            print("错误：无法获取OPK参数")
            return

    # -----------------------------# 4. 构建旋转矩阵 r11~r33# -----------------------------
    # 使用opk2matrix函数计算旋转矩阵
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = opk2matrix(omega_deg, phi_deg, kappa_deg)

    # -----------------------------# 5. 输出 Cameras.txt# -----------------------------# 使用传入的photo_id参数或实际图像文件名作为PhotoID
    if photo_id is None:
        photo_id = os.path.splitext(os.path.basename(jpg_file))[0]
    else:
        # 如果传入的photo_id是数字索引，使用对应的图像文件名
        photo_id = os.path.splitext(os.path.basename(jpg_file))[0]

    # 创建DataFrame
    df = pd.DataFrame([[
        photo_id, camera_position[0], camera_position[1], camera_position[2],
        omega_deg, phi_deg, kappa_deg,
        r11, r12, r13,
        r21, r22, r23,
        r31, r32, r33
    ]], columns=[
        "PhotoID","X","Y","Z","Omega","Phi","Kappa",
        "r11","r12","r13","r21","r22","r23","r31","r32","r33"
    ])

    # 输出到Cameras.txt（追加模式）
    output_file = "Cameras.txt"
    with open(output_file, "a") as f:
        # 只写入数据行，不写入文件头
        df.to_csv(f, sep="\t", index=False, header=False)

    print(f"Cameras.txt 已生成: {output_file}")

    # -----------------------------# 6. 生成相机内参XML文件# -----------------------------
    # 生成相机内参XML文件
    calib_xml_file = "camera_calibration_generated.xml"
    if calibration_data:
        # 创建XML根元素
        root = ET.Element("calibration")
        
        # 添加投影类型
        projection_elem = ET.SubElement(root, "projection")
        projection_elem.text = "frame"
        
        # 添加图像尺寸
        width_elem = ET.SubElement(root, "width")
        width_elem.text = str(calibration_data['width'])
        
        height_elem = ET.SubElement(root, "height")
        height_elem.text = str(calibration_data['height'])
        
        # 添加焦距
        f_elem = ET.SubElement(root, "f")
        f_elem.text = str(calibration_data['f'])
        
        # 添加主点偏移
        cx_elem = ET.SubElement(root, "cx")
        cx_elem.text = str(calibration_data['cx'])
        
        cy_elem = ET.SubElement(root, "cy")
        cy_elem.text = str(calibration_data['cy'])
        
        # 添加倾斜参数
        b1_elem = ET.SubElement(root, "b1")
        b1_elem.text = str(calibration_data['b1'])
        
        # 添加畸变系数
        k1_elem = ET.SubElement(root, "k1")
        k1_elem.text = str(calibration_data['k1'])
        
        k2_elem = ET.SubElement(root, "k2")
        k2_elem.text = str(calibration_data['k2'])
        
        k3_elem = ET.SubElement(root, "k3")
        k3_elem.text = str(calibration_data['k3'])
        
        k4_elem = ET.SubElement(root, "k4")
        k4_elem.text = str(calibration_data['k4'])
        
        p1_elem = ET.SubElement(root, "p1")
        p1_elem.text = str(calibration_data['p1'])
        
        p2_elem = ET.SubElement(root, "p2")
        p2_elem.text = str(calibration_data['p2'])
        
        # 添加日期
        date_elem = ET.SubElement(root, "date")
        date_elem.text = calibration_data['date']
        
        # 美化XML格式
        import xml.dom.minidom
        rough_string = ET.tostring(root, encoding='utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ", encoding='UTF-8')
        
        # 写入文件
        with open(calib_xml_file, "wb") as f:
            f.write(pretty_xml)
        
        print(f"相机内参文件已生成: {calib_xml_file}")
    else:
        print("无法生成相机内参文件，缺少内参数据")


def batch_process_images(image_list_file, xml_file):
    """
    批量处理图像列表文件中的所有照片
    
    参数:
    image_list_file: image_list.json文件路径
    xml_file: BlocksExchangeUndistortAT_WithoutTiePoints.xml文件路径
    """
    import json
    import os
    
    # 读取image_list.json文件
    with open(image_list_file, 'r') as f:
        image_list = json.load(f)
    
    print(f"开始批量处理 {len(image_list)} 张照片")
    
    # 解析XML文件
    photo_data = parse_xml_file(xml_file)
    
    # 清空现有的Cameras.txt文件
    cameras_file = "Cameras.txt"
    with open(cameras_file, 'w') as f:
        f.write("# Cameras (0)\n")
        f.write("# PhotoID, X, Y, Z, Omega, Phi, Kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33\n")
    
    # 处理每张照片
    for i, image_info in enumerate(image_list):
        print(f"\n处理第 {i+1}/{len(image_list)} 张照片")
        
        # 获取照片信息
        photo_id = str(i)  # 使用索引作为photo_id
        image_id = image_info["id"]
        image_path = image_info["path"]
        origin_filename = image_info["origin_path"].split("/")[-1]
        
        print(f"  照片ID: {image_id}")
        print(f"  照片路径: {image_path}")
        print(f"  原始文件名: {origin_filename}")
        
        # 检查照片文件是否存在
        if not os.path.exists(image_path):
            print(f"  警告：照片文件 {image_path} 不存在，尝试跳过照片读取步骤")
            
            # 直接从XML获取OPK参数（已经在parse_xml_file中应用了变换）
            photo_key = f"photo_{photo_id}"
            if photo_key in photo_data:
                data = photo_data[photo_key]
                omega_deg = data['omega']
                phi_deg = data['phi']
                kappa_deg = data['kappa']
                
                camera_position = np.array([data['x'], data['y'], data['z']])
                
                print(f"  从XML获取的OPK角度（度）: Omega={omega_deg}, Phi={phi_deg}, Kappa={kappa_deg}")
                
                # 计算旋转矩阵
                r11, r12, r13, r21, r22, r23, r31, r32, r33 = opk2matrix(omega_deg, phi_deg, kappa_deg)
                
                # 使用原始文件名作为PhotoID
                photo_id_str = origin_filename.split('.')[0]
                
                # 创建DataFrame
                import pandas as pd
                df = pd.DataFrame([[
                    photo_id_str, camera_position[0], camera_position[1], camera_position[2],
                    omega_deg, phi_deg, kappa_deg,
                    r11, r12, r13,
                    r21, r22, r23,
                    r31, r32, r33
                ]], columns=[
                    "PhotoID","X","Y","Z","Omega","Phi","Kappa",
                    "r11","r12","r13","r21","r22","r23","r31","r32","r33"
                ])
                
                # 输出到Cameras.txt（追加模式）
                output_file = "Cameras.txt"
                with open(output_file, "a") as f:
                    df.to_csv(f, sep="\t", index=False, header=False)
                
                print(f"  Cameras.txt 已更新")
            else:
                print(f"  错误：在XML文件中未找到对应的OPK参数")
            continue
        
        # 处理照片
        process_image(image_path, xml_file, photo_id)
    
    print(f"\n批量处理完成，共处理了 {len(image_list)} 张照片")


if __name__ == "__main__":
    # 示例用法：批量处理所有照片
    image_list_file = "/Users/oldshen/Desktop/ImageVote_GridNet-HD_baseline_main/image_list.json"
    xml_file = "/Users/oldshen/Desktop/ImageVote_GridNet-HD_baseline_main/BlocksExchangeUndistortAT_WithoutTiePoints.xml"
    batch_process_images(image_list_file, xml_file)
    
    # 单张处理示例（注释掉）
    # jpg_file = "/Users/oldshen/Desktop/实训场测试/DJI_20250424164644_0002_D.JPG"
    # photo_id = "0"  # 使用第一个Photo元素的Id
    # process_image(jpg_file, xml_file, photo_id)