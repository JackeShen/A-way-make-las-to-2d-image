import numpy as np
import math

# 给定的OPK角度（度）- 使用更精确的数值
# 来自用户提供的正确数据
omega_deg = -30
phi_deg = -55
kappa_deg = -116

# 转换为弧度
omega = math.radians(omega_deg)
phi = math.radians(phi_deg)
kappa = math.radians(kappa_deg)

# 摄影测量标准：Z-Y-X旋转顺序（Kappa-Phi-Omega）
# 使用负角度以匹配正确的矩阵结果

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

# 输出结果
print(f"旋转矩阵元素：")
print(f"r11 = {r11}")
print(f"r12 = {r12}")
print(f"r13 = {r13}")
print(f"r21 = {r21}")
print(f"r22 = {r22}")
print(f"r23 = {r23}")
print(f"r31 = {r31}")
print(f"r32 = {r32}")
print(f"r33 = {r33}")

# 也可以直接打印整个旋转矩阵
print(f"\n完整旋转矩阵：")
print(R)