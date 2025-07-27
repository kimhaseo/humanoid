import numpy as np

# 회전 행렬 함수
def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def to_homogeneous(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

# 정방향 기구학
def forward_kinematics(theta1, theta2, theta3, theta4, theta5, d1, d2, L1, L2, L3):
    L1_total = d2 + L1
    T01 = to_homogeneous(rot_y(theta1), np.array([0, 0, d1]))
    T12 = to_homogeneous(rot_x(theta2), np.zeros(3))
    T23 = to_homogeneous(rot_z(theta3), np.array([0, 0, L1_total]))
    T34 = to_homogeneous(rot_y(theta4), np.array([0, 0, L2]))
    T45 = to_homogeneous(rot_y(theta5), np.array([0, 0, L3]))
    T05 = T01 @ T12 @ T23 @ T34 @ T45
    return T05

# 개선된 역기구학: theta3 먼저 계산
def inverse_kinematics(T06, d1, d2, L1, L2, L3):
    R06 = T06[:3,:3]
    P06 = T06[:3,3]

    # 1) 손목 위치 계산
    P_wc = P06 - R06[:,2] * L3
    x_wc, y_wc, z_wc = P_wc

    # 2) theta1: 어깨 Y축 회전 (앞뒤)
    theta1 = np.arctan2(x_wc, z_wc)

    # 3) theta3 계산을 위해 Y축 회전 제거
    R_y_inv = rot_y(-theta1)
    R1_6 = R_y_inv @ R06

    # 4) theta3: 어깨 Z축 회전 (좌우 비틀림)
    theta3 = np.arctan2(R1_6[1,0], R1_6[0,0])

    # 5) 보정된 손목 위치 계산 (theta3 Z축 역회전 + 어깨 높이 보정)
    R_z_inv = rot_z(-theta3)
    P_wc_corr = R_z_inv @ (P_wc - np.array([0, d1, 0]))
    x_corr, y_corr, z_corr = P_wc_corr

    # 6) r, y 계산 (팔꿈치 평면 거리)
    r = np.sqrt(x_corr**2 + z_corr**2)
    y = y_corr

    # 7) 팔꿈치 각도 theta4 계산 (코사인 법칙)
    L1_total = d2 + L1
    D = (L1_total**2 + L2**2 - (r**2 + y**2)) / (2 * L1_total * L2)
    D = np.clip(D, -1.0, 1.0)
    theta4 = np.arccos(D)

    # 8) 어깨 X축 회전 theta2 계산
    phi = np.arctan2(y, r)
    psi = np.arctan2(L2 * np.sin(theta4), L1_total + L2 * np.cos(theta4))
    theta2 = phi - psi

    # 9) T04 계산
    T01 = to_homogeneous(rot_y(theta1), np.array([0, 0, d1]))
    T12 = to_homogeneous(rot_x(theta2), np.zeros(3))
    T23 = to_homogeneous(rot_z(theta3), np.array([0, 0, L1_total]))
    T34 = to_homogeneous(rot_y(theta4), np.zeros(3))
    T04 = T01 @ T12 @ T23 @ T34
    R04 = T04[:3,:3]

    # 10) 팔꿈치 이후 회전 R46 계산
    R46 = R04.T @ R06

    # 11) 손목 회전 theta5 계산 (Y축 회전)
    theta5 = np.arctan2(-R46[2,0], R46[2,2])

    return theta1, theta2, theta3, theta4, theta5

# 테스트 예시
if __name__ == "__main__":
    px, py, pz = 100, 40, 70
    R_desired = rot_z(0) @ rot_y(0) @ rot_x(0)
    T06 = to_homogeneous(R_desired, np.array([px, py, pz]))

    d1 = 40
    d2 = 50
    L1 = 100
    L2 = 100
    L3 = 50

    angles = inverse_kinematics(T06, d1, d2, L1, L2, L3)
    print("관절 각도 (도):", np.degrees(angles))

    T05 = forward_kinematics(*angles, d1, d2, L1, L2, L3)
    print("계산된 엔드이펙터 위치:", T05[:3,3])
    print("목표 엔드이펙터 위치:", T06[:3,3])
